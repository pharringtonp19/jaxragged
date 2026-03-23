"""
ragged: A JAX transformation that makes functions automatically ignore
masked/padded elements in arrays.

Usage:
    import jaxragged as rag

    ma = rag.MaskedArray.from_ragged([[1, 2, 3], [4, 5]])

    def f(x):
        return jnp.mean(x)

    result = jit(vmap(rag(f)))(ma)
    # → [2.0, 4.5]  (correct means, ignoring padding)
"""

import functools
from dataclasses import dataclass

import jax
import jax._src.core as core
import jax.numpy as jnp
from jax import tree_util
from jax._src.lax import lax as lax_impl


# ============================================================
# MaskedArray — user-facing padded array type
# ============================================================


class MaskedArray:
    """An array paired with a boolean mask. True = real data, False = padding."""

    def __init__(self, data, mask):
        self.data = jnp.asarray(data)
        self.mask = jnp.asarray(mask, dtype=bool)

    @staticmethod
    def from_ragged(arrays):
        """Create a MaskedArray from a list of variable-length arrays."""
        max_len = max(len(a) for a in arrays)
        padded = []
        masks = []
        for a in arrays:
            a = jnp.asarray(a, dtype=jnp.float32)
            pad_len = max_len - len(a)
            padded.append(
                jnp.concatenate([a, jnp.zeros(pad_len, dtype=a.dtype)])
            )
            masks.append(
                jnp.concatenate(
                    [jnp.ones(len(a), dtype=bool), jnp.zeros(pad_len, dtype=bool)]
                )
            )
        return MaskedArray(jnp.stack(padded), jnp.stack(masks))

    def __repr__(self):
        return f"MaskedArray(data={self.data}, mask={self.mask})"


tree_util.register_pytree_node(
    MaskedArray,
    lambda ma: ((ma.data, ma.mask), None),
    lambda _, xs: MaskedArray(xs[0], xs[1]),
)


# ============================================================
# MaskedVal — tracks a value + mask through the jaxpr interpreter
# ============================================================


@dataclass
class MaskedVal:
    """A value paired with mask metadata as it flows through the interpreter."""

    val: jax.Array
    mask: jax.Array | None = None  # boolean, True = valid
    valid_count: jax.Array | None = None  # count of valid elements after reduction
    reduced_from_size: int | None = None  # original axis size before reduction


def _read(env, v):
    """Read a variable from the environment. Handles literals."""
    if isinstance(v, core.Literal):
        return MaskedVal(val=v.val)
    return env[v]


def _write(env, v, mval):
    """Write a MaskedVal to the environment."""
    env[v] = mval


# ============================================================
# Masking rules registry
# ============================================================

mask_rules: dict[core.Primitive, callable] = {}


def _register(prim):
    def decorator(fn):
        mask_rules[prim] = fn
        return fn
    return decorator


def _combine_masks(m1, m2):
    if m1 is not None and m2 is not None:
        return jnp.logical_and(m1, m2)
    return m1 if m1 is not None else m2


# --- Element-wise binary ops ---

def _binary_elementwise(prim):
    @_register(prim)
    def rule(inputs, **params):
        a, b = inputs
        result = prim.bind(a.val, b.val, **params)
        return MaskedVal(val=result, mask=_combine_masks(a.mask, b.mask))
    return rule


_binary_elementwise(lax_impl.add_p)
_binary_elementwise(lax_impl.sub_p)
_binary_elementwise(lax_impl.mul_p)
_binary_elementwise(lax_impl.max_p)
_binary_elementwise(lax_impl.min_p)


# --- Element-wise unary ops ---

def _unary_elementwise(prim):
    @_register(prim)
    def rule(inputs, **params):
        a = inputs[0]
        result = prim.bind(a.val, **params)
        return MaskedVal(val=result, mask=a.mask)
    return rule


_unary_elementwise(lax_impl.neg_p)
_unary_elementwise(lax_impl.abs_p)
_unary_elementwise(lax_impl.sqrt_p)
_unary_elementwise(lax_impl.integer_pow_p)

if hasattr(lax_impl, "square_p"):
    _unary_elementwise(lax_impl.square_p)


# --- Reductions ---
# Replace masked values with identity element, then reduce.
# Track valid_count for mean/std correction.

def _reduction_rule(prim, identity_fn):
    @_register(prim)
    def rule(inputs, **params):
        a = inputs[0]
        val = a.val
        mask = a.mask
        axes = params.get("axes", ())

        valid_count = None
        reduced_size = None

        if mask is not None:
            identity = identity_fn(val.dtype)
            val = jnp.where(mask, val, identity)
            if axes:
                valid_count = jnp.sum(mask.astype(val.dtype), axis=axes)
                reduced_size = mask.shape[axes[0]]

        result = prim.bind(val, **params)
        return MaskedVal(val=result, mask=None,
                         valid_count=valid_count, reduced_from_size=reduced_size)
    return rule


_reduction_rule(
    lax_impl.reduce_sum_p,
    lambda dtype: jnp.zeros((), dtype=dtype),
)
_reduction_rule(
    lax_impl.reduce_max_p,
    lambda dtype: (
        jnp.finfo(dtype).min if jnp.issubdtype(dtype, jnp.floating)
        else jnp.iinfo(dtype).min
    ),
)
_reduction_rule(
    lax_impl.reduce_min_p,
    lambda dtype: (
        jnp.finfo(dtype).max if jnp.issubdtype(dtype, jnp.floating)
        else jnp.iinfo(dtype).max
    ),
)
_reduction_rule(
    lax_impl.reduce_prod_p,
    lambda dtype: jnp.ones((), dtype=dtype),
)


# --- Division ---
# Detect "sum / N" pattern from jnp.mean and replace N with valid count.

@_register(lax_impl.div_p)
def _div_rule(inputs, **params):
    a, b = inputs

    lhs = a.val
    rhs = b.val

    if a.valid_count is not None and a.reduced_from_size is not None:
        rhs_is_size = jnp.equal(rhs, jnp.array(a.reduced_from_size, dtype=rhs.dtype))
        rhs = jnp.where(rhs_is_size, a.valid_count, rhs)

    result = lax_impl.div_p.bind(lhs, rhs, **params)
    return MaskedVal(val=result, mask=_combine_masks(a.mask, b.mask))


# --- Broadcast in dim ---

@_register(lax_impl.broadcast_in_dim_p)
def _broadcast_rule(inputs, **params):
    a = inputs[0]
    result = lax_impl.broadcast_in_dim_p.bind(a.val, **params)

    out_mask = None
    if a.mask is not None:
        out_mask = lax_impl.broadcast_in_dim_p.bind(a.mask, **params)

    return MaskedVal(val=result, mask=out_mask,
                     valid_count=a.valid_count, reduced_from_size=a.reduced_from_size)


# --- Select (jnp.where) ---

@_register(lax_impl.select_n_p)
def _select_rule(inputs, **params):
    vals = [inp.val for inp in inputs]
    result = lax_impl.select_n_p.bind(*vals, **params)
    out_mask = _combine_masks(inputs[1].mask, inputs[2].mask) if len(inputs) > 2 else inputs[1].mask
    return MaskedVal(val=result, mask=out_mask)


# --- Convert element type ---

@_register(lax_impl.convert_element_type_p)
def _convert_rule(inputs, **params):
    a = inputs[0]
    result = lax_impl.convert_element_type_p.bind(a.val, **params)
    return MaskedVal(val=result, mask=a.mask,
                     valid_count=a.valid_count, reduced_from_size=a.reduced_from_size)


# --- Comparison ops (used by std internals) ---

def _comparison_op(prim):
    @_register(prim)
    def rule(inputs, **params):
        a, b = inputs
        result = prim.bind(a.val, b.val, **params)
        return MaskedVal(val=result, mask=_combine_masks(a.mask, b.mask))
    return rule

for _p_name in ["gt_p", "ge_p", "lt_p", "le_p", "eq_p", "ne_p"]:
    if hasattr(lax_impl, _p_name):
        _comparison_op(getattr(lax_impl, _p_name))


# ============================================================
# Jaxpr interpreter — walks the flat primitive list
# ============================================================


def _eval_masked_closed_jaxpr(closed_jaxpr, in_mvals):
    """Evaluate a ClosedJaxpr with mask-aware interpretation.

    Args:
        closed_jaxpr: a jax ClosedJaxpr (has .jaxpr and .consts)
        in_mvals: list of MaskedVal inputs
    Returns:
        list of MaskedVal outputs
    """
    jaxpr = closed_jaxpr.jaxpr
    env = {}

    # Bind constants
    for var, const in zip(jaxpr.constvars, closed_jaxpr.consts):
        _write(env, var, MaskedVal(val=const))

    # Bind inputs
    for var, mval in zip(jaxpr.invars, in_mvals):
        _write(env, var, mval)

    # Walk through each equation (primitive application)
    for eqn in jaxpr.eqns:
        inputs = [_read(env, v) for v in eqn.invars]

        if eqn.primitive.name == "jit" and "jaxpr" in eqn.params:
            # Recursively interpret the inner jaxpr with mask awareness
            inner_jaxpr = eqn.params["jaxpr"]
            out_mvals = _eval_masked_closed_jaxpr(inner_jaxpr, inputs)
            if not isinstance(out_mvals, list):
                out_mvals = [out_mvals]
        elif eqn.primitive in mask_rules:
            out_mval = mask_rules[eqn.primitive](inputs, **eqn.params)
            out_mvals = [out_mval] if not isinstance(out_mval, list) else out_mval
        else:
            # Fallback: run primitive normally, propagate first mask found
            vals = [inp.val for inp in inputs]
            result = eqn.primitive.bind(*vals, **eqn.params)
            any_mask = next((inp.mask for inp in inputs if inp.mask is not None), None)

            if eqn.primitive.multiple_results:
                out_mvals = [MaskedVal(val=r, mask=any_mask) for r in result]
            else:
                out_mvals = [MaskedVal(val=result, mask=any_mask)]

        # Write outputs to environment
        for var, mv in zip(eqn.outvars, out_mvals):
            _write(env, var, mv)

    # Read outputs
    return [_read(env, v) for v in jaxpr.outvars]


def _eval_masked_jaxpr(jaxpr, consts, data, mask):
    """Entry point: evaluate a jaxpr with a single masked input."""
    in_mvals = [MaskedVal(val=data, mask=mask)]
    out_mvals = _eval_masked_closed_jaxpr(jaxpr, in_mvals)
    if len(out_mvals) == 1:
        return out_mvals[0].val
    return tuple(mv.val for mv in out_mvals)


# ============================================================
# ragged() — the public transformation API
# ============================================================


def ragged(f):
    """Transform f to automatically ignore masked/padded elements.

    Usage:
        ma = MaskedArray(data, mask)

        result = ragged(f)(ma)
        # Equivalent to f applied only to the real elements

        # Composes with jit and vmap:
        result = jit(vmap(ragged(f)))(batch_of_masked_arrays)
    """

    @functools.wraps(f)
    def wrapped(masked_array):
        if isinstance(masked_array, MaskedArray):
            data, mask = masked_array.data, masked_array.mask
        else:
            return f(masked_array)

        # Step 1: trace f to get a flat jaxpr (no jit wrappers, just primitives)
        jaxpr = jax.make_jaxpr(f)(data)

        # Step 2: evaluate the jaxpr with mask-aware interpretation
        return _eval_masked_jaxpr(jaxpr, jaxpr.consts, data, mask)

    return wrapped

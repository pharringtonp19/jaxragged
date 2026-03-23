# JAX Ragged

A [Jaxpr interpreter](https://docs.jax.dev/en/latest/notebooks/Writing_custom_interpreters_in_Jax.html) that makes JAX functions automatically ignore padded elements in ragged arrays.

```python
import jax
import jax.numpy as jnp
import jaxragged as rag

ma = rag.MaskedArray.from_ragged([[1, 2, 3], [4, 5]])

result = jax.vmap(rag(jnp.mean))(ma)
# → [2.0, 4.5]  (correct means, ignoring padding)
```

## Motivation

JAX requires all arrays to have the same shape. When working with variable-length data, users must pad arrays to a uniform size. This padding corrupts results for standard operations:

```python
jnp.mean([4, 5, 0, 0])  # → 2.25 instead of 4.5
jnp.min([4, 5, 0, 0])   # → 0 instead of 4
jnp.max([-3, -1, 0, 0]) # → 0 instead of -1
```

The typical workaround is manually threading masks through every function — error-prone and tedious.

`jaxragged` solves this: write normal JAX code, wrap it with `ragged()`, and padding is handled transparently.

## How It Works

`ragged(f)` uses `jax.make_jaxpr` to trace your function into a flat list of JAX primitives, then walks through and evaluates each one with mask-aware rules:

- **Reductions** (`sum`, `max`, `min`, `prod`): masked values are replaced with identity elements (0 for sum, -inf for max, etc.)
- **Mean / Std**: the "divide by N" pattern is detected and N is replaced with the count of valid elements
- **Element-wise ops**: run normally, masks are propagated
- **Nested `jit` expressions**: recursively interpreted with mask awareness

This is the same interpreter pattern used by JAX's own `vmap` and `grad` transformations.

## Installation

```
pip install jaxragged
```

## Usage

### Basic

```python
import jax.numpy as jnp
import jaxragged as rag

data = jnp.array([4.0, 5.0, 0.0, 0.0])
mask = jnp.array([True, True, False, False])
ma = rag.MaskedArray(data, mask)

rag(jnp.mean)(ma)  # → 4.5
rag(jnp.min)(ma)   # → 4.0
rag(jnp.max)(ma)   # → 5.0
rag(jnp.std)(ma)   # → 0.5
```

### From ragged lists

```python
ma = rag.MaskedArray.from_ragged([[1, 2, 3], [4, 5], [6]])
# Automatically pads to uniform length and creates the mask
```

### Custom functions

```python
def my_func(x):
    return jnp.sum(x ** 2) / jnp.sum(x)

rag(my_func)(ma)  # padding ignored throughout
```

### Composing with `jit` and `vmap`

```python
import jax

ma = rag.MaskedArray.from_ragged([[1, 2, 3], [4, 5]])

# vmap over a batch of ragged arrays
jax.vmap(rag(jnp.mean))(ma)  # → [2.0, 4.5]

# jit for performance
jax.jit(jax.vmap(rag(jnp.mean)))(ma)  # → [2.0, 4.5]
```

## Supported Operations

| Category | Operations |
|---|---|
| Reductions | `sum`, `max`, `min`, `mean`, `std`, `var`, `prod` |
| Arithmetic | `+`, `-`, `*`, `/`, `**`, `abs`, `neg`, `sqrt` |
| Comparison | `>`, `>=`, `<`, `<=`, `==`, `!=` |
| Other | `broadcast_in_dim`, `convert_element_type`, `select_n` |

Unsupported primitives fall back to normal execution with best-effort mask propagation.

## Acknowledgements

Architecture inspired by [jaxpurify](https://github.com/dodgebc/jaxpurify).

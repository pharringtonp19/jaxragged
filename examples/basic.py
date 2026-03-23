"""Basic usage of jaxragged."""

import jax
import jax.numpy as jnp

import jaxragged as jr

# Create a batch of ragged arrays
ma = jr.MaskedArray.from_ragged([[1, 2, 3], [4, 5], [6, 7, 8, 9]])
print(f"data:\n{ma.data}\n")
print(f"mask:\n{ma.mask}\n")

# Without ragged: padding corrupts results
print("Without ragged (wrong):")
print(f"  mean per row: {jnp.mean(ma.data, axis=1)}")
print(f"  min per row:  {jnp.min(ma.data, axis=1)}")
print()

# With ragged: padding is ignored
print("With ragged (correct):")
print(f"  mean per row: {jax.vmap(jr.ragged(jnp.mean))(ma)}")
print(f"  min per row:  {jax.vmap(jr.ragged(jnp.min))(ma)}")
print(f"  max per row:  {jax.vmap(jr.ragged(jnp.max))(ma)}")
print(f"  sum per row:  {jax.vmap(jr.ragged(jnp.sum))(ma)}")
print(f"  std per row:  {jax.vmap(jr.ragged(jnp.std))(ma)}")
print()

# Custom functions work too
def rms(x):
    return jnp.sqrt(jnp.mean(x ** 2))

print(f"  rms per row:  {jax.vmap(jr.ragged(rms))(ma)}")
print()

# Composes with jit
jitted = jax.jit(jax.vmap(jr.ragged(jnp.mean)))
print(f"  jit(vmap(ragged(mean))): {jitted(ma)}")

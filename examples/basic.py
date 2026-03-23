"""Basic usage of jaxragged."""

import jax
import jax.numpy as jnp

import jaxragged as rag

# Create a batch of ragged arrays
ma = rag.MaskedArray.from_ragged([[1, 2, 3], [4, 5], [6, 7, 8, 9]])
print(f"data:\n{ma.data}\n")
print(f"mask:\n{ma.mask}\n")

# Without ragged: padding corrupts results
print("Without ragged (wrong):")
print(f"  mean per row: {jnp.mean(ma.data, axis=1)}")
print(f"  min per row:  {jnp.min(ma.data, axis=1)}")
print()

# With ragged: padding is ignored
print("With ragged (correct):")
print(f"  mean per row: {rag(jnp.mean)(ma)}")
print(f"  min per row:  {rag(jnp.min)(ma)}")
print(f"  max per row:  {rag(jnp.max)(ma)}")
print(f"  sum per row:  {rag(jnp.sum)(ma)}")
print(f"  std per row:  {rag(jnp.std)(ma)}")
print()

# Custom functions work too
def rms(x):
    return jnp.sqrt(jnp.mean(x ** 2))

print(f"  rms per row:  {rag(rms)(ma)}")
print()

# Composes with jit
jitted = jax.jit(rag(jnp.mean))
print(f"  jit(rag(mean)): {jitted(ma)}")

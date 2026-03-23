"""jaxragged: Transparent masking for ragged arrays in JAX."""

import sys
import types

from jaxragged.ragged import MaskedArray, ragged

__all__ = ["MaskedArray", "ragged"]


class _Module(types.ModuleType):
    """Makes `import jaxragged as rag; rag(f)` work."""

    def __call__(self, f):
        return ragged(f)


# Replace this module with our callable version
_mod = _Module(__name__)
_mod.__dict__.update(
    {k: v for k, v in globals().items() if not k.startswith("_mod")}
)
_mod.__file__ = __file__
_mod.__spec__ = __spec__
sys.modules[__name__] = _mod

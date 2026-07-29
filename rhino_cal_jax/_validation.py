"""Shared structural guards.

Private. These enforce the package's central rule -- a failure mode that
produces a finite, correctly-shaped, WRONG answer must raise -- and are shared
so that the wording and the behaviour cannot drift between modules.
"""

import jax
import jax.numpy as jnp

from rhino_cal_jax.errors import ValidationError


def require_complex(name: str, value: jax.Array) -> jax.Array:
    """Reject a real reflection coefficient, returning it converted.

    A real ``Gamma`` makes the quadrature coupling ``Im(Gamma F)`` identically
    zero, so ``T_sin`` silently drops out of the model: a finite,
    correctly-shaped, wrong answer.
    """
    value = jnp.asarray(value)
    if not jnp.issubdtype(value.dtype, jnp.complexfloating):
        raise ValidationError(
            f"{name} must be complex (got dtype {value.dtype}). A real reflection "
            "coefficient silently zeroes the sine coupling; pass e.g. "
            f"{name} + 0j if it really is purely real."
        )
    return value

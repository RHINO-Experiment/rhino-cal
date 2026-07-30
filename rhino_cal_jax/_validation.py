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


def require_coupling_columns(name: str, stacked: jax.Array) -> jax.Array:
    """Reject a ``stacked`` array without a trailing axis of the four couplings.

    Shared by :meth:`~rhino_cal_jax.reflection.Couplings.from_stacked` and
    :func:`~rhino_cal_jax.power.design_matrix`, which expect the identical
    ``(..., n_freq, 4)`` shape, so the wording and the behaviour cannot drift
    apart between them.
    """
    stacked = jnp.asarray(stacked)
    if stacked.ndim < 2 or stacked.shape[-1] != 4:
        raise ValidationError(
            f"{name} expects a trailing axis of 4 couplings, got shape "
            f"{stacked.shape}."
        )
    return stacked

"""A synchrotron power law standing in for the antenna temperature.

This is not a sky model in any serious sense -- it is the smooth, bright,
spectrally simple thing you point the calibration machinery at while testing it.
Real sky models belong upstream (rheplicant's sky engines, limTOD).
"""

import jax
import jax.numpy as jnp

from rhino_cal_jax.errors import ValidationError


def _require_scalar_or_matching_freq(name: str, value: jax.Array, freq: jax.Array) -> None:
    """Reject a ``value`` that is neither scalar nor shaped like ``freq``.

    NumPy broadcasting would happily stretch a length-1 array across the whole
    band (one value applied to every channel), or broadcast a
    differently-shaped array against ``freq`` as an outer product instead of a
    per-channel value -- both finite, correctly shaped by NumPy's rules, and
    wrong. Shape-only, so it stays safe under ``jit``.
    """
    if value.ndim != 0 and value.shape != freq.shape:
        raise ValidationError(
            f"{name} must be scalar or shaped like freq {freq.shape}, got "
            f"{value.shape}. A length-1 {name} would broadcast silently across "
            f"the whole band, and any other mismatched shape would form an "
            f"outer product with freq instead of one value per channel."
        )


def synchrotron_temperature(
    freq: jax.Array,
    *,
    t_ref: float | jax.Array = 180.0,
    beta: float | jax.Array = -2.6,
    freq_ref: float | jax.Array = 210e6,
) -> jax.Array:
    """``T(nu) = t_ref (nu / freq_ref) ** beta``.

    Args:
        freq: channel frequencies [Hz].
        t_ref: brightness temperature at ``freq_ref`` [K]; scalar or shaped
            like ``freq``.
        beta: spectral index (negative for synchrotron); scalar or shaped
            like ``freq``.
        freq_ref: reference frequency [Hz]; scalar or shaped like ``freq``.

    Returns:
        Brightness temperature [K], shaped like ``freq``. Differentiable in
        ``t_ref`` and ``beta``, which is what makes it usable as a fitted
        foreground rather than a fixed backdrop.

    Raises:
        ValidationError: if ``t_ref``, ``beta`` or ``freq_ref`` is neither a
            scalar nor exactly ``freq``-shaped.
    """
    freq = jnp.asarray(freq)
    t_ref = jnp.asarray(t_ref)
    beta = jnp.asarray(beta)
    freq_ref = jnp.asarray(freq_ref)
    _require_scalar_or_matching_freq("t_ref", t_ref, freq)
    _require_scalar_or_matching_freq("beta", beta, freq)
    _require_scalar_or_matching_freq("freq_ref", freq_ref, freq)
    return t_ref * jnp.power(freq / freq_ref, beta)

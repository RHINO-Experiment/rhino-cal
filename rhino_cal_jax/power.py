"""Draft Eqs. 1 and 11: system temperature and the recorded power.

The bracket of Eq. 1 is exactly linear in the temperature vector
``(T_src, T_unc, T_cos, T_sin, T_rx)``. That is not an approximation to be
checked at run time -- it is the structure the GCR sampler is built on -- so
:func:`system_temperature` is written as a contraction against the coupling
spectra rather than as five hand-written products.
"""

import jax
import jax.numpy as jnp

from rhino_cal_jax.errors import ValidationError
from rhino_cal_jax.reflection import Couplings


def system_temperature(
    coup: Couplings,
    *,
    t_src: jax.Array | float,
    t_unc: jax.Array | float,
    t_cos: jax.Array | float,
    t_sin: jax.Array | float,
    t_rx: jax.Array | float,
) -> jax.Array:
    """``T_sys`` -- the bracket of draft Eq. 1 (equivalently Eq. 11).

    Args:
        coup: coupling spectra for the connected source. Fields are
            ``(n_freq,)``, or ``(n_time, n_freq)`` once gathered through a
            switch cycle.
        t_src: source noise temperature [K].
        t_unc: uncorrelated noise-wave temperature [K].
        t_cos: in-phase noise-wave temperature [K].
        t_sin: quadrature noise-wave temperature [K].
        t_rx: receiver offset temperature [K] (the draft's ``T_rx``; the numpy
            reference calls it ``t_0``). It has no coupling -- it enters the
            bracket bare.

    Every temperature broadcasts against the coupling shape, so a scalar, a
    ``(n_freq,)`` spectrum and a ``(n_time, n_freq)`` field are all accepted.

    **Convention for time variation.** A bare 1-D temperature is always read as
    per-*frequency*. To vary a temperature with time instead, pass an explicit
    ``(n_time, 1)`` column. This is not a limitation that could be checked away:
    against a ``(n_time, n_freq)`` coupling a bare ``(n_time,)`` array is
    indistinguishable in shape from the legitimate and far more common
    ``(n_freq,)`` spectrum, so no runtime guard can tell a per-time vector from
    a per-frequency one when ``n_time == n_freq``. It would broadcast along
    frequency and return a finite, correctly-shaped, wrong ``T_sys``. Lengths
    that match neither axis already raise, so the column form is the only thing
    a caller has to remember.

    Returns:
        ``T_sys`` with the broadcast shape.
    """
    return (
        t_src * coup.c_src
        + t_unc * coup.k_unc
        + t_cos * coup.k_cos
        + t_sin * coup.k_sin
        + jnp.asarray(t_rx)
    )


def radiometer_power(t_sys: jax.Array, gain: jax.Array | float) -> jax.Array:
    """``d = G T_sys`` -- draft Eq. 1 without the noise term.

    Args:
        t_sys: system temperature [K], from :func:`system_temperature`.
        gain: ``G(nu, t)`` [power per kelvin]; scalar, ``(n_freq,)``, or
            ``(n_time, n_freq)``.

    Returns:
        Recorded spectral power, broadcast to the joint shape.
    """
    return jnp.asarray(gain) * t_sys


def design_matrix(stacked: jax.Array) -> jax.Array:
    """Flatten stacked couplings into the ``(n_row, 4)`` matrix of the linear model.

    Args:
        stacked: ``(..., n_freq, 4)`` from
            :attr:`~rhino_cal_jax.reflection.Couplings.stacked`, optionally
            gathered onto a time axis by
            :meth:`~rhino_cal_jax.switching.SwitchCycle.gather`.

    Returns:
        ``(prod(leading dims) * n_freq, 4)``. Multiplying by
        ``(T_src, T_unc, T_cos, T_sin)`` reproduces ``T_sys - T_rx`` flattened in
        C order, which is what the GCR of draft Eqs. 30-31 solves.

    **Rows are per leading element, not per observed sample.** Nothing here
    checks whether the leading (``...``) axes of ``stacked`` are per-*source*
    or per-*time*. ``design_matrix(couplings(...).stacked)`` (one row set per
    source) and ``design_matrix(cycle.gather(couplings(...).stacked))`` (one
    row set per observed sample, via
    :meth:`~rhino_cal_jax.switching.SwitchCycle.gather`) both return a
    ``(n_row, 4)`` array of the identical shape whenever ``n_time ==
    n_source`` -- exactly the minimal three-calibrator setup this package's
    README shows -- but with different rows unless the switch order happens to
    be the identity. Skipping :meth:`~rhino_cal_jax.switching.SwitchCycle.gather`
    does not raise; it silently changes what each row means.

    Raises:
        ValidationError: if the trailing axis is not the four couplings.
    """
    stacked = jnp.asarray(stacked)
    if stacked.ndim < 2 or stacked.shape[-1] != 4:
        raise ValidationError(
            f"design_matrix expects a trailing axis of 4 couplings, got shape "
            f"{stacked.shape}."
        )
    return stacked.reshape(-1, 4)


def add_radiometer_noise(
    power: jax.Array,
    key: jax.Array,
    *,
    t_int: float | jax.Array,
    delta_nu: float | jax.Array,
    fold_negative: bool = False,
) -> jax.Array:
    """Apply the fractional radiometer noise of draft Eq. 8.

    ``d -> d (1 + w)`` with ``w ~ N(0, sigma_w)`` and
    ``sigma_w = 1 / sqrt(delta_nu * t_int)`` -- multiplicative, so the absolute
    scatter tracks the power.

    Args:
        power: noiseless power from :func:`radiometer_power`.
        key: a typed JAX PRNG key (``jax.random.key(seed)``).
        t_int: integration time per sample [s].
        delta_nu: channel bandwidth [Hz].
        fold_negative: reproduce the numpy reference's ``abs(P + n)``. **Off by
            default and to be left off in any scientific use** -- folding
            reflects the negative tail and biases the mean upward whenever
            ``delta_nu * t_int`` is not large (at ``sigma_w = 1`` the mean of a
            unit-power signal rises to ``E|N(1,1)| = 1.167``), which silently
            breaks the Gaussian likelihood the GCR assumes. It exists so the
            consistency suite can reproduce the reference bit for bit.

    Returns:
        The noisy power, same shape as ``power``.

    Like :func:`system_temperature`, a bare 1-D ``t_int`` or ``delta_nu`` is read
    as per-*frequency*; pass an explicit ``(n_time, 1)`` column to vary either
    with time. The same argument applies -- against a ``(n_time, n_freq)`` power
    array a bare ``(n_time,)`` vector is indistinguishable in shape from a
    per-channel one, so no guard can tell them apart when ``n_time == n_freq``.
    In practice both are scalars: integration time and channel bandwidth are
    fixed for a run.

    Note:
        Draft Eq. 1 writes the noise as an additive ``n_w`` inside the bracket
        while Eq. 8 writes it as fractional; the two agree only for
        ``n_w = T_sys w``. Eq. 8 is what the radiometer equation means and what
        the numpy reference implements, so it is what this function does.
    """
    power = jnp.asarray(power)
    sigma_w = 1.0 / jnp.sqrt(jnp.asarray(delta_nu) * jnp.asarray(t_int))
    noisy = power * (1.0 + sigma_w * jax.random.normal(key, power.shape, power.dtype))
    return jnp.abs(noisy) if fold_negative else noisy

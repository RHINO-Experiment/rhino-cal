"""Draft Eqs. 1 and 11: system temperature and the recorded power.

The bracket of Eq. 1 is exactly linear in the temperature vector
``(T_src, T_unc, T_cos, T_sin, T_rx)``. That is not an approximation to be
checked at run time -- it is the structure the GCR sampler is built on -- so
:func:`system_temperature` is written as a contraction against the coupling
spectra rather than as five hand-written products.
"""

import jax
import jax.numpy as jnp

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

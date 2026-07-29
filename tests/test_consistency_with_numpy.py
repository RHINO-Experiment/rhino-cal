"""Cross-check rhino_cal_jax against the numpy implementation in simulation/.

The reference is `simulation.radiometer_power.compute_radiometer_power`, the
function every numpy notebook in this repository ultimately calls. Agreement is
demanded at float64 round-off over a grid that includes the extremes, because
the failure mode that matters here is a *finite, correctly-shaped, wrong*
number -- the kind a spot check at one nice parameter value cannot see.
"""

import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from rhino_cal_jax.power import radiometer_power, system_temperature
from rhino_cal_jax.reflection import couplings
from simulation.radiometer_power import compute_radiometer_power

# Extremes on purpose: a matched source (Gamma = 0) zeroes three of the four
# couplings, and |Gamma| -> 1 is where F is most sensitive to the receiver.
MAGS = (0.0, 0.05, 0.5, 0.95)
PHASES = (0.0, np.pi / 3, np.pi, -2.0)
TEMPS = (
    (300.0, 250.0, 30.0, -40.0, 290.0),   # a realistic set
    (0.0, 0.0, 0.0, 0.0, 0.0),            # everything off
    (1200.0, 0.0, 0.0, 0.0, 0.0),         # noise-diode-only
    (10.0, 5000.0, -900.0, 900.0, 1.0),   # noise waves dominating
)


def _numpy_power(t, g_src, g_rec, gain):
    """The reference. Keyword args throughout: its positional order is
    (t_src, t_unc, t_sin, t_cos, t_0) -- sin BEFORE cos, which is easy to
    transpose by accident and would silently swap two couplings."""
    t_src, t_unc, t_cos, t_sin, t_rx = t
    return compute_radiometer_power(
        t_src=t_src, t_unc=t_unc, t_sin=t_sin, t_cos=t_cos, t_0=t_rx,
        gamma_rec=g_rec, gamma_src=g_src, gain=gain, add_noise=False,
    )


def _jax_power(t, g_src, g_rec, gain):
    t_src, t_unc, t_cos, t_sin, t_rx = t
    coup = couplings(jnp.asarray(g_src), jnp.asarray(g_rec))
    t_sys = system_temperature(
        coup, t_src=t_src, t_unc=t_unc, t_cos=t_cos, t_sin=t_sin, t_rx=t_rx
    )
    return np.asarray(radiometer_power(t_sys, gain=jnp.asarray(gain)))


@pytest.mark.parametrize("temps", TEMPS)
@pytest.mark.parametrize("mag_src,phase_src", list(itertools.product(MAGS, PHASES)))
@pytest.mark.parametrize("mag_rec", MAGS)
def test_eq1_matches_the_numpy_reference(temps, mag_src, phase_src, mag_rec):
    g_src = np.array([mag_src * np.exp(1j * phase_src)])
    g_rec = np.array([mag_rec * np.exp(-0.7j)])
    gain = np.array([1000.0])

    reference = _numpy_power(temps, g_src, g_rec, gain)
    ours = _jax_power(temps, g_src, g_rec, gain)

    assert np.all(np.isfinite(reference)) == np.all(np.isfinite(ours))
    scale = max(abs(float(reference[0])), 1.0)
    assert abs(float(ours[0]) - float(reference[0])) / scale < 1e-13


def test_agreement_holds_across_a_frequency_band():
    """A per-channel Gamma, which is how real S11 measurements arrive."""
    rng = np.random.default_rng(0)
    n_freq = 64
    g_src = 0.3 * np.exp(1j * np.linspace(0, 6.0, n_freq)) * rng.uniform(0.5, 1.0, n_freq)
    g_rec = 0.12 * np.exp(-1j * np.linspace(0, 2.0, n_freq))
    gain = np.linspace(900.0, 1100.0, n_freq)
    temps = (300.0, 250.0, 30.0, -40.0, 290.0)

    reference = _numpy_power(temps, g_src, g_rec, gain)
    ours = _jax_power(temps, g_src, g_rec, gain)
    np.testing.assert_allclose(ours, reference, rtol=1e-13)


def test_agreement_with_per_channel_temperatures():
    """T_unc/T_cos/T_sin are smooth functions of frequency, not scalars."""
    n_freq = 32
    nu = np.linspace(-1.0, 1.0, n_freq)
    temps = (
        250.0 + 20.0 * nu,
        240.0 - 15.0 * nu**2,
        30.0 * nu,
        -40.0 + 5.0 * nu,
        290.0 + nu,
    )
    g_src = np.full(n_freq, 0.25 + 0.1j)
    g_rec = np.full(n_freq, 0.08 - 0.03j)
    gain = np.full(n_freq, 1000.0)

    np.testing.assert_allclose(
        _jax_power(temps, g_src, g_rec, gain),
        _numpy_power(temps, g_src, g_rec, gain),
        rtol=1e-13,
    )

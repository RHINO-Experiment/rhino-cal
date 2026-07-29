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
from rhino_cal_jax.reflection import Couplings, couplings
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


class TestTimeFrequencyShapes:
    """The (n_time, n_freq) support the docstring advertises.

    Checked against the numpy reference row by row: the reference is a
    per-spectrum function, so a time axis is just N independent calls, and that
    is exactly the equivalence worth pinning.
    """

    def _reference_rows(self, temps_per_time, g_src, g_rec, gain):
        return np.stack([
            _numpy_power(t, g_src, g_rec, gain) for t in temps_per_time
        ])

    def test_a_time_axis_matches_the_reference_row_by_row(self):
        n_time, n_freq = 5, 8
        g_src = 0.3 * np.exp(1j * np.linspace(0.0, 4.0, n_freq))
        g_rec = np.full(n_freq, 0.09 - 0.02j)
        gain = np.linspace(950.0, 1050.0, n_freq)

        # T_src drifts with time; the noise waves are fixed spectra.
        t_src_per_time = 300.0 + 10.0 * np.arange(n_time)
        t_unc = 250.0 - 5.0 * np.linspace(-1.0, 1.0, n_freq)
        temps_per_time = [
            (t_src_per_time[i], t_unc, 30.0, -40.0, 290.0) for i in range(n_time)
        ]
        reference = self._reference_rows(temps_per_time, g_src, g_rec, gain)

        coup = couplings(jnp.asarray(g_src), jnp.asarray(g_rec))
        broadcast_coup = Couplings.from_stacked(
            jnp.broadcast_to(coup.stacked, (n_time, n_freq, 4))
        )
        ours = radiometer_power(
            system_temperature(
                broadcast_coup,
                t_src=jnp.asarray(t_src_per_time)[:, None],   # (n_time, 1) column
                t_unc=jnp.asarray(t_unc), t_cos=30.0, t_sin=-40.0, t_rx=290.0,
            ),
            gain=jnp.asarray(gain),
        )
        assert np.asarray(ours).shape == (n_time, n_freq)
        np.testing.assert_allclose(np.asarray(ours), reference, rtol=1e-13)

    def test_a_bare_per_time_vector_is_read_as_per_frequency(self):
        """The documented trap, pinned so the convention cannot drift.

        n_time == n_freq here on purpose: that is the case no shape check could
        ever disambiguate, so the behaviour is defined by convention and this
        test is what holds the convention in place.
        """
        n = 4
        coup = Couplings.from_stacked(jnp.ones((n, n, 4)))
        bare = jnp.arange(1.0, n + 1.0)

        as_frequency = system_temperature(
            coup, t_src=bare, t_unc=0.0, t_cos=0.0, t_sin=0.0, t_rx=0.0
        )
        as_time = system_temperature(
            coup, t_src=bare[:, None], t_unc=0.0, t_cos=0.0, t_sin=0.0, t_rx=0.0
        )
        # Bare 1-D varies along the FREQUENCY axis: every row is identical.
        np.testing.assert_allclose(np.asarray(as_frequency), np.tile(bare, (n, 1)))
        # The explicit column varies along TIME: every row is constant.
        np.testing.assert_allclose(
            np.asarray(as_time), np.repeat(np.asarray(bare)[:, None], n, axis=1)
        )
        assert not np.allclose(np.asarray(as_frequency), np.asarray(as_time))

    def test_a_temperature_matching_neither_axis_raises(self):
        coup = Couplings.from_stacked(jnp.ones((5, 8, 4)))
        with pytest.raises(ValueError):
            system_temperature(
                coup, t_src=jnp.ones(3), t_unc=0.0, t_cos=0.0, t_sin=0.0, t_rx=0.0
            )

"""The synchrotron power law used as a stand-in antenna temperature."""

import astropy.units as un
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rhino_cal_jax.errors import ValidationError
from rhino_cal_jax.sky import synchrotron_temperature
from simulation.toy_sky import synchrotron_temperatures

FREQ = np.linspace(60e6, 85e6, 16)


def test_it_returns_the_amplitude_at_the_reference_frequency():
    got = synchrotron_temperature(jnp.array([210e6]), t_ref=180.0, beta=-2.6)
    assert float(got[0]) == pytest.approx(180.0, rel=1e-12)


def test_it_follows_the_declared_power_law():
    got = np.asarray(synchrotron_temperature(jnp.asarray(FREQ), t_ref=180.0, beta=-2.6))
    np.testing.assert_allclose(got, 180.0 * (FREQ / 210e6) ** -2.6, rtol=1e-13)


def test_a_steeper_index_gives_more_signal_below_the_reference():
    at_70 = jnp.array([70e6])
    assert float(synchrotron_temperature(at_70, beta=-2.8)[0]) > float(
        synchrotron_temperature(at_70, beta=-2.4)[0]
    )


def test_the_amplitude_scales_linearly():
    a = synchrotron_temperature(jnp.asarray(FREQ), t_ref=180.0)
    b = synchrotron_temperature(jnp.asarray(FREQ), t_ref=360.0)
    np.testing.assert_allclose(np.asarray(b), 2.0 * np.asarray(a), rtol=1e-13)


def test_it_is_differentiable_in_both_parameters():
    """The point of the port: these are fittable, not just evaluable."""
    grads = jax.grad(
        lambda t_ref, beta: jnp.sum(
            synchrotron_temperature(jnp.asarray(FREQ), t_ref=t_ref, beta=beta)
        ),
        argnums=(0, 1),
    )(180.0, -2.6)
    for g in grads:
        assert np.isfinite(float(g)) and abs(float(g)) > 0.0


def test_the_amplitude_gradient_is_the_shape_itself():
    """dT/dt_ref = (nu/nu_ref)^beta exactly, a closed form worth pinning."""
    grad = jax.grad(
        lambda t_ref: jnp.sum(synchrotron_temperature(jnp.asarray(FREQ), t_ref=t_ref))
    )(180.0)
    np.testing.assert_allclose(
        float(grad), float(np.sum((FREQ / 210e6) ** -2.6)), rtol=1e-12
    )


def test_it_jits_and_vmaps_over_the_spectral_index():
    betas = jnp.array([-2.4, -2.6, -2.8])
    out = jax.jit(
        jax.vmap(lambda b: synchrotron_temperature(jnp.asarray(FREQ), beta=b))
    )(betas)
    assert out.shape == (3, FREQ.size)
    assert np.all(np.isfinite(np.asarray(out)))


def test_it_matches_the_numpy_reference():
    """``synchrotron_temperatures`` returns a dimensionless-but-unsimplified Quantity.

    Internally it forms ``(freqs [Hz] / (210 MHz))**beta``: numerically
    dimensionless, but astropy leaves the unit as a composite
    ``MHz**(13/5) / Hz**(13/5)`` rather than simplifying it away. Plain
    ``np.asarray(...)`` (and even ``.value``) silently returns the number in
    that unsimplified unit, which is off from the true temperature by a factor
    of ``(1e6) ** 2.6`` -- it does not raise, it just gives the wrong number.
    ``.to_value(dimensionless_unscaled)`` forces the Hz/MHz factors to cancel
    and recovers the physical value.
    """
    reference = synchrotron_temperatures(FREQ.copy(), T_210=180.0, beta=-2.6)
    assert isinstance(reference, un.Quantity)

    # Demonstrate the trap before working around it.
    naive = np.asarray(reference)
    assert not np.allclose(naive, 180.0 * (FREQ / 210e6) ** -2.6, rtol=1e-6)

    reference_value = reference.to_value(un.dimensionless_unscaled)
    ours = synchrotron_temperature(jnp.asarray(FREQ), t_ref=180.0, beta=-2.6)
    np.testing.assert_allclose(np.asarray(ours), reference_value, rtol=1e-13)


class TestRejections:
    """A ``t_ref``/``beta``/``freq_ref`` that is neither scalar nor freq-shaped.

    Mirrors test_reflection.py::TestRejections: the docstring promises a
    result "shaped like freq", which NumPy's broadcasting would otherwise
    violate silently -- a length-1 array stretches across the whole band, and
    an array of any other shape forms an outer product with freq.
    """

    SMALL_FREQ = jnp.linspace(60e6, 85e6, 5)

    @pytest.mark.parametrize("kwarg", ["t_ref", "beta", "freq_ref"])
    def test_a_length_one_value_is_refused_rather_than_broadcast(self, kwarg):
        with pytest.raises(ValidationError, match="shape"):
            synchrotron_temperature(self.SMALL_FREQ, **{kwarg: jnp.array([300.0])})

    @pytest.mark.parametrize("kwarg", ["t_ref", "beta", "freq_ref"])
    def test_an_extra_rank_value_is_refused_rather_than_an_outer_product(self, kwarg):
        """A (n_freq, 1) column silently produced an (n_freq, n_freq) outer product."""
        with pytest.raises(ValidationError, match="shape"):
            synchrotron_temperature(
                self.SMALL_FREQ, **{kwarg: jnp.arange(1.0, 6.0)[:, None]}
            )

    def test_a_scalar_value_is_still_legal(self):
        got = synchrotron_temperature(self.SMALL_FREQ, t_ref=180.0, beta=-2.6, freq_ref=210e6)
        assert got.shape == self.SMALL_FREQ.shape

    def test_a_freq_shaped_value_is_still_legal(self):
        got = synchrotron_temperature(
            self.SMALL_FREQ,
            t_ref=jnp.linspace(170.0, 190.0, self.SMALL_FREQ.size),
            beta=jnp.full(self.SMALL_FREQ.size, -2.6),
            freq_ref=jnp.full(self.SMALL_FREQ.size, 210e6),
        )
        assert got.shape == self.SMALL_FREQ.shape

"""Reflection-coefficient construction for calibration loads and the receiver."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rhino_cal_jax.errors import ValidationError
from rhino_cal_jax.loads import Load, Receiver, cable_gamma, termination_gamma
from simulation.loads import Load as NumpyLoad

FREQ = np.linspace(60e6, 85e6, 16)


class TestTerminationGamma:
    def test_open_is_plus_one(self):
        np.testing.assert_allclose(np.asarray(termination_gamma("open", 4)), 1.0 + 0j)

    def test_short_is_minus_one(self):
        np.testing.assert_allclose(np.asarray(termination_gamma("short", 4)), -1.0 + 0j)

    def test_matched_is_zero(self):
        np.testing.assert_allclose(np.asarray(termination_gamma("matched", 4)), 0.0 + 0j)

    def test_a_resistive_termination_uses_the_mismatch_formula(self):
        got = termination_gamma("resistive", 3, impedance=75.0, z0=50.0)
        np.testing.assert_allclose(np.asarray(got), (75.0 - 50.0) / (75.0 + 50.0) + 0j)

    def test_the_result_is_always_complex(self):
        """A real dtype here would be refused downstream by couplings()."""
        assert jnp.issubdtype(termination_gamma("open", 4).dtype, jnp.complexfloating)

    def test_the_shape_is_one_value_per_channel(self):
        assert termination_gamma("open", 7).shape == (7,)

    def test_an_unknown_termination_is_refused(self):
        with pytest.raises(ValidationError, match="termination"):
            termination_gamma("banana", 4)

    def test_resistive_without_an_impedance_is_refused(self):
        with pytest.raises(ValidationError, match="impedance"):
            termination_gamma("resistive", 4)


class TestCableGamma:
    def test_a_zero_length_cable_is_transparent(self):
        term = termination_gamma("open", FREQ.size)
        np.testing.assert_allclose(
            np.asarray(cable_gamma(term, FREQ, length=0.0)), np.asarray(term), rtol=1e-14
        )

    def test_the_round_trip_phase_matches_the_numpy_load_convention(self):
        """Reproduces simulation/loads.py: one-way phase, s21 applied twice."""
        term = termination_gamma("open", FREQ.size)
        length, loss = 3.0, 0.9
        expected = (
            loss * np.exp(-1j * 2 * np.pi * length * FREQ / 299792458.0)
        ) ** 2 * np.asarray(term)
        np.testing.assert_allclose(
            np.asarray(cable_gamma(term, FREQ, length=length, loss=loss)),
            expected, rtol=1e-13,
        )

    def test_a_lossy_cable_shrinks_the_magnitude_as_the_square_of_the_loss(self):
        term = termination_gamma("open", FREQ.size)
        lossless = np.abs(np.asarray(cable_gamma(term, FREQ, length=3.0, loss=1.0)))
        lossy = np.abs(np.asarray(cable_gamma(term, FREQ, length=3.0, loss=0.5)))
        np.testing.assert_allclose(lossy, 0.25 * lossless, rtol=1e-13)

    def test_the_velocity_factor_stretches_the_phase(self):
        """A slower cable of length L matches a vacuum cable of length L/vf.

        A velocity factor vf < 1 means the wave propagates at vf*c, so the
        one-way transit time -- and hence the phase -- is the same as a
        vacuum cable (vf=1) of the longer length L/vf, not the shorter L*vf.
        """
        term = termination_gamma("open", FREQ.size)
        slow = cable_gamma(term, FREQ, length=3.0, velocity_factor=0.66)
        fast = cable_gamma(term, FREQ, length=3.0 / 0.66, velocity_factor=1.0)
        np.testing.assert_allclose(np.asarray(slow), np.asarray(fast), rtol=1e-13)

    def test_a_cabled_open_stays_on_the_unit_circle_when_lossless(self):
        """No loss means no magnitude change -- only phase rotation."""
        term = termination_gamma("open", FREQ.size)
        got = cable_gamma(term, FREQ, length=5.0, loss=1.0)
        np.testing.assert_allclose(np.abs(np.asarray(got)), 1.0, rtol=1e-13)


class TestContainers:
    def test_a_load_keeps_its_label_static(self):
        load = Load(gamma_src=jnp.zeros(4, dtype=complex), t_src=jnp.array(300.0),
                    label="ambient")
        leaves = jax.tree_util.tree_leaves(load)
        assert len(leaves) == 2  # gamma_src and t_src only; the label is static
        assert load.label == "ambient"

    def test_a_load_refuses_a_real_gamma(self):
        with pytest.raises(ValidationError, match="complex"):
            Load(gamma_src=jnp.zeros(4), t_src=jnp.array(300.0), label="ambient")

    def test_a_load_refuses_a_two_dimensional_gamma(self):
        with pytest.raises(ValidationError, match="1D"):
            Load(gamma_src=jnp.zeros((2, 4), dtype=complex), t_src=jnp.array(300.0),
                 label="x")

    def test_loads_are_differentiable_in_their_temperature(self):
        load = Load(gamma_src=jnp.zeros(4, dtype=complex), t_src=jnp.array(300.0),
                    label="x")
        grad = jax.grad(lambda ld: jnp.sum(ld.t_src))(load)
        assert float(grad.t_src) == pytest.approx(1.0)

    def test_a_receiver_rejects_a_gamma_that_does_not_match_its_gain(self):
        with pytest.raises(ValidationError, match="n_freq"):
            Receiver(gamma_rec=jnp.zeros(4, dtype=complex), gain=jnp.ones(5))

    def test_a_receiver_accepts_a_time_dependent_gain(self):
        rx = Receiver(gamma_rec=jnp.zeros(4, dtype=complex), gain=jnp.ones((10, 4)))
        assert rx.gain.shape == (10, 4)

    def test_a_receiver_accepts_a_scalar_gain(self):
        rx = Receiver(gamma_rec=jnp.zeros(4, dtype=complex), gain=jnp.array(1000.0))
        assert rx.gain.ndim == 0

    def test_a_receiver_refuses_a_real_gamma(self):
        with pytest.raises(ValidationError, match="complex"):
            Receiver(gamma_rec=jnp.zeros(4), gain=jnp.ones(4))


class TestConsistencyWithNumpyLoads:
    @pytest.mark.parametrize("kind", ["open", "short", "matched"])
    def test_termination_matches_the_numpy_load(self, kind):
        reference = NumpyLoad(
            physical_temperature=300.0, freqs=FREQ.copy(), termination_type=kind,
            label="ref",
        )
        ours = termination_gamma(kind, FREQ.size)
        np.testing.assert_allclose(
            np.asarray(ours), np.asarray(reference.gamma_src, dtype=complex),
            rtol=1e-14, atol=1e-16,
        )

    @pytest.mark.parametrize("length,loss", [(2.5, 0.95), (0.4, 1.0), (10.0, 0.8)])
    def test_cabled_open_matches_the_numpy_load(self, length, loss):
        reference = NumpyLoad(
            physical_temperature=300.0, freqs=FREQ.copy(), termination_type="open",
            effective_cable_length=length, cable_loss=loss, label="ref",
        )
        ours = cable_gamma(
            termination_gamma("open", FREQ.size), FREQ, length=length, loss=loss
        )
        np.testing.assert_allclose(
            np.asarray(ours), np.asarray(reference.gamma_src), rtol=1e-12, atol=1e-15
        )

    def test_a_cabled_short_matches_the_numpy_load(self):
        reference = NumpyLoad(
            physical_temperature=300.0, freqs=FREQ.copy(), termination_type="short",
            effective_cable_length=1.5, cable_loss=0.9, label="ref",
        )
        ours = cable_gamma(
            termination_gamma("short", FREQ.size), FREQ, length=1.5, loss=0.9
        )
        np.testing.assert_allclose(
            np.asarray(ours), np.asarray(reference.gamma_src), rtol=1e-12, atol=1e-15
        )


class TestTransforms:
    def test_cable_gamma_is_differentiable_in_length_and_loss(self):
        term = termination_gamma("open", FREQ.size)

        def loss_fn(length, loss):
            return jnp.sum(jnp.abs(cable_gamma(term, FREQ, length=length, loss=loss)))

        grads = jax.grad(loss_fn, argnums=(0, 1))(3.0, 0.9)
        for g in grads:
            assert np.isfinite(float(g))
        # |Gamma| = loss^2 for an open, independent of length -> d/dlength == 0
        assert float(grads[0]) == pytest.approx(0.0, abs=1e-9)
        assert abs(float(grads[1])) > 0.0

    def test_cable_gamma_jits(self):
        term = termination_gamma("open", FREQ.size)
        out = jax.jit(lambda t: cable_gamma(t, FREQ, length=2.0, loss=0.9))(term)
        assert out.shape == (FREQ.size,)
        assert np.all(np.isfinite(np.asarray(out)))

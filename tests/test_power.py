"""Draft Eqs. 1 and 8: system temperature, recorded power, radiometer noise."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rhino_cal_jax.errors import ValidationError
from rhino_cal_jax.power import (
    add_radiometer_noise,
    design_matrix,
    radiometer_power,
    system_temperature,
)
from rhino_cal_jax.reflection import couplings
from rhino_cal_jax.switching import SwitchCycle

G_SRC = jnp.array([0.25 + 0.1j, 0.2 - 0.05j])
G_REC = jnp.array([0.08 - 0.03j, 0.07 + 0.01j])
TEMPS = dict(t_src=300.0, t_unc=250.0, t_cos=30.0, t_sin=-40.0, t_rx=290.0)


class TestSystemTemperature:
    def test_a_matched_pair_reduces_to_t_src_plus_t_rx(self):
        coup = couplings(jnp.zeros(2, dtype=complex), jnp.zeros(2, dtype=complex))
        np.testing.assert_allclose(
            np.asarray(system_temperature(coup, **TEMPS)), 300.0 + 290.0, rtol=1e-14
        )

    def test_it_is_exactly_linear_in_the_temperature_vector(self):
        """Not approximately: the GCR is built on this being an identity."""
        coup = couplings(G_SRC, G_REC)
        a = dict(t_src=100.0, t_unc=50.0, t_cos=10.0, t_sin=-5.0, t_rx=200.0)
        b = dict(t_src=-30.0, t_unc=400.0, t_cos=-70.0, t_sin=60.0, t_rx=17.0)
        combined = {k: 2.5 * a[k] - 1.5 * b[k] for k in a}
        np.testing.assert_allclose(
            np.asarray(system_temperature(coup, **combined)),
            2.5 * np.asarray(system_temperature(coup, **a))
            - 1.5 * np.asarray(system_temperature(coup, **b)),
            rtol=1e-13,
        )

    def test_zero_temperatures_give_exactly_zero(self):
        coup = couplings(G_SRC, G_REC)
        np.testing.assert_array_equal(
            np.asarray(system_temperature(coup, **dict.fromkeys(TEMPS, 0.0))), 0.0
        )

    def test_per_channel_temperatures_broadcast(self):
        coup = couplings(G_SRC, G_REC)
        out = system_temperature(
            coup, t_src=jnp.array([300.0, 310.0]), t_unc=250.0,
            t_cos=30.0, t_sin=-40.0, t_rx=290.0,
        )
        assert out.shape == (2,)


class TestDesignMatrix:
    def test_it_flattens_time_and_frequency_into_rows(self):
        assert design_matrix(jnp.zeros((5, 8, 4))).shape == (40, 4)

    def test_its_product_reproduces_system_temperature(self):
        """The matrix form and the direct form are the same model."""
        coup = couplings(G_SRC, G_REC)
        vector = jnp.array([300.0, 250.0, 30.0, -40.0])
        direct = system_temperature(
            coup, t_src=300.0, t_unc=250.0, t_cos=30.0, t_sin=-40.0, t_rx=0.0
        )
        np.testing.assert_allclose(
            np.asarray(design_matrix(coup.stacked) @ vector),
            np.asarray(direct).ravel(), rtol=1e-14,
        )

    def test_a_wrong_trailing_axis_is_refused(self):
        with pytest.raises(ValidationError, match="4"):
            design_matrix(jnp.zeros((2, 3)))

    def test_a_one_dimensional_input_is_refused(self):
        with pytest.raises(ValidationError, match="4"):
            design_matrix(jnp.zeros(4))

    def test_two_switched_loads_span_all_four_coupling_directions(self):
        g_src = jnp.stack([
            jnp.array([0.30 + 0.10j, 0.28 - 0.02j]),
            jnp.array([0.02 + 0.00j, 0.01 + 0.03j]),
        ])
        cycle = SwitchCycle(source_index=jnp.array([0, 1, 0, 1]), labels=("ant", "load"))
        matrix = design_matrix(cycle.gather(couplings(g_src, G_REC).stacked))
        assert np.linalg.matrix_rank(np.asarray(matrix)) == 4

    def test_one_load_cannot_span_the_four_coupling_directions(self):
        """One load contributes ONE row per channel, so rank <= n_freq.

        Two channels therefore cap the rank at 2, short of the four coupling
        directions -- no matter how many time samples are taken, because every
        sample repeats the same row. That row-counting argument, not any
        proportionality between the columns, is what switching fixes.
        """
        one = couplings(G_SRC, G_REC).stacked[None, ...]  # (1, 2 freq, 4)
        cycle = SwitchCycle(source_index=jnp.zeros(4, dtype=int), labels=("ant",))
        matrix = design_matrix(cycle.gather(one))
        assert np.linalg.matrix_rank(np.asarray(matrix)) == 2  # == n_freq, < 4


class TestRadiometerNoise:
    def test_the_fractional_scatter_matches_one_over_root_bt(self):
        """Draft Eq. 8: sigma_w = 1 / sqrt(delta_nu t_int)."""
        noisy = add_radiometer_noise(
            jnp.full((20000,), 1000.0), jax.random.key(0), t_int=1.0, delta_nu=1e4
        )
        fractional = np.asarray(noisy) / 1000.0 - 1.0
        assert float(np.std(fractional)) == pytest.approx(1e-2, rel=0.05)
        assert abs(float(np.mean(fractional))) < 5e-4

    def test_the_noise_scales_with_the_power_itself(self):
        """Multiplicative: doubling the power doubles the absolute scatter."""
        key = jax.random.key(1)
        small = add_radiometer_noise(jnp.full((20000,), 100.0), key, t_int=1.0, delta_nu=1e4)
        large = add_radiometer_noise(jnp.full((20000,), 200.0), key, t_int=1.0, delta_nu=1e4)
        ratio = float(np.std(np.asarray(large))) / float(np.std(np.asarray(small)))
        assert ratio == pytest.approx(2.0, rel=1e-6)

    def test_a_longer_integration_narrows_the_scatter_as_the_square_root(self):
        key = jax.random.key(2)
        short = add_radiometer_noise(jnp.full((20000,), 1000.0), key, t_int=1.0, delta_nu=1e4)
        long = add_radiometer_noise(jnp.full((20000,), 1000.0), key, t_int=4.0, delta_nu=1e4)
        ratio = float(np.std(np.asarray(short))) / float(np.std(np.asarray(long)))
        assert ratio == pytest.approx(2.0, rel=1e-6)

    def test_a_wider_channel_narrows_the_scatter_as_the_square_root(self):
        key = jax.random.key(5)
        narrow = add_radiometer_noise(jnp.full((20000,), 1000.0), key, t_int=1.0, delta_nu=1e4)
        wide = add_radiometer_noise(jnp.full((20000,), 1000.0), key, t_int=1.0, delta_nu=4e4)
        ratio = float(np.std(np.asarray(narrow))) / float(np.std(np.asarray(wide)))
        assert ratio == pytest.approx(2.0, rel=1e-6)

    def test_folding_is_off_by_default_and_biases_the_mean_when_on(self):
        """The numpy reference takes abs(P + n), which is not a noise model.

        At a deliberately low B*tau the fold is visible: it reflects the negative
        tail and pushes the mean above the true power, towards the analytic
        E|N(1,1)| = 1.1666.
        """
        power = jnp.full((40000,), 1.0)
        kwargs = dict(t_int=1.0, delta_nu=1.0)  # sigma_w = 1, so the tail is large
        plain = add_radiometer_noise(power, jax.random.key(3), **kwargs)
        folded = add_radiometer_noise(power, jax.random.key(3), fold_negative=True, **kwargs)
        assert float(np.mean(np.asarray(plain))) == pytest.approx(1.0, abs=0.02)
        assert float(np.mean(np.asarray(folded))) == pytest.approx(1.1666, abs=0.02)
        np.testing.assert_allclose(np.asarray(folded), np.abs(np.asarray(plain)), rtol=1e-14)

    def test_the_same_key_reproduces_the_same_draw(self):
        power = jnp.full((100,), 1000.0)
        a = add_radiometer_noise(power, jax.random.key(7), t_int=1.0, delta_nu=1e4)
        b = add_radiometer_noise(power, jax.random.key(7), t_int=1.0, delta_nu=1e4)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_different_keys_give_different_draws(self):
        power = jnp.full((100,), 1000.0)
        a = add_radiometer_noise(power, jax.random.key(7), t_int=1.0, delta_nu=1e4)
        b = add_radiometer_noise(power, jax.random.key(8), t_int=1.0, delta_nu=1e4)
        assert not np.allclose(np.asarray(a), np.asarray(b))

    def test_it_preserves_a_two_dimensional_shape(self):
        noisy = add_radiometer_noise(
            jnp.full((7, 5), 1000.0), jax.random.key(4), t_int=1.0, delta_nu=1e4
        )
        assert noisy.shape == (7, 5)

    def test_a_scalar_bandwidth_is_the_normal_case_and_a_column_varies_by_time(self):
        """The documented convention for non-scalar t_int / delta_nu.

        n_time == n_freq here on purpose: that is the case no shape check could
        disambiguate, so the behaviour is fixed by convention and this holds it.
        """
        n = 4
        power = jnp.full((n, n), 1000.0)
        key = jax.random.key(11)
        t_int = jnp.array([1.0, 4.0, 9.0, 16.0])
        per_time = t_int[:, None]  # (n_time, 1) column

        column = add_radiometer_noise(power, key, t_int=per_time, delta_nu=1.0)
        bare = add_radiometer_noise(power, key, t_int=t_int, delta_nu=1.0)

        # Same key and shape drive both calls, so the underlying normal draws are
        # identical; only the broadcast axis of sigma_w differs between them.
        draws = np.asarray(jax.random.normal(key, power.shape))
        frac_column = np.asarray(column) / 1000.0 - 1.0
        frac_bare = np.asarray(bare) / 1000.0 - 1.0
        scale = 1.0 / np.sqrt(np.asarray(t_int))

        # The column scales sigma_w row-wise: row i has sigma_w = 1/sqrt(t_int[i]).
        np.testing.assert_allclose(frac_column, draws * scale[:, None], rtol=1e-12)
        # The bare vector instead scales column-wise, along frequency.
        np.testing.assert_allclose(frac_bare, draws * scale[None, :], rtol=1e-12)
        # The two broadcast axes disagree, so the results disagree too.
        assert not np.allclose(frac_bare[1], frac_column[1])


class TestTransforms:
    def test_gradients_reach_every_temperature(self):
        coup = couplings(G_SRC, G_REC)

        def loss(t_src, t_unc, t_cos, t_sin, t_rx):
            t_sys = system_temperature(
                coup, t_src=t_src, t_unc=t_unc, t_cos=t_cos, t_sin=t_sin, t_rx=t_rx
            )
            return jnp.sum(radiometer_power(t_sys, gain=1000.0))

        grads = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(300.0, 250.0, 30.0, -40.0, 290.0)
        for g in grads:
            assert np.isfinite(float(g)) and abs(float(g)) > 0.0

    def test_the_whole_forward_model_jits(self):
        @jax.jit
        def forward(g_src, g_rec, t_src, key):
            coup = couplings(g_src, g_rec)
            t_sys = system_temperature(
                coup, t_src=t_src, t_unc=250.0, t_cos=30.0, t_sin=-40.0, t_rx=290.0
            )
            return add_radiometer_noise(
                radiometer_power(t_sys, gain=1000.0), key, t_int=1.0, delta_nu=1e4
            )

        out = forward(G_SRC, G_REC, 300.0, jax.random.key(0))
        assert out.shape == (2,)
        assert np.all(np.isfinite(np.asarray(out)))

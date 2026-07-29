"""Draft Eqs. 2-6: the four source-dependent coupling spectra."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rhino_cal_jax.errors import ValidationError
from rhino_cal_jax.reflection import Couplings, couplings, reflection_factor


class TestReflectionFactor:
    def test_matched_receiver_is_unity(self):
        """Gamma_rec = 0 makes F = 1 exactly (draft Eq. 3)."""
        f = reflection_factor(jnp.array([0.3 + 0.1j]), jnp.array([0.0 + 0.0j]))
        assert f == pytest.approx(1.0 + 0.0j)

    def test_matches_the_closed_form(self):
        g_src = jnp.array([0.2 - 0.1j, 0.0 + 0.4j])
        g_rec = jnp.array([0.05 + 0.02j, -0.1 + 0.0j])
        expected = np.sqrt(1 - np.abs(np.asarray(g_rec)) ** 2) / (
            1 - np.asarray(g_rec) * np.asarray(g_src)
        )
        np.testing.assert_allclose(
            np.asarray(reflection_factor(g_src, g_rec)), expected, rtol=1e-14
        )


class TestCouplings:
    def test_a_matched_source_kills_every_noise_wave_coupling(self):
        """Gamma_src = 0: all three noise-wave couplings vanish.

        c_src does NOT become 1 here. It keeps the double-mismatch factor
        (1 - |Gamma_rec|^2) -- the textbook power-transfer efficiency -- because
        a mismatched receiver still reflects |Gamma_rec|^2 straight back out.
        Asserting 1.0 with a mismatched receiver would be wrong physics, and
        the repo's own numpy reference returns 0.99 for this input.
        """
        gamma_rec = jnp.full(3, 0.1 + 0.0j)
        c = couplings(jnp.zeros(3, dtype=complex), gamma_rec)
        np.testing.assert_allclose(
            np.asarray(c.c_src), 1.0 - np.abs(np.asarray(gamma_rec)) ** 2, rtol=1e-14
        )
        np.testing.assert_allclose(np.asarray(c.k_unc), 0.0, atol=1e-15)
        np.testing.assert_allclose(np.asarray(c.k_cos), 0.0, atol=1e-15)
        np.testing.assert_allclose(np.asarray(c.k_sin), 0.0, atol=1e-15)

    def test_a_fully_matched_pair_transfers_everything(self):
        """Both ports matched: F = 1, c_src = 1, nothing reflected anywhere."""
        c = couplings(jnp.zeros(3, dtype=complex), jnp.zeros(3, dtype=complex))
        np.testing.assert_allclose(np.asarray(c.c_src), 1.0, rtol=1e-14)
        np.testing.assert_allclose(np.asarray(c.stacked[:, 1:]), 0.0, atol=1e-15)

    def test_k_unc_carries_the_square_of_F(self):
        """D1: draft Eq. 4 prints |Gamma|^2 |F|, the model is |Gamma|^2 |F|^2.

        Pinned because the two differ by |F|, which is 1 only for a matched
        receiver -- exactly the case a careless test would pick.
        """
        g_src, g_rec = jnp.array([0.4 + 0.2j]), jnp.array([0.3 - 0.15j])
        f = reflection_factor(g_src, g_rec)
        c = couplings(g_src, g_rec)
        np.testing.assert_allclose(
            np.asarray(c.k_unc),
            np.abs(np.asarray(g_src)) ** 2 * np.abs(np.asarray(f)) ** 2,
            rtol=1e-14,
        )
        assert not np.allclose(
            np.asarray(c.k_unc), np.abs(np.asarray(g_src)) ** 2 * np.abs(np.asarray(f))
        )

    def test_cos_and_sin_are_the_real_and_imaginary_parts_of_gamma_F(self):
        g_src, g_rec = jnp.array([0.4 + 0.2j]), jnp.array([0.3 - 0.15j])
        prod = np.asarray(g_src) * np.asarray(reflection_factor(g_src, g_rec))
        c = couplings(g_src, g_rec)
        np.testing.assert_allclose(np.asarray(c.k_cos), prod.real, rtol=1e-14)
        np.testing.assert_allclose(np.asarray(c.k_sin), prod.imag, rtol=1e-14)

    def test_stacked_orders_the_columns_as_documented(self):
        c = couplings(jnp.array([0.2 + 0.1j]), jnp.array([0.05 + 0.0j]))
        stacked = c.stacked
        assert stacked.shape == (1, 4)
        np.testing.assert_allclose(np.asarray(stacked[:, 0]), np.asarray(c.c_src))
        np.testing.assert_allclose(np.asarray(stacked[:, 1]), np.asarray(c.k_unc))
        np.testing.assert_allclose(np.asarray(stacked[:, 2]), np.asarray(c.k_cos))
        np.testing.assert_allclose(np.asarray(stacked[:, 3]), np.asarray(c.k_sin))

    def test_broadcasts_a_source_axis_against_a_shared_receiver(self):
        g_src = jnp.array([[0.2 + 0.1j, 0.3 + 0.0j], [0.0 + 0.0j, -0.1 + 0.2j]])  # (2 src, 2 freq)
        g_rec = jnp.array([0.05 + 0.0j, 0.06 - 0.01j])  # (2 freq)
        c = couplings(g_src, g_rec)
        assert c.c_src.shape == (2, 2)
        assert c.stacked.shape == (2, 2, 4)


class TestRejections:
    def test_a_real_gamma_is_refused(self):
        """A real Gamma silently zeroes k_sin -- finite, right-shaped, wrong."""
        with pytest.raises(ValidationError, match="complex"):
            couplings(jnp.array([0.2, 0.3]), jnp.array([0.05 + 0.0j, 0.0 + 0.0j]))

    def test_a_real_gamma_rec_is_refused(self):
        with pytest.raises(ValidationError, match="complex"):
            couplings(jnp.array([0.2 + 0.0j]), jnp.array([0.05]))

    def test_a_length_one_gamma_rec_is_refused_rather_than_broadcast(self):
        """The silent-broadcast trap: one receiver value applied to every channel.

        NumPy semantics would happily stretch a (1,) gamma_rec across three
        channels and return a finite, correctly-shaped, wrong Couplings.
        """
        with pytest.raises(ValidationError, match="channels"):
            couplings(jnp.array([0.2 + 0.1j, 0.3 + 0.0j, -0.1 + 0.2j]),
                      jnp.array([0.05 + 0.0j]))

    def test_a_scalar_gamma_rec_is_refused(self):
        with pytest.raises(ValidationError, match="1D"):
            couplings(jnp.array([0.2 + 0.1j, 0.3 + 0.0j]), jnp.array(0.05 + 0.0j))

    def test_a_gamma_rec_longer_than_the_band_is_refused(self):
        with pytest.raises(ValidationError, match="channels"):
            couplings(jnp.array([0.2 + 0.1j, 0.3 + 0.0j]),
                      jnp.full(5, 0.05 + 0.0j))


class TestBoundaries:
    """Extreme reflection coefficients: failures must be loud, never finite-wrong."""

    # The source phase is OFFSET from the receiver phase, not conjugate to it.
    # With g_src = m e^{-i phi} against g_rec = m' e^{+i phi} the product
    # Gamma_src Gamma_rec = m m' e^{i(phi - phi)} is always REAL, so the whole
    # sweep would run with a real denominator and a real F -- 64 cells that
    # never touch the complex half of the function, and blind to any swapped
    # real/imaginary term inside reflection_factor. The offset makes
    # max |Im F| ~ 0.56 across the grid instead of ~1e-13.
    PHASE_OFFSET = np.pi / 4

    @pytest.mark.parametrize("mag_rec", [0.0, 0.5, 0.9, 0.999])
    @pytest.mark.parametrize("mag_src", [0.0, 0.5, 0.9, 0.999])
    @pytest.mark.parametrize("phase", [0.0, np.pi / 3, np.pi, -np.pi / 2])
    def test_physical_reflections_stay_finite(self, mag_rec, mag_src, phase):
        g_rec = jnp.array([mag_rec * np.exp(1j * phase)])
        g_src = jnp.array([mag_src * np.exp(1j * (phase + self.PHASE_OFFSET))])
        c = couplings(g_src, g_rec)
        assert np.all(np.isfinite(np.asarray(c.stacked)))

    @pytest.mark.parametrize("mag_rec", [0.0, 0.5, 0.9, 0.999])
    @pytest.mark.parametrize("mag_src", [0.0, 0.5, 0.9, 0.999])
    @pytest.mark.parametrize("phase", [0.0, np.pi / 3, np.pi, -np.pi / 2])
    def test_the_exact_identities_hold_across_the_grid(self, mag_rec, mag_src, phase):
        """Two algebraic identities the four couplings must satisfy exactly.

        ``c_src + k_unc == |F|^2`` (the |Gamma|^2 terms cancel) and
        ``k_cos^2 + k_sin^2 == k_unc`` (both are |Gamma F|^2). They are cheap,
        they hold to round-off, and either would have caught a dropped power
        like the one the draft's Eq. 4 contains -- which is exactly the class
        of mistake this module had to be checked for.
        """
        g_rec = jnp.array([mag_rec * np.exp(1j * phase)])
        g_src = jnp.array([mag_src * np.exp(1j * (phase + self.PHASE_OFFSET))])
        c = couplings(g_src, g_rec)
        abs2_f = np.abs(np.asarray(reflection_factor(g_src, g_rec))) ** 2
        np.testing.assert_allclose(
            np.asarray(c.c_src) + np.asarray(c.k_unc), abs2_f, rtol=1e-13, atol=1e-15
        )
        np.testing.assert_allclose(
            np.asarray(c.k_cos) ** 2 + np.asarray(c.k_sin) ** 2,
            np.asarray(c.k_unc), rtol=1e-13, atol=1e-15,
        )

    def test_an_overunity_receiver_gives_nan_not_a_plausible_number(self):
        """|Gamma_rec| > 1 is unphysical; sqrt of a negative must not be silently real."""
        c = couplings(jnp.array([0.2 + 0.0j]), jnp.array([1.5 + 0.0j]))
        assert np.all(np.isnan(np.asarray(c.stacked)))

    def test_the_resonance_pole_is_not_finite(self):
        """Gamma_src * Gamma_rec -> 1 is a genuine pole of Eq. 3.

        Asserted as "not finite" rather than "Inf" on purpose: complex IEEE
        arithmetic turns the pole into NaN, not Inf. At this input the numerator
        vanishes too (|Gamma_rec| = 1), giving 0/0; and even for a clean pole
        with |Gamma_rec| < 1, taking the modulus of an infinite complex value
        evaluates 0*inf and yields NaN. Plain numpy agrees. Either way it is
        loud, which is all this test needs to establish.
        """
        c = couplings(jnp.array([1.0 + 0.0j]), jnp.array([1.0 + 0.0j]))
        assert not np.all(np.isfinite(np.asarray(c.stacked)))


class TestStackedRoundTrip:
    """`stacked` and `from_stacked` must be exact inverses.

    A transposed column index at a distant call site would give a Couplings
    that is finite, correctly shaped and wrong, so the order is defined once
    and this pins the round trip.
    """

    def test_from_stacked_inverts_stacked(self):
        c = couplings(jnp.array([0.25 + 0.1j, 0.2 - 0.05j]),
                      jnp.array([0.08 - 0.03j, 0.07 + 0.01j]))
        back = Couplings.from_stacked(c.stacked)
        for field in ("c_src", "k_unc", "k_cos", "k_sin"):
            np.testing.assert_array_equal(
                np.asarray(getattr(back, field)), np.asarray(getattr(c, field))
            )

    def test_from_stacked_reads_the_columns_in_the_documented_order(self):
        """Column identity, not just shape -- a transposition must be visible."""
        stacked = jnp.asarray(np.arange(8.0).reshape(2, 4))
        c = Couplings.from_stacked(stacked)
        np.testing.assert_array_equal(np.asarray(c.c_src), [0.0, 4.0])
        np.testing.assert_array_equal(np.asarray(c.k_unc), [1.0, 5.0])
        np.testing.assert_array_equal(np.asarray(c.k_cos), [2.0, 6.0])
        np.testing.assert_array_equal(np.asarray(c.k_sin), [3.0, 7.0])

    def test_from_stacked_refuses_a_wrong_trailing_axis(self):
        with pytest.raises(ValidationError, match="4"):
            Couplings.from_stacked(jnp.zeros((2, 3)))

    def test_mismatched_field_shapes_are_refused_at_construction(self):
        """Fail at construction naming the field, not later inside `stacked`."""
        with pytest.raises(ValidationError, match="shape"):
            Couplings(c_src=jnp.zeros(3), k_unc=jnp.zeros(5),
                      k_cos=jnp.zeros(3), k_sin=jnp.zeros(3))


class TestTransforms:
    def test_gradients_flow_to_both_reflection_coefficients(self):
        def loss(re_src, im_src, re_rec, im_rec):
            c = couplings(re_src + 1j * im_src, re_rec + 1j * im_rec)
            return jnp.sum(c.stacked)

        grads = jax.grad(loss, argnums=(0, 1, 2, 3))(
            jnp.array([0.2]), jnp.array([0.1]), jnp.array([0.05]), jnp.array([0.02])
        )
        for g in grads:
            assert np.all(np.isfinite(np.asarray(g)))
            assert not np.allclose(np.asarray(g), 0.0)

    def test_jit_and_vmap_round_trip(self):
        g_src = jnp.array([[0.2 + 0.1j], [0.3 - 0.2j]])
        g_rec = jnp.array([0.05 + 0.0j])
        direct = couplings(g_src, g_rec).stacked
        mapped = jax.jit(jax.vmap(couplings, in_axes=(0, None)))(g_src, g_rec).stacked
        np.testing.assert_allclose(np.asarray(direct), np.asarray(mapped), rtol=1e-14)

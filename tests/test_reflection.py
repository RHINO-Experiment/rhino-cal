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
        """Gamma_src = Gamma_rec = 1 is the exact coincidence of Eq. 3's pole.

        Asserted as "not finite" rather than "Inf" on purpose: this input gives
        NaN, not Inf, because numerator and denominator vanish together --
        ``sqrt(1 - |Gamma_rec|^2) = 0`` and ``1 - Gamma_src Gamma_rec = 0`` --
        a genuine 0/0. Plain numpy agrees. This is NOT the general behaviour of
        the pole: see :func:`test_a_reachable_near_resonance_is_large_but_finite`
        for the physically reachable case, which stays finite.
        """
        c = couplings(jnp.array([1.0 + 0.0j]), jnp.array([1.0 + 0.0j]))
        assert not np.all(np.isfinite(np.asarray(c.stacked)))

    def test_a_reachable_near_resonance_is_large_but_finite(self):
        """The pole approached with both |Gamma| < 1 is large, not NaN.

        Only the exact coincidence Gamma_src = Gamma_rec = 1 (previous test)
        gives 0/0. Here both reflection coefficients stay inside the unit
        circle -- the physically reachable regime -- and the resonance instead
        gives a large but perfectly finite |F| (~7071 at this input).
        """
        g = jnp.array([1.0 - 1e-8 + 0.0j])
        f = reflection_factor(g, g)
        assert np.all(np.isfinite(np.asarray(f)))
        assert np.abs(np.asarray(f))[0] == pytest.approx(7071.07, rel=1e-3)
        c = couplings(g, g)
        assert np.all(np.isfinite(np.asarray(c.stacked)))

    def test_float64_stays_accurate_at_a_demanding_near_unity_gamma_rec(self):
        """Pin float64 accuracy right where ``1 - |Gamma_rec|^2`` cancels hardest.

        ``1 - |Gamma_rec|^2`` is a difference of nearly-equal numbers, so it
        loses precision as ``|Gamma_rec| -> 1`` -- in float32 this is the
        silent accuracy loss documented on
        :func:`~rhino_cal_jax.reflection.reflection_factor`. This suite runs
        float64 throughout (``tests/conftest.py``), so this pins that float64
        stays accurate at a demanding ``1 - |Gamma_rec|^2 = 1e-6`` against an
        independently computed (mpmath, 50-digit) reference -- a future
        refactor to a worse-conditioned formulation would show up here as
        numeric drift rather than as a silent, untested regression.
        """
        mpmath = pytest.importorskip("mpmath")
        mpmath.mp.dps = 50
        g_src_mp = mpmath.mpc("0.3", "0.1")
        one_minus_mag2 = mpmath.mpf("1e-6")
        mag_rec = mpmath.sqrt(1 - one_minus_mag2)
        g_rec_mp = mpmath.mpc(mag_rec, 0)
        f_mp = mpmath.sqrt(1 - abs(g_rec_mp) ** 2) / (1 - g_src_mp * g_rec_mp)
        reference = (1 - abs(g_src_mp) ** 2) * abs(f_mp) ** 2

        c = couplings(jnp.array([0.3 + 0.1j]), jnp.array([complex(mag_rec, 0.0)]))
        np.testing.assert_allclose(float(c.c_src[0]), float(reference), rtol=1e-9)


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

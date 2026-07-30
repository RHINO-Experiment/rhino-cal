"""Draft Eqs. 11-12: which source is connected at each time sample."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rhino_cal_jax.errors import ValidationError
from rhino_cal_jax.loads import Load, termination_gamma
from rhino_cal_jax.reflection import couplings
from rhino_cal_jax.switching import SwitchCycle, stack_load_gammas
from utils.utils import assign_states


class TestConstruction:
    def test_from_labels_maps_names_to_indices(self):
        cycle = SwitchCycle.from_labels(
            ["antenna", "load", "antenna", "noise_diode"],
            labels=("antenna", "load", "noise_diode"),
        )
        np.testing.assert_array_equal(np.asarray(cycle.source_index), [0, 1, 0, 2])

    def test_an_unknown_label_is_refused(self):
        with pytest.raises(ValidationError, match="banana"):
            SwitchCycle.from_labels(["antenna", "banana"], labels=("antenna", "load"))

    def test_an_out_of_range_index_is_refused(self):
        with pytest.raises(ValidationError, match="out of range"):
            SwitchCycle(source_index=jnp.array([0, 3]), labels=("a", "b"))

    def test_a_negative_index_is_refused(self):
        with pytest.raises(ValidationError, match="out of range"):
            SwitchCycle(source_index=jnp.array([-1, 0]), labels=("a", "b"))

    def test_a_float_index_is_refused(self):
        """Float indices would round silently and mis-assign samples."""
        with pytest.raises(ValidationError, match="integer"):
            SwitchCycle(source_index=jnp.array([0.0, 1.0]), labels=("a", "b"))

    def test_a_two_dimensional_index_is_refused(self):
        with pytest.raises(ValidationError, match="1D"):
            SwitchCycle(source_index=jnp.zeros((2, 2), dtype=int), labels=("a", "b"))

    def test_duplicate_labels_are_refused(self):
        """Two sources with the same name make gather results unattributable."""
        with pytest.raises(ValidationError, match="duplicate"):
            SwitchCycle(source_index=jnp.array([0, 1]), labels=("a", "a"))

    def test_n_source_and_n_time_report_the_shape(self):
        cycle = SwitchCycle(source_index=jnp.array([0, 1, 1]), labels=("a", "b"))
        assert cycle.n_source == 2
        assert cycle.n_time == 3


class TestSchedule:
    def test_a_schedule_assigns_each_sample_to_the_last_change_before_it(self):
        cycle = SwitchCycle.from_schedule(
            times=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            switch_times=np.array([0.0, 2.0, 3.5]),
            switch_labels=["antenna", "load", "antenna"],
            labels=("antenna", "load"),
        )
        np.testing.assert_array_equal(np.asarray(cycle.source_index), [0, 0, 1, 1, 0])

    def test_samples_before_the_first_change_take_the_first_state(self):
        cycle = SwitchCycle.from_schedule(
            times=np.array([-1.0, 0.5]),
            switch_times=np.array([0.0, 1.0]),
            switch_labels=["antenna", "load"],
            labels=("antenna", "load"),
        )
        np.testing.assert_array_equal(np.asarray(cycle.source_index), [0, 0])

    def test_it_matches_the_numpy_assign_states(self):
        """The reference the numpy pipeline uses to label its spectra."""
        rng = np.random.default_rng(1)
        times = np.sort(rng.uniform(0.0, 100.0, 200))
        switch_times = np.arange(0.0, 100.0, 7.0)
        names = np.array(["antenna", "load", "noise_diode"])
        switch_labels = list(names[np.arange(switch_times.size) % 3])

        reference = assign_states(times, switch_times, np.array(switch_labels))
        cycle = SwitchCycle.from_schedule(
            times, switch_times, switch_labels, labels=tuple(names)
        )
        np.testing.assert_array_equal(
            names[np.asarray(cycle.source_index)], np.asarray(reference)
        )

    def test_a_schedule_whose_labels_and_times_disagree_is_refused(self):
        with pytest.raises(ValidationError, match="switch_labels"):
            SwitchCycle.from_schedule(
                times=np.array([0.0, 1.0]),
                switch_times=np.array([0.0, 1.0, 2.0]),
                switch_labels=["antenna"],
                labels=("antenna", "load"),
            )


class TestGather:
    def test_one_hot_is_a_permutation_matrix_of_the_index(self):
        cycle = SwitchCycle(source_index=jnp.array([2, 0, 1]), labels=("a", "b", "c"))
        np.testing.assert_array_equal(
            np.asarray(cycle.one_hot()),
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )

    def test_gather_selects_the_connected_source_per_sample(self):
        per_source = jnp.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
        cycle = SwitchCycle(source_index=jnp.array([2, 0, 0, 1]), labels=("a", "b", "c"))
        np.testing.assert_array_equal(
            np.asarray(cycle.gather(per_source)),
            [[30.0, 31.0], [10.0, 11.0], [10.0, 11.0], [20.0, 21.0]],
        )

    def test_gather_rejects_a_mismatched_source_axis(self):
        cycle = SwitchCycle(source_index=jnp.array([0, 1]), labels=("a", "b"))
        with pytest.raises(ValidationError, match="n_source"):
            cycle.gather(jnp.zeros((3, 4)))

    def test_gather_rejects_a_scalar(self):
        cycle = SwitchCycle(source_index=jnp.array([0, 1]), labels=("a", "b"))
        with pytest.raises(ValidationError, match="n_source"):
            cycle.gather(jnp.asarray(1.0))

    def test_gather_carries_the_trailing_column_axis_of_stacked_couplings(self):
        g_src = jnp.array([[0.2 + 0.1j, 0.1 + 0.0j], [0.0 + 0.0j, 0.3 - 0.1j]])
        g_rec = jnp.array([0.05 + 0.0j, 0.05 + 0.0j])
        stacked = couplings(g_src, g_rec).stacked  # (2 src, 2 freq, 4)
        cycle = SwitchCycle(source_index=jnp.array([0, 1, 1]), labels=("a", "b"))
        assert cycle.gather(stacked).shape == (3, 2, 4)

    def test_gather_agrees_with_the_one_hot_contraction(self):
        """The index form and Eq. 12's theta must be the same operator -- for 2-D."""
        per_source = jnp.arange(12.0).reshape(3, 4)
        cycle = SwitchCycle(source_index=jnp.array([1, 0, 2, 2]), labels=("a", "b", "c"))
        np.testing.assert_allclose(
            np.asarray(cycle.gather(per_source)),
            np.asarray(cycle.one_hot() @ per_source),
            rtol=1e-14,
        )

    def test_one_hot_matmul_diverges_from_gather_for_3d_input_at_n_source_eq_n_freq(self):
        """The 2-D equivalence above does NOT extend to rank-3 input.

        At n_source == n_freq, `one_hot() @ per_source` does not raise -- `@`
        is a plain matrix product over the leading two axes, so it silently
        returns a same-shaped but numerically different array from
        `gather(per_source)`. This is exactly the case a generic-shape mismatch
        would have caught loudly, and exactly the case stacked couplings
        (n_source, n_freq, 4) hit whenever n_source == n_freq.
        """
        n = 4
        per_source = jnp.arange(float(n**3)).reshape(n, n, n)
        cycle = SwitchCycle(source_index=jnp.array([2, 0, 1, 3]), labels=("a", "b", "c", "d"))

        gathered = cycle.gather(per_source)
        via_matmul = cycle.one_hot() @ per_source

        assert gathered.shape == via_matmul.shape == (n, n, n)
        assert not np.allclose(np.asarray(gathered), np.asarray(via_matmul))


class TestStackLoadGammas:
    def _loads(self, n_freq=4):
        return [
            Load(gamma_src=termination_gamma("open", n_freq), t_src=jnp.array(0.0),
                 label="antenna"),
            Load(gamma_src=termination_gamma("matched", n_freq), t_src=jnp.array(300.0),
                 label="ambient"),
            Load(gamma_src=termination_gamma("short", n_freq), t_src=jnp.array(400.0),
                 label="hot"),
        ]

    def test_it_stacks_in_order_and_returns_the_labels(self):
        stacked, labels = stack_load_gammas(self._loads())
        assert stacked.shape == (3, 4)
        assert labels == ("antenna", "ambient", "hot")
        np.testing.assert_allclose(np.asarray(stacked[0]), 1.0 + 0j)
        np.testing.assert_allclose(np.asarray(stacked[1]), 0.0 + 0j)
        np.testing.assert_allclose(np.asarray(stacked[2]), -1.0 + 0j)

    def test_the_result_stays_complex(self):
        stacked, _ = stack_load_gammas(self._loads())
        assert jnp.issubdtype(stacked.dtype, jnp.complexfloating)

    def test_disagreeing_channel_counts_are_refused(self):
        loads = self._loads()
        loads[1] = Load(gamma_src=termination_gamma("matched", 7),
                        t_src=jnp.array(300.0), label="ambient")
        with pytest.raises(ValidationError, match="n_freq"):
            stack_load_gammas(loads)

    def test_duplicate_labels_are_refused(self):
        loads = self._loads()
        loads[2] = Load(gamma_src=termination_gamma("short", 4),
                        t_src=jnp.array(400.0), label="antenna")
        with pytest.raises(ValidationError, match="duplicate"):
            stack_load_gammas(loads)

    def test_an_empty_sequence_is_refused(self):
        with pytest.raises(ValidationError, match="at least one"):
            stack_load_gammas([])

    def test_it_composes_with_a_switch_cycle(self):
        """The intended end-to-end use: loads -> stacked Gamma -> per-time couplings."""
        loads = self._loads()
        stacked, labels = stack_load_gammas(loads)
        cycle = SwitchCycle.from_labels(
            [ld.label for ld in loads] * 2, labels=labels
        )
        coup = couplings(stacked, termination_gamma("resistive", 4, impedance=45.0))
        gathered = cycle.gather(coup.stacked)
        assert gathered.shape == (6, 4, 4)


class TestIdentifiability:
    """What switching buys, counted per frequency channel.

    Each switch position contributes ONE equation per channel against the three
    noise-wave unknowns there, so the rank is min(n_src, 3) * n_freq. Verified
    numerically: 1 load -> 4/12, 2 -> 8/12, 3 -> 12/12 at n_freq = 4.

    This counts only (T_unc, T_cos, T_sin) with T_rx taken as known, which is
    exactly what `_design` below builds (it drops the source column and never
    gives T_rx a column at all). With T_rx also free per channel the count
    becomes min(n_src, 4) * n_freq, and four loads would be needed.
    """

    N_FREQ = 4
    G_SRC = np.array([
        [0.30 + 0.10j, 0.28 + 0.05j, 0.26 + 0.00j, 0.24 - 0.05j],
        [0.02 + 0.00j, 0.01 + 0.03j, 0.00 + 0.02j, -0.01 + 0.01j],
        [-0.60 + 0.15j, -0.62 - 0.10j, -0.64 + 0.05j, -0.66 + 0.20j],
    ])
    G_REC = np.full(4, 0.08 - 0.03j)

    def _design(self, n_source):
        """Per-channel design matrix for (T_unc, T_cos, T_sin) at each channel."""
        coup = couplings(jnp.asarray(self.G_SRC[:n_source]), jnp.asarray(self.G_REC))
        cycle = SwitchCycle(
            source_index=jnp.arange(12) % n_source,
            labels=tuple(str(i) for i in range(n_source)),
        )
        # columns 1..3 are (k_unc, k_cos, k_sin); column 0 is the source term.
        k = np.asarray(cycle.gather(coup.stacked))[..., 1:]
        rows = []
        for t in range(k.shape[0]):
            for f in range(self.N_FREQ):
                row = np.zeros(3 * self.N_FREQ)
                row[3 * f:3 * (f + 1)] = k[t, f]
                rows.append(row)
        return np.array(rows)

    @pytest.mark.parametrize("n_source,expected", [(1, 4), (2, 8), (3, 12)])
    def test_rank_is_one_equation_per_channel_per_load(self, n_source, expected):
        assert np.linalg.matrix_rank(self._design(n_source)) == expected

    def test_three_loads_make_the_per_channel_system_square(self):
        assert np.linalg.matrix_rank(self._design(3)) == 3 * self.N_FREQ

    def test_fewer_than_three_loads_leave_it_deficient(self):
        for n_source in (1, 2):
            assert np.linalg.matrix_rank(self._design(n_source)) < 3 * self.N_FREQ


class TestTransforms:
    def test_gather_is_jittable(self):
        cycle = SwitchCycle(source_index=jnp.array([0, 1]), labels=("a", "b"))
        per_source = jnp.array([[1.0], [2.0]])
        out = jax.jit(lambda c, p: c.gather(p))(cycle, per_source)
        np.testing.assert_allclose(np.asarray(out), [[1.0], [2.0]])

    def test_gather_is_differentiable_in_the_gathered_values(self):
        cycle = SwitchCycle(source_index=jnp.array([0, 1, 1]), labels=("a", "b"))

        def loss(per_source):
            return jnp.sum(cycle.gather(per_source))

        grad = jax.grad(loss)(jnp.zeros((2, 2)))
        # source 0 selected once, source 1 twice
        np.testing.assert_allclose(np.asarray(grad), [[1.0, 1.0], [2.0, 2.0]])

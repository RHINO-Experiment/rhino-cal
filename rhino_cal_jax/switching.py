"""Draft Eqs. 11-12: the Dicke switch that chooses the connected source.

Equation 12 defines ``theta(t_j, phi_h)`` as one-hot over sources. Storing it
densely would be an ``(n_time, n_source)`` matrix of mostly zeros, so this
module stores the index and offers :meth:`SwitchCycle.one_hot` for the cases
where the matrix form is what a downstream formula wants. ``one_hot() @ x``
reproduces ``gather(x)`` only when ``x`` is 2-D: it is an ordinary matrix
product, not a gather, so for the 3-D stacked-coupling arrays this module's
own :meth:`SwitchCycle.gather` documents as first-class input, ``@`` either
raises (for a generic shape) or -- at the ``n_source == n_freq`` coincidence
-- silently returns something else instead. Use :meth:`SwitchCycle.gather`
itself, or an explicit ``einsum``, for anything higher-rank.

Why this layer exists: each source has its *own* reflection coefficient, so the
couplings differ per sample -- and that difference is the only thing that makes
the noise-wave temperatures identifiable. Count equations per frequency channel,
because the temperatures are functions of frequency and nothing ties channels
together a priori: each switch position contributes exactly one equation per
channel, so the design matrix has rank ``min(n_src, 3) * n_freq``. This counts
only ``T_unc, T_cos, T_sin`` with ``T_rx`` taken as known; if ``T_rx`` is also
free per channel the count becomes ``min(n_src, 4) * n_freq`` and four loads
are needed instead of three. One load leaves the three-unknown case deficient
by a factor of three, three loads make that case square, and that is why real
experiments switch between four or five calibrators. Sharing a single
``Gamma`` across the cycle collapses every source onto the same row.
"""

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from rhino_cal_jax.errors import ValidationError
from rhino_cal_jax.loads import Load


class SwitchCycle(eqx.Module):
    """Which source is connected at each time sample.

    Attributes:
        source_index: ``(n_time,)`` integer index into ``labels``.
        labels: source names, in the order the index refers to (static).
    """

    source_index: jax.Array = eqx.field(converter=jnp.asarray)
    # str(name), not just tuple(...): a caller who built labels from a numpy string
    # array (e.g. names[mask]) hands us numpy.str_ scalars, which equinox's static
    # field check treats as array-like (numpy.generic is in its array types) and
    # warns about. Coercing to plain str avoids a spurious warning on every such call.
    labels: tuple[str, ...] = eqx.field(
        static=True, converter=lambda labels: tuple(str(name) for name in labels)
    )

    def __check_init__(self):
        if not jnp.issubdtype(self.source_index.dtype, jnp.integer):
            raise ValidationError(
                f"source_index must be an integer array, got dtype "
                f"{self.source_index.dtype}. A float index would round silently "
                "and mis-assign samples to sources."
            )
        if self.source_index.ndim != 1:
            raise ValidationError(
                f"source_index must be 1D (n_time,), got ndim={self.source_index.ndim}."
            )
        if len(set(self.labels)) != len(self.labels):
            raise ValidationError(
                f"labels contain duplicates ({self.labels}); gathered results would "
                "not be attributable to a source."
            )
        # Range check needs concrete values, so it is skipped under tracing. An
        # out-of-range index would otherwise be clamped silently by JAX's gather
        # semantics and quietly assign samples to the wrong source.
        try:
            as_np = np.asarray(self.source_index)
        except (jax.errors.TracerArrayConversionError, TypeError):  # pragma: no cover
            return
        if as_np.size and (as_np.min() < 0 or as_np.max() >= len(self.labels)):
            raise ValidationError(
                f"source_index values span [{as_np.min()}, {as_np.max()}] which is "
                f"out of range for {len(self.labels)} labels {self.labels}."
            )

    @property
    def n_source(self) -> int:
        """Number of distinct sources in the cycle."""
        return len(self.labels)

    @property
    def n_time(self) -> int:
        """Number of time samples."""
        return int(self.source_index.shape[0])

    @classmethod
    def from_labels(
        cls, per_sample: Sequence[str], *, labels: tuple[str, ...]
    ) -> "SwitchCycle":
        """Build from an explicit label per time sample.

        Args:
            per_sample: ``(n_time,)`` source label for each sample.
            labels: the source ordering to index against.

        Raises:
            ValidationError: if a sample carries a label not in ``labels``.
        """
        lookup = {name: i for i, name in enumerate(labels)}
        unknown = sorted({s for s in per_sample if s not in lookup})
        if unknown:
            raise ValidationError(
                f"Sample labels {unknown} are not among the declared sources {labels}."
            )
        return cls(source_index=jnp.asarray([lookup[s] for s in per_sample]), labels=labels)

    @classmethod
    def from_schedule(
        cls,
        times,
        switch_times,
        switch_labels: Sequence[str],
        *,
        labels: tuple[str, ...],
    ) -> "SwitchCycle":
        """Build from a switch schedule: each sample takes the last state at or before it.

        Samples earlier than the first switch take the first state, matching
        ``utils.utils.assign_states`` in the numpy pipeline.

        Args:
            times: ``(n_time,)`` sample times.
            switch_times: ``(n_change,)`` times at which the state changes.
            switch_labels: ``(n_change,)`` source label taking effect at each change.
            labels: the source ordering to index against.

        Raises:
            ValidationError: if ``switch_labels`` and ``switch_times`` differ in length.
        """
        switch_times = np.asarray(switch_times)
        if len(switch_labels) != switch_times.shape[0]:
            raise ValidationError(
                f"switch_labels has {len(switch_labels)} entries but switch_times has "
                f"{switch_times.shape[0]}; each change needs exactly one label."
            )
        idx = np.searchsorted(switch_times, np.asarray(times), side="right") - 1
        idx = np.clip(idx, 0, len(switch_labels) - 1)
        return cls.from_labels([switch_labels[i] for i in idx], labels=labels)

    def one_hot(self) -> jax.Array:
        """``theta(t_j, phi_h)`` of Eq. 12 as a dense ``(n_time, n_source)`` matrix."""
        return jax.nn.one_hot(self.source_index, self.n_source)

    def gather(self, per_source: jax.Array) -> jax.Array:
        """Select each sample's connected source from a per-source array.

        Args:
            per_source: ``(n_source, ...)`` -- typically ``(n_source, n_freq)`` for
                one coupling, or ``(n_source, n_freq, 4)`` for stacked couplings.

        Returns:
            ``(n_time, ...)`` with the leading axis replaced by time. Samples
            whose index is out of range come back as NaN -- see below.

        Note:
            ``one_hot() @ per_source`` reproduces this only for 2-D
            ``per_source``. For the 3-D stacked-coupling case above, ``@`` is a
            plain matrix product over the leading two axes, not a gather: at
            ``n_source == n_freq`` it silently returns a same-shaped but
            numerically different array instead of raising. Use :meth:`gather`
            itself for anything of rank > 2.

        Note:
            ``__check_init__`` rejects an out-of-range ``source_index``, but that
            check needs concrete values and is skipped under tracing -- which is
            the production path. JAX's own gather semantics would then CLAMP the
            index and hand back a neighbouring source's couplings: finite,
            correctly shaped, and attributed to the wrong load. Out-of-range
            samples are therefore filled with NaN instead. A caller who means
            "this sample has no source" should say so downstream (mask it, or
            drop it from the likelihood); this method will not invent one.
            :meth:`one_hot` needs no such treatment -- an unmatched sample
            already gets an all-zero row, which selects nothing rather than
            something wrong.

        Raises:
            ValidationError: if the leading axis is not ``n_source``.
        """
        per_source = jnp.asarray(per_source)
        if per_source.ndim == 0 or per_source.shape[0] != self.n_source:
            got = "scalar" if per_source.ndim == 0 else str(per_source.shape[0])
            raise ValidationError(
                f"gather expects a leading n_source={self.n_source} axis, got {got}."
            )
        gathered = per_source[self.source_index]
        if not jnp.issubdtype(gathered.dtype, jnp.inexact):
            # Integer payloads have no NaN to fill with; clamping is all JAX
            # offers, so say nothing rather than pretend otherwise.
            return gathered
        in_range = (self.source_index >= 0) & (self.source_index < self.n_source)
        return jnp.where(
            in_range.reshape((-1,) + (1,) * (gathered.ndim - 1)),
            gathered,
            jnp.asarray(jnp.nan, dtype=gathered.dtype),
        )


def stack_load_gammas(loads: Sequence[Load]) -> tuple[jax.Array, tuple[str, ...]]:
    """Stack a set of loads into the ``(n_source, n_freq)`` form the switch needs.

    Args:
        loads: the switchable sources, in the order the switch will index them.

    Returns:
        ``(gamma_stacked, labels)`` -- a complex ``(n_source, n_freq)`` array and
        the matching label tuple, ready for
        :func:`~rhino_cal_jax.reflection.couplings` and
        :meth:`SwitchCycle.gather`.

    Raises:
        ValidationError: if ``loads`` is empty, the channel counts disagree, or
            two loads share a label.
    """
    loads = list(loads)
    if not loads:
        raise ValidationError("stack_load_gammas needs at least one load.")
    n_freq = {int(ld.gamma_src.shape[0]) for ld in loads}
    if len(n_freq) != 1:
        raise ValidationError(
            "Loads disagree on n_freq: "
            + ", ".join(f"{ld.label!r}={ld.gamma_src.shape[0]}" for ld in loads)
        )
    labels = tuple(ld.label for ld in loads)
    if len(set(labels)) != len(labels):
        raise ValidationError(f"Loads have duplicate labels: {labels}.")
    return jnp.stack([ld.gamma_src for ld in loads]), labels

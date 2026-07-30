"""Draft Eqs. 2-6: the coupling spectra a source imposes on the receiver.

These four quantities are what multiply the temperatures in Eq. 1. They depend
on the reflection coefficients alone -- no temperature enters -- which is why
they can be built once per source and reused for every time sample, and why
the resulting array is directly the design matrix of the linear system the
GCR solves.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from rhino_cal_jax._validation import require_complex, require_coupling_columns
from rhino_cal_jax.errors import ValidationError


def _require_matching_channels(gamma_src: jax.Array, gamma_rec: jax.Array) -> None:
    """Reject a ``gamma_rec`` whose channel axis does not match ``gamma_src``.

    NumPy broadcasting would happily stretch a length-1 or scalar ``gamma_rec``
    across every channel, applying one receiver reflection to the whole band and
    returning a finite, correctly-shaped, wrong result. Shape-only, so it stays
    safe under ``jit``.
    """
    if gamma_rec.ndim != 1:
        raise ValidationError(
            f"gamma_rec must be 1D (n_freq,), got shape {gamma_rec.shape}."
        )
    n_src = None if gamma_src.ndim == 0 else gamma_src.shape[-1]
    if n_src != gamma_rec.shape[0]:
        raise ValidationError(
            f"gamma_src has {n_src} channels but gamma_rec has "
            f"{gamma_rec.shape[0]}. A mismatched gamma_rec would broadcast "
            "silently and apply one receiver reflection to every channel."
        )


def reflection_factor(gamma_src: jax.Array, gamma_rec: jax.Array) -> jax.Array:
    """``F = sqrt(1 - |Gamma_rec|^2) / (1 - Gamma_src Gamma_rec)`` (draft Eq. 3).

    Args:
        gamma_src: complex source reflection coefficient, ``(..., n_freq)``.
        gamma_rec: complex receiver reflection coefficient, ``(n_freq,)``.

    Returns:
        Complex ``F``, broadcast to the shape of ``gamma_src``.

    Raises:
        ValidationError: if either coefficient has a real dtype.

    Note:
        Three extreme cases, checked directly rather than assumed:

        * ``|Gamma_rec| > 1`` (an active receiver, unphysical) yields NaN
          directly: ``1 - |Gamma_rec|^2`` is then negative, and its real
          square root is NaN. Nothing to do with an infinite modulus.
        * The resonance ``Gamma_src Gamma_rec -> 1``, approached with both
          ``|Gamma| < 1`` (the physically reachable case), does *not* yield
          NaN -- it gives a large but finite ``F`` (e.g. ``|F| ~ 7071`` at
          ``Gamma_src = Gamma_rec = 1 - 1e-8``).
        * Only the exact coincidence ``Gamma_src = Gamma_rec = 1`` yields
          NaN, and there because numerator and denominator vanish together
          (``sqrt(1 - |Gamma_rec|^2) = 0`` and ``1 - Gamma_src Gamma_rec =
          0``) -- a genuine ``0 / 0``, not an overflow.

        All three are left loud on purpose: nothing here special-cases the
        pole or the overunity input.

    Note:
        ``1 - |Gamma_rec|^2`` is a difference of nearly-equal numbers once
        ``|Gamma_rec|`` approaches 1, so precision cancels there -- silently:
        a finite, correctly-shaped, increasingly wrong ``c_src``. Measured
        relative error of ``Couplings.c_src`` in float32 against a float64
        reference, at ``Gamma_src = 0.3 + 0.1j``:

        * ``1 - |Gamma_rec|^2 = 1e-1`` -> 2.7e-7
        * ``1 - |Gamma_rec|^2 = 1e-3`` -> 4.7e-5
        * ``1 - |Gamma_rec|^2 = 1e-5`` -> 1.4e-3
        * ``1 - |Gamma_rec|^2 = 1e-7`` -> 0.19

        Every ``gamma_rec`` in this package's README example and default test
        fixtures is O(0.03-0.3); this suite's own boundary sweep pushes
        ``|Gamma_rec|`` up to 0.999 (and to the unphysical 1.0-1.5 for the NaN
        tests above), but always under this suite's float64, so none of that
        exercises the float32 issue described here. It would bite a float32
        gradient sampler exploring ``gamma_rec`` as a fitted parameter with no
        box constraint. Run float64 (as this package's own test suite does;
        see ``tests/conftest.py``), or keep ``|Gamma_rec|`` away from the unit
        circle. No runtime guard is added: the threshold is precision- and
        use-dependent, and a value check would not be jit-safe.
    """
    gamma_src = require_complex("gamma_src", gamma_src)
    gamma_rec = require_complex("gamma_rec", gamma_rec)
    _require_matching_channels(gamma_src, gamma_rec)
    return jnp.sqrt(1.0 - jnp.abs(gamma_rec) ** 2) / (1.0 - gamma_src * gamma_rec)


class Couplings(eqx.Module):
    """The four coupling spectra of draft Eq. 1, for one or many sources.

    Attributes:
        c_src: ``(1 - |Gamma|^2) |F|^2`` -- the source term (Eq. 2).
        k_unc: ``|Gamma|^2 |F|^2`` -- the uncorrelated noise wave (Eq. 4).
        k_cos: ``Re(Gamma F)`` -- the in-phase noise wave (Eq. 5).
        k_sin: ``Im(Gamma F)`` -- the quadrature noise wave (Eq. 6).

    All four share the shape of ``gamma_src``: ``(n_freq,)`` for a single
    source, ``(n_source, n_freq)`` for a switched set.
    """

    c_src: jax.Array = eqx.field(converter=jnp.asarray)
    k_unc: jax.Array = eqx.field(converter=jnp.asarray)
    k_cos: jax.Array = eqx.field(converter=jnp.asarray)
    k_sin: jax.Array = eqx.field(converter=jnp.asarray)

    def __check_init__(self):
        # Without this, mismatched fields construct happily and only fail later
        # inside `stacked`, with a message that names no field.
        shapes = {
            "c_src": self.c_src.shape, "k_unc": self.k_unc.shape,
            "k_cos": self.k_cos.shape, "k_sin": self.k_sin.shape,
        }
        if len(set(shapes.values())) != 1:
            raise ValidationError(
                f"Couplings fields must all share one shape, got {shapes}."
            )

    @property
    def stacked(self) -> jax.Array:
        """``(..., n_freq, 4)`` in the order ``(c_src, k_unc, k_cos, k_sin)``.

        This is the design matrix: contracting it with the temperature vector
        ``(T_src, T_unc, T_cos, T_sin)`` reproduces every term of Eq. 1 except
        the receiver offset ``T_rx``, which has no coupling of its own.

        Deliberately not cached: it is a pure ``jnp.stack`` over four leaves
        already in hand, so XLA folds it away on the jitted path this module is
        built for, while a cached attribute would complicate pytree flattening
        of a frozen Module for no real saving.
        """
        return jnp.stack([self.c_src, self.k_unc, self.k_cos, self.k_sin], axis=-1)

    @classmethod
    def from_stacked(cls, stacked: jax.Array) -> "Couplings":
        """Inverse of :attr:`stacked`: split a ``(..., n_freq, 4)`` array apart.

        Exists so that no call site ever hand-unpacks the column indices. A
        transposed index somewhere downstream would yield a ``Couplings`` that
        is finite, correctly shaped and wrong; defining the order in exactly one
        place is what prevents it.

        Args:
            stacked: ``(..., n_freq, 4)`` ordered as :attr:`stacked` produces.

        Returns:
            The corresponding :class:`Couplings`.

        Raises:
            ValidationError: if the trailing axis is not the four couplings.
        """
        stacked = require_coupling_columns("from_stacked", stacked)
        return cls(
            c_src=stacked[..., 0], k_unc=stacked[..., 1],
            k_cos=stacked[..., 2], k_sin=stacked[..., 3],
        )


def couplings(gamma_src: jax.Array, gamma_rec: jax.Array) -> Couplings:
    """Build the coupling spectra of draft Eqs. 2-6.

    Args:
        gamma_src: complex source reflection coefficient. ``(n_freq,)`` for one
            source, or ``(n_source, n_freq)`` for a switched set.
        gamma_rec: complex receiver reflection coefficient, ``(n_freq,)``,
            broadcast across the source axis.

    Returns:
        A :class:`Couplings` whose fields share the shape of ``gamma_src``.

    Raises:
        ValidationError: if either coefficient has a real dtype.
    """
    f = reflection_factor(gamma_src, gamma_rec)
    gamma_src = jnp.asarray(gamma_src)
    abs2_src = jnp.abs(gamma_src) ** 2
    abs2_f = jnp.abs(f) ** 2
    product = gamma_src * f
    return Couplings(
        c_src=(1.0 - abs2_src) * abs2_f,
        # D1: the draft's Eq. 4 prints a single |F|; Eq. 2 shows both squares,
        # the numpy reference squares it, and so does the transfer matrix in
        # gcr/transfer_matrix_construction.py. |F|^2 it is.
        k_unc=abs2_src * abs2_f,
        k_cos=jnp.real(product),
        k_sin=jnp.imag(product),
    )

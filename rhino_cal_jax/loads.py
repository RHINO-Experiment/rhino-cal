"""Reflection coefficients for calibration loads and the receiver.

The numpy repository builds ``Gamma`` with three different cable conventions:
``simulation/loads.py::Load`` uses a one-way phase ``-2 pi L nu / c`` with no
velocity factor and squares ``s21``; ``receiver_simulation.calculate_cable_params``
divides that phase by a velocity factor; and ``TerminatedCable`` writes the
round trip directly as ``exp(-4j pi nu L eps / c)``. This module implements the
``Load`` convention with an explicit ``velocity_factor`` defaulting to 1.0, so
it reduces to ``Load`` exactly and reaches the other two by argument.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from rhino_cal_jax.errors import ValidationError

SPEED_OF_LIGHT: float = 299792458.0
"""Vacuum speed of light [m/s] -- matches astropy.constants.c to the metre."""

_TERMINATIONS = ("open", "short", "matched", "resistive")


def _require_complex(name: str, value: jax.Array) -> jax.Array:
    """Reject a real reflection coefficient.

    A real ``Gamma`` makes the quadrature coupling ``Im(Gamma F)`` identically
    zero, so ``T_sin`` silently drops out of the model downstream: a finite,
    correctly-shaped, wrong answer.
    """
    value = jnp.asarray(value)
    if not jnp.issubdtype(value.dtype, jnp.complexfloating):
        raise ValidationError(
            f"{name} must be complex (got dtype {value.dtype}); a real reflection "
            "coefficient silently zeroes the sine coupling."
        )
    return value


def termination_gamma(
    kind: str,
    n_freq: int,
    *,
    impedance: float | None = None,
    z0: float = 50.0,
) -> jax.Array:
    """Reflection coefficient of an ideal termination, one value per channel.

    Args:
        kind: ``"open"`` (+1), ``"short"`` (-1), ``"matched"`` (0), or
            ``"resistive"`` (``(Z - Z0) / (Z + Z0)``).
        n_freq: number of channels; the value is constant across them.
        impedance: termination impedance [Ohm]; required for ``"resistive"``.
        z0: characteristic impedance [Ohm].

    Returns:
        A complex ``(n_freq,)`` array.

    Raises:
        ValidationError: on an unknown ``kind``, or ``"resistive"`` with no
            ``impedance``.
    """
    if kind not in _TERMINATIONS:
        raise ValidationError(
            f"Unknown termination {kind!r}; expected one of {_TERMINATIONS}."
        )
    if kind == "open":
        value = 1.0 + 0.0j
    elif kind == "short":
        value = -1.0 + 0.0j
    elif kind == "matched":
        value = 0.0 + 0.0j
    else:
        if impedance is None:
            raise ValidationError(
                "termination_gamma('resistive', ...) needs an impedance [Ohm]."
            )
        value = complex((impedance - z0) / (impedance + z0), 0.0)
    return jnp.full((n_freq,), value, dtype=complex)


def cable_gamma(
    gamma_termination: jax.Array,
    freq: jax.Array,
    *,
    length: float | jax.Array,
    velocity_factor: float | jax.Array = 1.0,
    loss: float | jax.Array = 1.0,
) -> jax.Array:
    """Move a termination behind a length of cable.

    The signal traverses the cable twice, so the one-way transmission
    ``s21 = loss * exp(-2j pi L nu / (vf c))`` is applied squared.

    Args:
        gamma_termination: complex ``(n_freq,)`` reflection at the far end.
        freq: channel frequencies [Hz], ``(n_freq,)``.
        length: physical cable length [m].
        velocity_factor: propagation velocity as a fraction of ``c``.
        loss: one-way amplitude transmission (1.0 = lossless).

    Returns:
        The complex ``(n_freq,)`` reflection seen at the near end.
    """
    phase = -2.0 * jnp.pi * length * jnp.asarray(freq) / (velocity_factor * SPEED_OF_LIGHT)
    s21 = loss * jnp.exp(1j * phase)
    return s21 * s21 * jnp.asarray(gamma_termination)


class Load(eqx.Module):
    """A source that can be switched to the receiver input.

    Covers the antenna and every calibration load alike -- from the model's
    point of view they differ only in ``gamma_src`` and ``t_src``.

    Attributes:
        gamma_src: complex ``(n_freq,)`` reflection coefficient.
        t_src: noise temperature [K]; scalar, ``(n_freq,)`` or
            ``(n_time, n_freq)``.
        label: identifier used by the switch cycle (static).
    """

    gamma_src: jax.Array = eqx.field(converter=jnp.asarray)
    t_src: jax.Array = eqx.field(converter=jnp.asarray)
    label: str = eqx.field(static=True)

    def __check_init__(self):
        _require_complex("gamma_src", self.gamma_src)
        if self.gamma_src.ndim != 1:
            raise ValidationError(
                f"gamma_src must be 1D (n_freq,), got ndim={self.gamma_src.ndim}."
            )


class Receiver(eqx.Module):
    """The receiver: its input reflection coefficient and its power gain.

    Attributes:
        gamma_rec: complex ``(n_freq,)`` receiver reflection coefficient.
        gain: ``G(nu)`` as ``(n_freq,)``, ``G(nu, t)`` as ``(n_time, n_freq)``,
            or a scalar.
    """

    gamma_rec: jax.Array = eqx.field(converter=jnp.asarray)
    gain: jax.Array = eqx.field(converter=jnp.asarray)

    def __check_init__(self):
        _require_complex("gamma_rec", self.gamma_rec)
        if self.gamma_rec.ndim != 1:
            raise ValidationError(
                f"gamma_rec must be 1D (n_freq,), got ndim={self.gamma_rec.ndim}."
            )
        if self.gain.ndim not in (0, 1, 2):
            raise ValidationError(f"gain must be 0/1/2-D, got ndim={self.gain.ndim}.")
        if self.gain.ndim >= 1 and self.gain.shape[-1] != self.gamma_rec.shape[0]:
            raise ValidationError(
                f"gain has n_freq={self.gain.shape[-1]} but gamma_rec has "
                f"n_freq={self.gamma_rec.shape[0]}."
            )

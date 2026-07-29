"""Exception hierarchy.

One root (:class:`RhinoCalError`) so callers can catch everything this package
raises, and leaf classes that also subclass the builtin a caller would
naturally reach for.
"""


class RhinoCalError(Exception):
    """Base class for every error raised by rhino_cal_jax."""


class ValidationError(RhinoCalError, ValueError):
    """An input failed a structural check (shape, dtype, or declared size)."""

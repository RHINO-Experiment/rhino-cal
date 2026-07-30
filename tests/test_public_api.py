"""Everything a user needs is reachable from the package root."""

import jax
import jax.numpy as jnp
import numpy as np

import rhino_cal_jax


def test_the_documented_surface_is_exported():
    expected = {
        "Couplings", "Load", "Receiver", "RhinoCalError", "SPEED_OF_LIGHT",
        "SwitchCycle", "ValidationError", "add_radiometer_noise", "cable_gamma",
        "couplings", "design_matrix", "radiometer_power", "reflection_factor",
        "stack_load_gammas", "synchrotron_temperature", "system_temperature",
        "termination_gamma",
    }
    assert set(rhino_cal_jax.__all__) == expected
    # __version__ is public but not part of the star-import surface.
    assert rhino_cal_jax.__version__


def test_the_exported_names_have_the_documented_kind():
    """Each exported name is the kind of object its use elsewhere requires.

    Replaces the old `test_every_exported_name_actually_resolves`, which could
    not fail: every name in `__all__` is eagerly bound by a top-of-file
    import, so `getattr(...) is not None` was true by construction for
    anything that already passed the set-equality check above. This instead
    probes the actual *kind* of each export -- classes are classes, exceptions
    are exceptions, functions are callable, and the one constant is a plain
    float -- which would catch e.g. a class shadowed by a same-named function.
    """
    classes = {"Couplings", "Load", "Receiver", "SwitchCycle"}
    exceptions = {"RhinoCalError", "ValidationError"}
    constants = {"SPEED_OF_LIGHT"}
    functions = set(rhino_cal_jax.__all__) - classes - exceptions - constants

    for name in classes:
        assert isinstance(getattr(rhino_cal_jax, name), type)
    for name in exceptions:
        obj = getattr(rhino_cal_jax, name)
        assert isinstance(obj, type) and issubclass(obj, Exception)
    for name in functions:
        assert callable(getattr(rhino_cal_jax, name))
    for name in constants:
        assert isinstance(getattr(rhino_cal_jax, name), float)


def test_speed_of_light_is_bit_identical_to_astropy():
    """Pins the claim in loads.py's docstring for the re-exported constant."""
    import astropy.constants as const

    assert rhino_cal_jax.SPEED_OF_LIGHT == const.c.si.value


def test_an_end_to_end_switched_observation_runs():
    """The smallest complete use: three loads, a switch cycle, noise.

    Three is the minimum that makes the per-channel noise-wave system square --
    each switch position contributes one equation per frequency channel. That
    counts only (T_unc, T_cos, T_sin) with T_rx taken as known, as it is here
    (t_rx=290.0 is a fixed argument, not solved for); with T_rx also free per
    channel, four loads would be needed instead of three.
    """
    n_freq = 8
    freq = jnp.linspace(60e6, 85e6, n_freq)

    antenna = rhino_cal_jax.Load(
        gamma_src=rhino_cal_jax.cable_gamma(
            rhino_cal_jax.termination_gamma("open", n_freq), freq, length=2.0, loss=0.9
        ),
        t_src=rhino_cal_jax.synchrotron_temperature(freq),
        label="antenna",
    )
    ambient = rhino_cal_jax.Load(
        gamma_src=rhino_cal_jax.termination_gamma("resistive", n_freq, impedance=52.0),
        t_src=jnp.array(300.0),
        label="ambient",
    )
    hot = rhino_cal_jax.Load(
        gamma_src=rhino_cal_jax.cable_gamma(
            rhino_cal_jax.termination_gamma("short", n_freq), freq, length=0.4, loss=0.98
        ),
        t_src=jnp.array(400.0),
        label="hot",
    )
    receiver = rhino_cal_jax.Receiver(
        gamma_rec=rhino_cal_jax.termination_gamma("resistive", n_freq, impedance=45.0),
        gain=jnp.full(n_freq, 1000.0),
    )

    gamma_stacked, labels = rhino_cal_jax.stack_load_gammas([antenna, ambient, hot])
    cycle = rhino_cal_jax.SwitchCycle.from_labels(list(labels) * 4, labels=labels)

    coup = rhino_cal_jax.Couplings.from_stacked(
        cycle.gather(rhino_cal_jax.couplings(gamma_stacked, receiver.gamma_rec).stacked)
    )
    t_src = cycle.gather(
        jnp.stack([jnp.broadcast_to(ld.t_src, (n_freq,))
                   for ld in (antenna, ambient, hot)])
    )
    t_sys = rhino_cal_jax.system_temperature(
        coup, t_src=t_src, t_unc=250.0, t_cos=30.0, t_sin=-40.0, t_rx=290.0
    )
    power = rhino_cal_jax.add_radiometer_noise(
        rhino_cal_jax.radiometer_power(t_sys, receiver.gain),
        jax.random.key(0), t_int=1.0, delta_nu=1e4,
    )

    assert power.shape == (12, n_freq)
    assert np.all(np.isfinite(np.asarray(power)))
    assert np.all(np.asarray(power) > 0.0)


def test_the_whole_forward_model_is_differentiable_end_to_end():
    """The reason this package exists: gradients reach the noise waves."""
    n_freq = 4
    freq = jnp.linspace(60e6, 85e6, n_freq)
    gamma_stacked = jnp.stack([
        rhino_cal_jax.cable_gamma(
            rhino_cal_jax.termination_gamma("open", n_freq), freq, length=2.0
        ),
        rhino_cal_jax.termination_gamma("resistive", n_freq, impedance=52.0),
    ])
    gamma_rec = rhino_cal_jax.termination_gamma("resistive", n_freq, impedance=45.0)
    cycle = rhino_cal_jax.SwitchCycle(
        source_index=jnp.arange(6) % 2, labels=("antenna", "ambient")
    )
    coup = rhino_cal_jax.Couplings.from_stacked(
        cycle.gather(rhino_cal_jax.couplings(gamma_stacked, gamma_rec).stacked)
    )

    def total(t_nw):
        t_sys = rhino_cal_jax.system_temperature(
            coup, t_src=300.0, t_unc=t_nw[0], t_cos=t_nw[1], t_sin=t_nw[2], t_rx=290.0
        )
        return jnp.sum(rhino_cal_jax.radiometer_power(t_sys, gain=1000.0))

    grad = jax.grad(total)(jnp.array([250.0, 30.0, -40.0]))
    assert np.all(np.isfinite(np.asarray(grad)))
    assert np.all(np.abs(np.asarray(grad)) > 0.0)


def test_the_design_matrix_is_reachable_from_the_root():
    """couplings -> switch -> design matrix, the identifiability diagnostic.

    Not rheplicant's GCR entry point -- that is built by autodiff directly
    around ``Couplings.stacked``. This just confirms the diagnostic chain is
    reachable from the package root.
    """
    n_freq = 4
    gamma_stacked = jnp.stack([
        jnp.full(n_freq, 0.30 + 0.10j), jnp.full(n_freq, 0.02 + 0.00j)
    ])
    gamma_rec = jnp.full(n_freq, 0.08 - 0.03j)
    cycle = rhino_cal_jax.SwitchCycle(source_index=jnp.arange(4) % 2, labels=("a", "b"))
    matrix = rhino_cal_jax.design_matrix(
        cycle.gather(rhino_cal_jax.couplings(gamma_stacked, gamma_rec).stacked)
    )
    assert matrix.shape == (16, 4)

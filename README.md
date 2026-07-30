# RHINO Calibration

Pipeline for the calibration of raw RHINO data and the eventual extraction of $T_{\rm ant}({\rm LST})$

Also includes simulation notebooks for testing the calibration on ideal cases and exploring limitations.

## `rhino_cal_jax` — the differentiable data model

A JAX/Equinox implementation of the noise-wave data model (Eq. 1 of the
Noise-Wave GCR note), independent of the numpy pipeline above and verified
against it channel by channel: 256 parameter cells agree to `1e-13` relative,
spanning matched to near-total reflection on both ports.

```bash
pip install -e .
pytest tests/
```

```python
import jax, jax.numpy as jnp, rhino_cal_jax as rcj

n_freq = 8
freq = jnp.linspace(60e6, 85e6, n_freq)
loads = [
    rcj.Load(gamma_src=rcj.termination_gamma("open", n_freq), t_src=jnp.array(0.0),
             label="antenna"),
    rcj.Load(gamma_src=rcj.termination_gamma("resistive", n_freq, impedance=52.0),
             t_src=jnp.array(300.0), label="ambient"),
    rcj.Load(gamma_src=rcj.termination_gamma("short", n_freq), t_src=jnp.array(400.0),
             label="hot"),
]
gamma, labels = rcj.stack_load_gammas(loads)
cycle = rcj.SwitchCycle.from_labels(list(labels) * 4, labels=labels)
coup = rcj.Couplings.from_stacked(
    cycle.gather(rcj.couplings(gamma, rcj.termination_gamma("resistive", n_freq,
                                                            impedance=45.0)).stacked)
)
# stack_load_gammas stacks only gamma_src -- t_src is not carried along with it,
# so it must be gathered separately, in the same load order, or it silently
# falls out of sync with the couplings above.
t_src = cycle.gather(jnp.stack([jnp.broadcast_to(ld.t_src, (n_freq,)) for ld in loads]))
t_sys = rcj.system_temperature(coup, t_src=t_src, t_unc=250.0, t_cos=30.0,
                               t_sin=-40.0, t_rx=290.0)
power = rcj.radiometer_power(t_sys, gain=1000.0)
```

### Where it differs from `simulation/`, deliberately

| | `simulation/` | `rhino_cal_jax` |
|---|---|---|
| `Γ` per switched source | one shared value | one per source, gathered by the switch |
| noisy power | `abs(P + n)` | `P (1 + w)`; the fold is opt-in via `fold_negative=` |
| cable phase | three conventions across three modules | one, with an explicit `velocity_factor` |
| `κ_unc` | `\|Γ\|² \|F\|²` | same — note the note's Eq. 4 prints a single `\|F\|` |

The first row is the one that matters scientifically. Each switch position
contributes one equation per frequency channel, so with per-channel noise-wave
temperatures the design matrix has rank `min(n_src, 3) × n_freq`: one load is
never enough, and three is the minimum that makes the system square. This
counts only `T_unc, T_cos, T_sin` with `T_rx` taken as known -- with `T_rx`
also free per channel the count becomes `min(n_src, 4) × n_freq` and four
loads are needed. That is why EDGES and REACH switch between four or five
calibrators.

The second row is not cosmetic either: folding the negative tail biases the
mean upward (towards `E|N(1,1)| = 1.167` at `σ_w = 1`) and breaks the Gaussian
likelihood a GCR sampler assumes.

### Known trap in the numpy side

`simulation/toy_sky.py::synchrotron_temperatures` divides a Hz-valued `freqs`
by `210 * un.MHz`, so its returned `Quantity` carries an **unsimplified** unit
`MHz^(13/5) / Hz^(13/5)`. Both `np.asarray(...)` and `.value` then hand back a
number that is too small by `(10^6)^2.6 ≈ 4 × 10^15`, with no error. Use
`.to_value(un.dimensionless_unscaled)`.

What was actually checked: `compute_radiometer_power`'s own arithmetic stays
inside astropy and simplifies the unit correctly when its inputs are
`Quantity`s, so this trap does not bite *that* function's return value — it is
consumers who strip units directly, via `np.asarray(...)` or `.value` as
above, who get bitten. This was not traced further into `ObservationHandler`,
which assigns each spectrum into a bare `np.empty(...)` array; whether that
path is exposed to the same trap for a `ToySky`-sourced antenna depends on
whether the other temperature terms it combines with carry units, which was
not checked.
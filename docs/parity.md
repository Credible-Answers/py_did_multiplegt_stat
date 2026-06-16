# Stata-to-Python parity

This page enumerates every Stata option in `did_multiplegt_stat` and its Python equivalent.

## How to read this page

- **Stata** column shows the exact option name as it appears in `did_multiplegt_stat.ado`.
- **Python kwarg** is what you pass to either `DIDMultiplegtStat(...)` or the functional
  `did_multiplegt_stat(...)`.
- **Notes** flag any semantic differences or wrapper behaviour.

## Positional arguments

| Stata | Python | Notes |
|---|---|---|
| `Y` *(1st varlist token)* | `Y="..."` | Outcome |
| `G` *(2nd)* | `ID="..."` | Unit identifier |
| `T` *(3rd)* | `Time="..."` | Period |
| `D` *(4th)* | `D="..."` | Treatment |
| `Z` *(5th, optional)* | `Z="..."` | Instrument (only with `estimator="ivwaoss"`) |

## Estimator selection

| Stata | Python | Notes |
|---|---|---|
| `estimator(as)` | `estimator="aoss"` | Internal name; the table label is still `AS`. |
| `estimator(was)` | `estimator="waoss"` | |
| `estimator(iv-was)` | `estimator="ivwaoss"` | Requires `Z`. |
| `estimator(as was)` | `estimator=["aoss", "waoss"]` | Both at once. |
| `estimation_method(ra|ps|dr)` | `estimation_method="ra"` / `"ps"` / `"dr"` | Default `"dr"` (or `"ra"` if `exact_match`). |

## Polynomial order

| Stata | Python |
|---|---|
| `or(1)` | `order=1` |
| `or(1 4 3 2)` | `order=[1, 4, 3, 2]` |
| `or(1 4 3 2 1 2 3 4)` | `order=[1, 4, 3, 2, 1, 2, 3, 4]` |

## Sample-restriction options

| Stata | Python | Notes |
|---|---|---|
| `exact_match` | `exact_match=True` | |
| `noextrapolation` | `noextrapolation=True` | |
| `switchers(up)` / `switchers(down)` | `switchers="up"` / `"down"` | |
| `on_placebo_sample` | `on_placebo_sample=True` | Cannot be combined with `placebo` or `ivwaoss`. |

## Heterogeneity options

| Stata | Python | Notes |
|---|---|---|
| `bysort g:` (prefix) | `by=["g"]` | Only time-invariant variables allowed. |
| `by_fd(K)` | `by_fd=K` | Quantile-bin switchers by `|ΔD|`. |
| `by_baseline(K)` | `by_baseline=K` | Quantile-bin by `D_{t-1}`. |
| `disaggregate` | `disaggregate=True` | |
| `as_vs_was` | `aoss_vs_waoss=True` | Requires both `aoss` and `waoss`. |

## Controls / weights / cluster

| Stata | Python | Notes |
|---|---|---|
| `controls(varlist)` | `controls=["v1", "v2"]` | |
| `weights(varname)` | `weight="v"` | **Renamed** — Stata is plural, Python is singular. |
| `cluster(varlist)` | `cluster="v"` | Python accepts only a single var. |
| `other_treatments(varlist)` | `other_treatments=["v1", "v2"]` | |

## Inference

| Stata | Python |
|---|---|
| `placebo(N)` | `placebo=N` |
| `bootstrap(N)` | `bootstrap=N` |
| `seed(N)` | `seed=N` |
| `trimming(N)` | `trimming=N` (1–100, percentage) |
| `cross_fitting(K)` | `cross_fitting=K` |

## TWFE comparison

Stata uses a string suboption: `twfe(same_sample percentile)`. Python uses a dict.

| Stata | Python |
|---|---|
| `twfe(same_sample)` | `twfe={"same_sample": True}` |
| `twfe(full_sample)` | `twfe={"full_sample": True}` |
| `twfe(percentile)` | `twfe={"percentile": True}` |
| `twfe(same_sample percentile)` | `twfe={"same_sample": True, "percentile": True}` |

`twfe=True` alone defaults to the normal-approximation, ambiguous-sample variant — for new
code we recommend always passing an explicit dict.

## Cross-validation

Stata uses a string suboption: `cross_validation(kfolds tolerance(0.01) max_k(5))`.
Python uses a dict:

```python
cross_validation={
    "algorithm": "kfolds",
    "tolerance": 0.01,
    "max_k": 5,
    "kfolds": 5,
    "seed": 0,
    "same_order_all_logits": False,
}
```

## Display options

| Stata | Python |
|---|---|
| `graph_off` | *(no plot is drawn unless you call `.plot()`)* |
| `bys_graph_off` | *(same as above)* |

The Stata `did_multiplegt_stat` always prints a graph by default; the Python version only
plots on explicit `.plot()` because libraries should not draw GUI windows from a fit call.

## Python-only options

| Python | Default | Meaning |
|---|---|---|
| `asinstata` | `False` | If `True`, use Stata-faithful regressions (statsmodels OLS + custom Newton-Raphson logit). |
| `iv_method` | `"manual"` | IV backend in the TWFE comparison: `"manual"`, `"linearmodels"`, or `"econtools"`. |
| `model_deltay` | `None` | Custom sklearn-style regressor for $E[\Delta Y \mid D_{t-1}, S=0]$. |
| `model_stayer` | `None` | Custom sklearn-style classifier for $P(S=0 \mid D_{t-1})$. |

## Reproducing Stata numbers exactly

If you need byte-level parity with the Stata ado-file:

```python
res = did_multiplegt_stat(df, Y, ID, Time, D, Z=Z,
                          asinstata=True,          # critical
                          # ... other options matching the Stata call
                          )
```

With `asinstata=True` and identical inputs, agreement is typically:

- Point estimates: ≲ 1e-7 relative error.
- Standard errors: ≲ 1e-6 relative error.
- Bootstrap CIs: not deterministic across runtimes — the MT19937-64 RNG is
  bit-equivalent to Stata, but pandas operations applied to bootstrap-resampled data
  can reorder rows differently across machines.

For the bootstrap, you can also import Stata's fold IDs via `cf_folds_file=` and
guarantee an exact CF replication.

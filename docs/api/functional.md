# Functional API

The functional API mirrors the Stata command surface: a single call that takes the panel,
returns a dict.

```python
from did_multiplegt_stat import did_multiplegt_stat, summary_did_multiplegt_stat
```

## `did_multiplegt_stat`

::: did_multiplegt_stat.did_multiplegt_stat
    options:
      show_signature: true

## `summary_did_multiplegt_stat`

::: did_multiplegt_stat.summary_did_multiplegt_stat

## `print_did_multiplegt_stat`

::: did_multiplegt_stat.print_did_multiplegt_stat

## Return value (`dict`)

The returned dict has the following keys (omitting placebo / by-group blocks when not
requested):

| Key | Type | Description |
|---|---|---|
| `args` | `dict` | A snapshot of every option passed to `did_multiplegt_stat`. |
| `results` | `dict` | The single main-results block when no `by`/`by_fd`/`by_baseline`. |
| `results_by_{j}` | `dict` | Per-by-group results block when `by`/`by_fd`/`by_baseline` is set. `j` runs from 1. |
| `by_levels` | `list` | Levels in order, so `by_levels[j-1]` matches `results_by_{j}`. |
| `first_stage` | `dict` | Same shape as the top-level dict, returned by the inner first-stage call when `estimator="ivwaoss"`. |
| `twfe_comparison` | `pd.DataFrame` | Bootstrap-based TWFE comparison table when `twfe=True`. |
| `val_quantiles` | `list` | Quantile cut-points when `by_fd` / `by_baseline` is set. |
| `switch_df` | `pd.DataFrame` | Per-bin switcher count + median `|ΔD|` when `by_fd`. |
| `_class` | `str` | Always `"did_multiplegt_stat"`. Useful sentinel for type checks. |

Inside each `results*` block:

| Key | Type | Description |
|---|---|---|
| `table` | `pd.DataFrame` | Main effects table for this block. |
| `table_placebo_{p}` | `pd.DataFrame` | Placebo table for placebo `p ∈ {1, …, N}`. |
| `N` | `int` | Number of observations used in this block. |
| `n_clusters` | `int` | Set only when `cluster=` was used. |
| `pairs` | `int` | Number of consecutive-period pairs in the panel. |
| `aoss_vs_waoss` | `pd.DataFrame` | Difference-test table when `aoss_vs_waoss=True`. |

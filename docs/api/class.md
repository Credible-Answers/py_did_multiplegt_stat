# `DIDMultiplegtStat` — class API

The scikit-learn style class is the recommended entry point.

```python
from did_multiplegt_stat import DIDMultiplegtStat
```

## Constructor

::: did_multiplegt_stat.DIDMultiplegtStat
    options:
      members:
        - __init__

## Fitting

::: did_multiplegt_stat.DIDMultiplegtStat.fit

## Inspection

::: did_multiplegt_stat.DIDMultiplegtStat.summary
::: did_multiplegt_stat.DIDMultiplegtStat.to_dataframe
::: did_multiplegt_stat.DIDMultiplegtStat.get_coefficients
::: did_multiplegt_stat.DIDMultiplegtStat.get_confidence_intervals

## Plotting

::: did_multiplegt_stat.DIDMultiplegtStat.plot
::: did_multiplegt_stat.DIDMultiplegtStat.plot_comparison

## Parameter management (sklearn-style)

::: did_multiplegt_stat.DIDMultiplegtStat.get_params
::: did_multiplegt_stat.DIDMultiplegtStat.set_params

## Attributes (set after `fit`)

| Attribute | Type | Description |
|---|---|---|
| `results_` | `dict` | Full results dictionary (source of truth). |
| `table_` | `pd.DataFrame` | Main results table: `Estimate, SE, LB CI, UB CI, Switchers, Stayers`. |
| `placebo_tables_` | `dict[int, pd.DataFrame] \| None` | Placebo tables keyed by placebo index. |
| `n_obs_` | `int` | Number of observations. |
| `n_clusters_` | `int \| None` | Number of clusters, when `cluster=` is set. |
| `by_levels_` | `list \| None` | Levels of by-group analysis. |
| `first_stage_` | `DIDMultiplegtStat \| None` | Nested fitted model for the first stage of IV-WAS. |
| `is_fitted_` | `bool` | Whether `.fit()` has been called. |

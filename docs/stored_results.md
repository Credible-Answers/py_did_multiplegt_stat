# Stored results

In Stata, `did_multiplegt_stat` is an `eclass` program. In Python, the same information is
available either as attributes on the fitted `DIDMultiplegtStat` model or as keys in the
dict returned by the functional API.

## Stata `e()`-results &harr; Python equivalents

| Stata | Python (class)                | Python (functional dict)             |
|---|---|---|
| `e(N)` | `model.n_obs_`                | `res["results"]["N"]`                |
| `e(depvar)` | `model._Y`                    | `res["args"]["Y"]`                   |
| `e(b)` | `model.get_coefficients()`    | from `res["results"]["table"]`       |
| `e(V)` *(variance matrix)* | derived from `model.table_["SE"]` | `res["results"]["table"]["SE"]`        |

### Effects' estimation

| Stata | Python                                                                                      |
|---|---|
| `e(AS)` | `model.to_dataframe()` &nbsp;or&nbsp; `res["results"]["table"]` &nbsp;— rows labelled `AS`, `as_2`, ... |
| `e(WAS)` | rows labelled `WAS`, `was_2`, ...                                                            |
| `e(IWAS)` | rows labelled `IWAS`, `iwas_2`, ...                                                          |

### Placebos' estimation

| Stata | Python |
|---|---|
| `e(Placebo_p_AS)` | `res["results"][f"table_placebo_{p}"]` &nbsp;(row 0)            |
| `e(Placebo_p_WAS)` | `res["results"][f"table_placebo_{p}"]` &nbsp;(row 1)            |
| `e(Placebo_p_IWAS)` | `res["results"][f"table_placebo_{p}"]` &nbsp;(row 2)            |

Or, via the class:

```python
model.placebo_tables_   # {1: DataFrame, 2: DataFrame, ...}
```

### By-group results

If the program is `bysort`-ed (Python: `by=["..."]`), or `by_fd(K)` / `by_baseline(K)` is
specified, each level `ℓ` (or quantile `k`) gets its own block:

| Stata | Python (functional dict) |
|---|---|
| `e(AS_ℓ)` | `res[f"results_by_{j}"]["table"]` *(AS rows)* |
| `e(WAS_ℓ)` | `res[f"results_by_{j}"]["table"]` *(WAS rows)* |
| `e(IWAS_ℓ)` | `res[f"results_by_{j}"]["table"]` *(IWAS rows)* |
| `e(Placebo_p_AS_ℓ)` | `res[f"results_by_{j}"][f"table_placebo_{p}"]` |

`res["by_levels"]` lists the levels in order, so `by_levels[j-1]` is the value of `ℓ` for
results block `j`.

## Class-API attributes

After calling `.fit()`, the following attributes are populated on the `DIDMultiplegtStat`
instance:

| Attribute | Type | What it is |
|---|---|---|
| `results_` | `dict` | Full functional-API return value — the source of truth. |
| `table_` | `pd.DataFrame` | Main results table for the first (or only) by-group. Columns: `Estimate, SE, LB CI, UB CI, Switchers, Stayers`. |
| `placebo_tables_` | `dict[int, pd.DataFrame] \| None` | Placebo tables keyed by placebo index. |
| `n_obs_` | `int` | Number of observations. |
| `n_clusters_` | `int \| None` | Number of clusters (if `cluster=` was set). |
| `by_levels_` | `list \| None` | Levels of by-group analysis. |
| `first_stage_` | `DIDMultiplegtStat \| None` | Nested fitted model for the first stage of IV-WAS. |
| `is_fitted_` | `bool` | Whether `.fit()` has been called. |

## Notes for `estout` users

The Stata command is compatible with `estout`. In Python, the closest workflow is to
`pd.concat([m1.to_dataframe(), m2.to_dataframe()], keys=["m1", "m2"])` and export to
LaTeX / CSV / Excel from pandas.

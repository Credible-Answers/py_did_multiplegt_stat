# Syntax

## Stata

```stata
[bysort varlist:] did_multiplegt_stat Y G T D [Z] [if] [in] ///
    [, estimator(string)        ///
       exact_match              ///
       as_vs_was                ///
       order(#/####/########)   ///
       cross_fitting(#)         ///
       controls(varlist)        ///
       weights(varname)         ///
       cluster(varlist)         ///
       switchers(string)        ///
       placebo(#)               ///
       on_placebo_sample        ///
       twfe(twfe_suboptions)    ///
       noextrapolation          ///
       trimming(#)              ///
       other_treatments(varlist)///
       by_fd(#)                 ///
       by_baseline(#)           ///
       disaggregate             ///
       graph_off                ///
       bys_graph_off            ///
       bootstrap(#)             ///
       seed(#)]
```

| Positional | Type | Meaning |
|---|---|---|
| `Y` | numeric | Outcome variable |
| `G` | numeric | Identifier of the unit of analysis |
| `T` | numeric | Time period |
| `D` | numeric | Treatment variable |
| `Z` *(optional)* | numeric | Instrumental variable |

## Python — class API

```python
from did_multiplegt_stat import DIDMultiplegtStat

model = DIDMultiplegtStat(
    # ---- main estimator selection ----
    estimator=None,           # "aoss" / "waoss" / "ivwaoss" or list
    estimation_method=None,   # "ra" / "ps" / "dr"
    order=1,                  # int or list of 1, 4, or 8 ints
    exact_match=False,
    noextrapolation=False,
    aoss_vs_waoss=False,

    # ---- testing parallel trends ----
    placebo=0,
    switchers=None,           # None | "up" | "down"
    on_placebo_sample=False,

    # ---- design controls ----
    controls=None,            # list[str]
    weight=None,              # str
    cluster=None,             # str
    other_treatments=None,    # list[str]
    trimming=0,
    cross_fitting=0,

    # ---- heterogeneity ----
    by=None,                  # list[str] (time-invariant)
    by_fd=None,               # int — bins of |ΔD|
    by_baseline=None,         # int — bins of D_{t-1}

    # ---- inference / comparison ----
    bootstrap=0,
    seed=0,
    twfe=False,               # bool or dict with same_sample/full_sample/percentile

    # ---- display ----
    disaggregate=False,

    # ---- cross-validation for polynomial order ----
    cross_validation=None,    # {"algorithm": "kfolds", "tolerance": 0.01,
                              #  "max_k": 5, "kfolds": 5, "seed": 0,
                              #  "same_order_all_logits": False}

    # ---- backend selection (Python-only) ----
    asinstata=False,          # True = Stata-faithful regressions
    iv_method="manual",       # "manual" / "linearmodels" / "econtools"
    model_deltay=None,        # custom sklearn-style regressor
    model_stayer=None,        # custom sklearn-style classifier
)

model.fit(df, Y="...", ID="...", Time="...", D="...", Z=None)
```

## Python — functional API

```python
from did_multiplegt_stat import did_multiplegt_stat

results = did_multiplegt_stat(
    df, Y, ID, Time, D,
    Z=None,
    # ... all of the keyword arguments above
)
```

## Stata-to-Python name mapping

A handful of names differ between Stata and Python. See [Stata parity](parity.md) for the
full table.

| Stata | Python |
|---|---|
| `weights(varname)` | `weight="varname"` |
| `as_vs_was` | `aoss_vs_waoss=True` |
| `cluster(varlist)` | `cluster="..."` *(single var)* |
| `bysort g:` prefix | `by=["g"]` |
| `or(1)` | `order=1` |
| `or(1 4 3 2)` | `order=[1, 4, 3, 2]` |
| `or(1 4 3 2 1 2 3 4)` *(IV)* | `order=[1, 4, 3, 2, 1, 2, 3, 4]` |
| *(none — graph displays inline)* | `model.plot()` |
| `graph_off` | *(no plot is shown unless you call `.plot()`)* |

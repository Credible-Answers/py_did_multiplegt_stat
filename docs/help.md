# Reference

!!! info "About this page"
    This is the canonical reference for `did_multiplegt_stat`, formatted to mirror
    the Stata help file. For the Python API specifically (methods, attributes of
    `DIDMultiplegtStat`), see the [Python API](api/class.md) section.

## Title

**did_multiplegt_stat** — Estimation of heterogeneity-robust difference-in-differences (DID)
estimators, with a binary, discrete, or continuous treatment or instrument, in designs with
stayers, assuming that past treatments do not affect the current outcome.

## Syntax

=== "Python — class"

    ```python
    from did_multiplegt_stat import DIDMultiplegtStat

    model = DIDMultiplegtStat(
        estimator=...,           # str or list: "aoss" / "waoss" / "ivwaoss"
        estimation_method=...,   # "ra" / "ps" / "dr"
        order=1,                 # int, list[1] | list[4] | list[8]
        exact_match=False,
        noextrapolation=False,
        placebo=0,
        switchers=None,          # None | "up" | "down"
        disaggregate=False,
        aoss_vs_waoss=False,
        by=None,                 # list[str]
        by_fd=None,              # int
        by_baseline=None,        # int
        other_treatments=None,   # list[str]
        controls=None,           # list[str]
        weight=None,             # str
        cluster=None,            # str
        cross_fitting=0,
        trimming=0,
        on_placebo_sample=False,
        bootstrap=0,
        seed=0,
        twfe=False,              # bool | {"same_sample": True, "percentile": True, ...}
        cross_validation=None,   # {"algorithm": "kfolds", "max_k": 5, ...}
        asinstata=False,         # True for Stata-faithful regressions
        iv_method="manual",      # "manual" | "linearmodels" | "econtools"
        model_deltay=None,       # custom sklearn-style regressor
        model_stayer=None,       # custom sklearn-style classifier
    )
    model.fit(df, Y="...", ID="...", Time="...", D="...", Z=None)
    ```

=== "Python — functional"

    ```python
    from did_multiplegt_stat import did_multiplegt_stat

    res = did_multiplegt_stat(df, Y, ID, Time, D, Z=None,
                              estimator=..., placebo=3, ...)
    ```

=== "Stata"

    ```stata
    [bysort varlist:] did_multiplegt_stat Y G T D [Z] [if] [in] ///
        [, estimator(string) exact_match as_vs_was ///
           order(#/####/########) cross_fitting(#) ///
           controls(varlist) weights(varname) cluster(varlist) ///
           switchers(string) placebo(#) on_placebo_sample ///
           twfe(twfe_suboptions) noextrapolation trimming(#) ///
           other_treatments(varlist) by_fd(#) by_baseline(#) ///
           disaggregate graph_off bys_graph_off bootstrap(#) seed(#)]
    ```

| Stata positional | Python kwarg | Meaning |
|---|---|---|
| `Y` | `Y=` | Outcome variable name |
| `G` | `ID=` | Unit identifier |
| `T` | `Time=` | Time period |
| `D` | `D=` | Treatment variable |
| `Z` *(optional)* | `Z=` | Instrumental variable |

## Description

### Data and design

The command uses panel data at the `(G, T)` level to estimate heterogeneity-robust DID
estimators, with a binary, discrete, or continuous treatment (or instrument). The command
can be used in designs where there is at least one pair of consecutive time periods between
which the treatment of some units, the **switchers**, changes, while the treatment of some
other units, the **stayers**, does not change.

### Target parameters

The command can estimate the **Average Slope (AS)** and the **Weighted Average Slope (WAS)**
parameters introduced in de Chaisemartin et al. (2025).

- The **AS** is the average, across switchers, of
  $\big(Y_t(D_t) - Y_t(D_{t-1})\big) / (D_t - D_{t-1})$, the effect on their period-$t$
  outcome of moving their period-$t$ treatment from its period-$(t-1)$ to its period-$t$
  value, scaled by the difference between these two values.
- The **WAS** is a weighted average of switchers' slopes
  $\big(Y_t(D_t) - Y_t(D_{t-1})\big) / (D_t - D_{t-1})$, where slopes receive a weight
  proportional to $|D_t - D_{t-1}|$, switchers' absolute treatment change from period
  $(t-1)$ to period $t$.

The variance of the WAS estimator is often smaller than that of the AS estimator,
especially when there are switchers that experience a small treatment change.

### Assumptions

When the data has more than two time periods, the command assumes a **static model**:
units' outcome at period $t$ only depends on their period-$t$ treatment, not on their
lagged treatments. See the `did_multiplegt_dyn` command for estimators allowing for dynamic
effects.

The command also makes a **parallel-trends assumption**: the counterfactual outcome
evolution switchers would have experienced if their treatment had not changed is assumed
to be equal to the outcome evolution of stayers with the same baseline treatment.
Importantly, this parallel-trends assumption is **conditional on the baseline treatment**:
comparing switchers and stayers with different baseline treatments would implicitly amount
to assuming that the treatment's effect is constant over time.

To test the parallel trends assumption underlying the estimators, the command can compute
**placebo estimators** comparing the outcome evolution of switchers and stayers with the
same baseline treatment *before* switchers' treatment changes.

### Estimators, when `exact_match` is specified

With a binary or discrete treatment, if the `exact_match` option is specified, the
estimators computed by the command compare the outcome evolution of switchers and stayers
with the same period-$(t-1)$ treatment. Then, the WAS estimator computed by
`did_multiplegt_stat` is numerically equivalent to the $\text{DID}_M$ estimator proposed by
de Chaisemartin and D'Haultfœuille (2020), and already computed by the `did_multiplegt_old`
command.

`did_multiplegt_stat` uses an **analytic formula** to compute the estimator's variance,
while `did_multiplegt_old` uses the bootstrap. Thus, the run time of `did_multiplegt_stat`
is typically much lower.

The `exact_match` option can only be specified when the treatment is binary or discrete:
with a continuously distributed treatment, one cannot find switchers and stayers with the
exact same period-$(t-1)$ treatment. With a discrete treatment taking a large number of
values, specifying this option may be undesirable — there may only be few switchers that
can be matched to a stayer with the exact same period-$(t-1)$ treatment, thus restricting
the estimation sample.

### Estimators, when `exact_match` is not specified

When the `exact_match` option is not specified, the command computes a **doubly-robust
estimator**, that combines regression adjustment and propensity-score reweighting to
compare switchers and stayers controlling for their period-$(t-1)$ treatment.

- The **regression adjustment** amounts to regressing, for all $t$, $Y_t - Y_{t-1}$ on a
  polynomial in $D_{t-1}$ in the sample of $(t-1)$-to-$t$ stayers, and using the
  regression to predict switchers' $Y_t - Y_{t-1}$.
- **Propensity score reweighting** is based on logistic regressions of an indicator for
  $(t-1)$-to-$t$ switchers on a polynomial in $D_{t-1}$.

### Instrumental-variable case

There may be instances where the parallel-trends assumption fails, but one has at hand an
instrument satisfying a similar parallel-trends assumption. For instance, one may be
interested in estimating the price-elasticity of a good's consumption, but prices respond
to supply and demand shocks, and the counterfactual consumption evolution of units
experiencing and not experiencing a price change may therefore not be the same. On the
other hand, taxes may not respond to supply and demand shocks and may satisfy a parallel-
trends assumption.

In such cases, the command can compute the **IV-WAS estimator** introduced in
de Chaisemartin et al. (2025) using doubly-robust estimators. The IV-WAS estimator is
equal to the WAS estimator of the instrument's reduced-form effect on the outcome
controlling for $D_{t-1}$, divided by the WAS estimator of the instrument's first-stage
effect on the treatment controlling for $D_{t-1}$. See the paper for some explanations as
to why controlling for $D_{t-1}$ is desirable in IV estimation.

## Notes

1. The Stata command is compatible with `estout`. The Python class exposes
   `.to_dataframe()` and `.get_coefficients()`, which integrate with any pandas-aware
   reporting toolkit.
2. The Stata command is byable (`bysort varlist:`). In Python this is the `by=[...]`
   keyword on `DIDMultiplegtStat` and on `did_multiplegt_stat()`.

## Authors

- Clément de Chaisemartin, Economics Department, Sciences Po, France
- Diego Ciccia, Sciences Po, France
- Xavier D'Haultfœuille, CREST-ENSAE, France
- Felix Knau, Sciences Po, France
- Felix Pasquier, CREST-ENSAE, France
- Doulo Sow, Sciences Po, France
- Gonzalo Vazquez-Bare, UCSB, USA

**Python port:** Anzony Quispe.

**Contact:** <chaisemartin.packages@gmail.com>

## References

- de Chaisemartin, C., D'Haultfœuille, X., Pasquier, F., Sow, D., Vazquez-Bare, G.
  (2024) *Difference-in-Differences for Continuous Treatments and Instruments with Stayers.*
- de Chaisemartin, C., D'Haultfœuille, X. (2020) *Two-Way Fixed Effects Estimators with
  Heterogeneous Treatment Effects.*
- de Chaisemartin, C., D'Haultfœuille, X. (2021) *Two-way Fixed Effects and
  Differences-in-Differences Estimators with Several Treatments.*
- Li, S., Linn, J., Muehlegger, E. (2014) *Gasoline Taxes and Consumer Behavior.*

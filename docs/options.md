# Options

All options exposed by the Stata command are available in Python with the same semantics.
Below, each Stata option is followed by its Python equivalent and explanation.

## Main options

### `estimator(string)` — `estimator=...`

Names of the estimator(s) to compute. Allowed values:

- `"aoss"` — AS (Average Slope), aliased as `"aoss"` in Python.
- `"waoss"` — WAS (Weighted Average Slope).
- `"ivwaoss"` — IV-WAS.

You can pass a string or a list:

```python
estimator="waoss"
estimator=["aoss", "waoss"]
```

If multiple estimators are requested in Stata, they must be separated by a single space.

### `exact_match` — `exact_match=True`

The DID estimators compare the outcome evolution of switchers and stayers with the **same
period-$(t-1)$ treatment** (or instrument) value. Only valid for binary or discrete
treatments. With a discrete treatment taking a large number of values, specifying this
option may be undesirable: there may only be few switchers that can be matched to a stayer
with the exact same period-$(t-1)$ treatment, restricting the estimation sample.

### `order(#/####/########)` — `order=int|list`

When `exact_match` is **not** specified, this option specifies the polynomial orders used
in the OLS regressions of $Y_t - Y_{t-1}$ on a polynomial in $D_{t-1}$ and/or in the
logistic regressions of an indicator for $(t-1)$-to-$t$ switchers on a polynomial in
$D_{t-1}$.

The option allows for 1, 4, or 8 arguments (8 only allowed for IV-WAS):

| Arguments | Meaning |
|---|---|
| `order=1` *(default)* | Same order for everything |
| `order=[1, 4, 3, 2]` | `[E(ΔY|D_{t-1}), P(S_t=0|D_{t-1}), P(S_{+t}=1|D_{t-1}), P(S_{-t}=1|D_{t-1})]` |
| `order=[1, 4, 3, 2, 1, 2, 3, 4]` | First 4: first-stage; last 4: reduced-form (IV only) |

### `placebo(#)` — `placebo=N`

The command computes placebo versions of each estimator requested. Actual estimators
compare the $(t-1)$-to-$t$ outcome evolution of $(t-1)$-to-$t$ switchers and stayers with
the same baseline treatment.

- When `N=1`, placebos compare the $(t-2)$-to-$(t-1)$ outcome evolution of $(t-1)$-to-$t$
  switchers and stayers with the same baseline treatment, restricting attention to
  $(t-2)$-to-$(t-1)$ stayers.
- When `N>1`, placebos comparing the outcome evolutions of $(t-1)$-to-$t$ switchers and
  stayers from $t-3$ to $t-2$, from $t-4$ to $t-3$, ..., and from $t-N-1$ to $t-N$ are
  also reported, always restricting attention to stayers between those pairs of periods.

### `as_vs_was` — `aoss_vs_waoss=True`

Shows a test that the AS and WAS are equal. Only valid when both AS and WAS estimation is
requested, i.e. `estimator=["aoss", "waoss"]`.

### `controls(varlist)` — `controls=[...]`

Estimators with control variables rely on a **conditional parallel trends assumption**: the
counterfactual outcome evolution of switchers had their treatment not changed is equal to
the outcome evolution of stayers with the same baseline treatment **and** the same value
of the controls.

When time-varying controls are inputted, the command compares the $(t-1)$-to-$t$ outcome
evolution of switchers and stayers with the same baseline treatment and with the same
controls at period $t-1$.

!!! warning
    Specifying too many control variables may lead to noisy estimators. If placebo
    estimators are small, insignificant, and precisely estimated without controls,
    including controls may not be necessary.

### `weights(varname)` — `weight="..."`

Compute estimators weighted by `varname`.

### `trimming(#)` — `trimming=N`

Integer between 0 and 100.

- If AS or WAS is requested: trim observations for which the estimated
  $P(S_t=0 \mid D_{t-1})$ is lower than `N`%.
- If IV-WAS: trim observations such that $P(SI_t=1 \mid Z_{t-1}, D_{t-1})$ is lower than `N`%.

Only relevant when probabilities are estimated with logit models — **not compatible** with
`exact_match`.

## Heterogeneous treatment effects

### `switchers(string)` — `switchers="up"|"down"`

- `"up"`: estimate the AS / WAS / IV-WAS for **switchers-up** only (units whose treatment
  or instrument increases from $t-1$ to $t$).
- `"down"`: estimate for **switchers-down** only.
- *Default:* all switchers.

### `by_fd(#)` — `by_fd=N`

Assess heterogeneity by the **absolute value of the treatment's (or instrument's) change**.
For example, `by_fd=5` splits switchers into 5 quintiles of $|\Delta D_t|$ (or $|\Delta Z_t|$),
then estimates treatment effects per subsample. If $|\Delta D_t|$ has mass points, the
command splits switchers into groups with as-equal-as-possible sizes.

### `by_baseline(#)` — `by_baseline=N`

Same as `by_fd` but quantiles are built on $D_{t-1}$ (or $Z_{t-1}$).

### `bysort varlist:` — `by=[...]`

In Stata, prefix the command with `bysort varlist:`. In Python, pass `by=["v1", "v2"]`.
**Only time-invariant variables are allowed.**

## Standard-error options

### `bootstrap(#)` — `bootstrap=N`

Compute bootstrap standard errors with `N` replications.

Use case: if the number of switchers or stayers is low, you may want to check the analytic
SEs against bootstrap SEs as a diagnostic. Currently only allowed when IV-WAS is requested
(asymptotic approximations are more likely to fail with weak instruments).

### `seed(#)` — `seed=N`

Set the bootstrap seed for reproducibility.

### `cluster(varlist)` — `cluster="..."`

Cluster standard errors at the level of the supplied variable.

## Advanced options

### `other_treatments(varlist)` — `other_treatments=[...]`

Control for other treatments that may also change over the panel. See de Chaisemartin and
D'Haultfœuille (2021) for further details.

### `noextrapolation` — `noextrapolation=True`

Keep only switchers whose period-$(t-1)$ treatment (or instrument) is between the minimum
and maximum values of stayers' period-$(t-1)$ treatment, enforcing the overlap condition.

### `cross_fitting(#)` — `cross_fitting=K`

Perform $K$-fold cross-fitting for the doubly-robust estimator. With `K=2`, the command
splits the sample into two halves, fits nuisance functions on one and estimates the
parameter on the other, then swaps roles. The final point estimate is a weighted average.

See Section 3.3 of de Chaisemartin et al. (2025).

### `on_placebo_sample` — `on_placebo_sample=True`

Compute the treatment-effect estimator on the subsample where the **first** placebo
estimator is computed. The resulting estimator remains valid if the first treatment lag
affects the outcome.

## TWFE comparison

### `twfe(twfe_suboptions)` — `twfe=True | {dict}`

Compare to a TWFE estimator from regressing $Y_{i,t}$ on $D_{i,t}$ and unit + year fixed
effects. With IV-WAS, a 2SLS-TWFE is used, with $Z_{i,t}$ as the instrument.

The command displays a table with the difference between the requested estimator and the
TWFE estimator, the $p$-value of the test of the difference, and the corresponding CI.
By default, 100 bootstrap replications. Increase with `bootstrap=N`. For replicability use
`seed=N`.

**You must specify either `full_sample` or `same_sample`.**

| Suboption | Description |
|---|---|
| `same_sample` | Estimate the TWFE regression on the same sample as `did_multiplegt_stat`. Use this when the command does not use all time periods (e.g., a period $p$ has no switchers, so the pair $(p-1, p)$ is dropped). |
| `full_sample` | Estimate the TWFE regression on the full sample. |
| `percentile` | Use the percentile bootstrap for $p$-values and CIs (default is normal approximation). |

Python:

```python
twfe={"same_sample": True}
twfe={"full_sample": True, "percentile": True}
```

## Display

### `disaggregate` — `disaggregate=True`

Show estimated AS / WAS / IV-WAS effects for each pair of consecutive time periods, on top
of the aggregated effects. By default, only aggregated effects are shown.

### `graph_off` and `bys_graph_off`

Stata's `did_multiplegt_stat` displays a graph by default. In Python, no graph is shown
unless you explicitly call `model.plot()`, so these flags are not needed.

## Backend selection (Python-only)

These options have no Stata equivalent — they control which Python regression engine is
used for the OLS / logit nuisance estimations.

### `asinstata=True | False`

| Value | Behaviour |
|---|---|
| `False` *(default)* | scikit-learn `LinearRegression` / `LogisticRegression`. Modern, fast. |
| `True` | statsmodels OLS + from-scratch Newton-Raphson logit replicating Stata's `logit, asis` to ~1e-7 relative error. |

### `iv_method="manual" | "linearmodels" | "econtools"`

IV regression backend (used in the bootstrap-based TWFE comparison for IV-WAS):

- `"manual"` *(default)* — two-step OLS (no external dependencies).
- `"linearmodels"` — uses `linearmodels.iv.IV2SLS` with proper IV standard errors.
  Install with `pip install "did-multiplegt-stat[linearmodels]"`.
- `"econtools"` — uses `econtools.metrics.ivreg`. Install with `pip install econtools`.

### `model_deltay`, `model_stayer`

Pluggable custom nuisance models. Must implement sklearn-style `.fit(X, y)` and
`.predict(X)` (regressor) or `.predict_proba(X)` (classifier).

```python
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

model = DIDMultiplegtStat(
    model_deltay=RandomForestRegressor(n_estimators=100),
    model_stayer=RandomForestClassifier(n_estimators=100),
)
```

Each `fit()` call deep-copies the template, so the original object is never mutated.

## Cross-validation for polynomial order

### `cross_validation=dict | None`

If specified, `order` is selected by $K$-fold cross-validation. Pass a dict with any of:

| Key | Default | Meaning |
|---|---|---|
| `algorithm` | `"kfolds"` | Currently only kfolds is implemented. |
| `tolerance` | `0.01` | Stop when the relative MSE improvement falls below this. |
| `max_k` | `5` | Maximum polynomial order to try. |
| `kfolds` | `5` | Number of folds. |
| `seed` | `0` | RNG seed. |
| `same_order_all_logits` | `False` | If `True`, use the same order for all logits. |

```python
model = DIDMultiplegtStat(
    cross_validation={"max_k": 4, "kfolds": 10, "seed": 42},
)
```

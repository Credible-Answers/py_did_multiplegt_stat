# Examples

These examples follow Section 7 of de Chaisemartin et al. (2024). The dataset is an
excerpt of Li et al. (2014) — gasoline taxes, prices, and consumption for 48 US states,
every year from 1966 to 2008.

In Stata you can download the dataset with:

```stata
ssc install did_multiplegt_stat
net get did_multiplegt_stat
use gazoline_did_multiplegt_stat.dta, clear
```

In Python the same `.dta` file is bundled in the package's test fixtures and can be loaded
with `pandas.read_stata`.

## Example 1 — Effect of gasoline taxes on log price

Stata:

```stata
did_multiplegt_stat lngpinc id year tau, or(1) estimator(as was) placebo(3) as_vs_was
```

Python — class API:

```python
import pandas as pd
from did_multiplegt_stat import DIDMultiplegtStat

df = pd.read_stata("gazoline_did_multiplegt_stat.dta")

model = DIDMultiplegtStat(
    estimator=["aoss", "waoss"],
    order=1,
    placebo=3,
    aoss_vs_waoss=True,
)
model.fit(df, Y="lngpinc", ID="id", Time="year", D="tau")
model.summary()
```

Python — functional API:

```python
from did_multiplegt_stat import did_multiplegt_stat, summary_did_multiplegt_stat

res = did_multiplegt_stat(
    df, Y="lngpinc", ID="id", Time="year", D="tau",
    estimator=["aoss", "waoss"], order=1, placebo=3, aoss_vs_waoss=True,
)
summary_did_multiplegt_stat(res)
```

## Example 2 — Effect of gasoline taxes on log consumption

Stata:

```stata
did_multiplegt_stat lngca id year tau, or(1) estimator(as was) placebo(3) as_vs_was
```

Python:

```python
model = DIDMultiplegtStat(
    estimator=["aoss", "waoss"], order=1, placebo=3, aoss_vs_waoss=True,
)
model.fit(df, Y="lngca", ID="id", Time="year", D="tau")
model.summary()
model.plot()
```

## Example 3 — IV estimate of the price elasticity of gasoline consumption

Stata:

```stata
did_multiplegt_stat lngca id year lngpinc tau, or(1) estimator(iv-was) placebo(3)
```

Python:

```python
model = DIDMultiplegtStat(estimator="ivwaoss", order=1, placebo=3)
model.fit(df,
          Y="lngca", ID="id", Time="year",
          D="lngpinc",  # endogenous variable (price)
          Z="tau")      # instrument (tax)
model.summary()

# First-stage results are stored on .first_stage_
model.first_stage_.summary()
```

## Reproducing the vignette tests

The package ships with parity tests that reproduce the six configurations in the
`vignette_asinstata` and `vignette_models` notebooks. Run them with:

```bash
pytest tests/test_vignette_parity.py -v
pytest tests/test_custom_models.py -v
```

## Custom nuisance models

Use any sklearn-style estimator as the nuisance regressor / classifier:

```python
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from did_multiplegt_stat import DIDMultiplegtStat

model = DIDMultiplegtStat(
    estimator=["aoss", "waoss"],
    model_deltay=RandomForestRegressor(n_estimators=200, random_state=42),
    model_stayer=RandomForestClassifier(n_estimators=200, random_state=42),
)
model.fit(df, Y="lngca", ID="id", Time="year", D="tau")
model.summary()
```

## Cluster-robust SEs

```python
model = DIDMultiplegtStat(estimator="waoss", cluster="state")
model.fit(df, Y="lngca", ID="id", Time="year", D="tau")
```

## By-group analysis (heterogeneity by `|ΔD|`)

```python
model = DIDMultiplegtStat(estimator="waoss", by_fd=5)
model.fit(df, Y="lngca", ID="id", Time="year", D="tau")
model.plot()  # one bar per quintile
```

## Bootstrap with the TWFE comparison (IV-WAS)

```python
model = DIDMultiplegtStat(
    estimator="ivwaoss",
    bootstrap=500,
    seed=42,
    twfe={"same_sample": True, "percentile": True},
)
model.fit(df, Y="lngca", ID="id", Time="year", D="lngpinc", Z="tau")
model.summary()
```

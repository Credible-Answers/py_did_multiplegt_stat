# did_multiplegt_stat

[![PyPI version](https://img.shields.io/pypi/v/did-multiplegt-stat.svg)](https://pypi.org/project/did-multiplegt-stat/)
[![Python versions](https://img.shields.io/pypi/pyversions/did-multiplegt-stat.svg)](https://pypi.org/project/did-multiplegt-stat/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Python implementation of the **`did_multiplegt_stat`** Stata package by
de Chaisemartin, D'Haultfœuille, Pasquier, Sow, and Vazquez-Bare (2024) —
heterogeneity-robust difference-in-differences estimators with stayers for
binary, discrete, or continuous treatments (and instruments).

The package estimates the **Average Slope (AS)**, **Weighted Average Slope
(WAS)**, and **IV-WAS** parameters in static designs where parallel trends is
assumed conditional on the baseline treatment.

## Installation

```bash
pip install did-multiplegt-stat
```

Optional extras:

```bash
pip install "did-multiplegt-stat[linearmodels]"   # alternative IV backend
pip install "did-multiplegt-stat[docs]"           # build docs locally
pip install "did-multiplegt-stat[dev]"            # tests + tooling
```

## Quick start: default scikit-learn regressions

By default, `DIDMultiplegtStat` uses scikit-learn's `LinearRegression` for the
outcome-change nuisance function and `LogisticRegression` for the stayer
probability nuisance functions. This is equivalent to setting
`asinstata=False`. The option is written explicitly below so that the backend
used by each example is unambiguous.

### Example 1: AS and WAS of gasoline taxes on log consumption

Original Stata command:

```stata
did_multiplegt_stat lngca id year tau, or(1) estimator(as was) placebo(3) as_vs_was
```

Python with the default scikit-learn OLS and logit regressions:

```python
# Uncomment this line when running the example in a Jupyter notebook:
# %pip install pandas statsmodels scikit-learn did-multiplegt-stat

import pandas as pd
from did_multiplegt_stat import DIDMultiplegtStat

data_url = (
    "https://raw.githubusercontent.com/Credible-Answers/"
    "py_did_multiplegt_stat/main/tests/data/gazoline_did_multiplegt_stat.dta"
)
df = pd.read_stata(data_url)

model = DIDMultiplegtStat(
    estimator=["as", "was"],
    order=1,
    placebo=3,
    as_vs_was=True,
    asinstata=False,  # scikit-learn LinearRegression and LogisticRegression
)
model.fit(df, Y="lngca", ID="id", Time="year", D="tau")
model.summary()
model.plot()
```

### Example 2: AS and WAS with no extrapolation

Original Stata command:

```stata
did_multiplegt_stat lngpinc id year tau, or(1) estimator(as was) estimation_method(dr) placebo(3) noextra as_vs_was
```

Python with the default scikit-learn OLS and logit regressions:

```python
# Uncomment this line when running the example in a Jupyter notebook:
# %pip install pandas statsmodels scikit-learn did-multiplegt-stat

import pandas as pd
from did_multiplegt_stat import DIDMultiplegtStat

data_url = (
    "https://raw.githubusercontent.com/Credible-Answers/"
    "py_did_multiplegt_stat/main/tests/data/gazoline_did_multiplegt_stat.dta"
)
df = pd.read_stata(data_url)

model = DIDMultiplegtStat(
    estimator=["as", "was"],
    order=1,
    placebo=3,
    noextrapolation=True,  # Stata: noextra
    as_vs_was=True,
    asinstata=False,  # scikit-learn LinearRegression and LogisticRegression
)
model.fit(df, Y="lngpinc", ID="id", Time="year", D="tau")
model.summary()
model.plot()
```

## Reproducing Stata's OLS and logit results

Set `asinstata=True` to replace the default scikit-learn nuisance regressions
with the Stata-faithful implementations: statsmodels OLS and the package's
Newton-Raphson logit matching Stata's `logit, asis` behavior. The following
cell runs both examples with the Stata-faithful backend:

```python
# Uncomment this line when running the example in a Jupyter notebook:
# %pip install pandas statsmodels scikit-learn did-multiplegt-stat

import pandas as pd
from did_multiplegt_stat import DIDMultiplegtStat

data_url = (
    "https://raw.githubusercontent.com/Credible-Answers/"
    "py_did_multiplegt_stat/main/tests/data/gazoline_did_multiplegt_stat.dta"
)
df = pd.read_stata(data_url)

# Stata-faithful version of Example 1
stata_model_1 = DIDMultiplegtStat(
    estimator=["as", "was"],
    order=1,
    placebo=3,
    as_vs_was=True,
    asinstata=True,
)
stata_model_1.fit(df, Y="lngca", ID="id", Time="year", D="tau")
stata_model_1.summary()
stata_model_1.plot()

# Stata-faithful version of Example 2
stata_model_2 = DIDMultiplegtStat(
    estimator=["as", "was"],
    order=1,
    placebo=3,
    noextrapolation=True,
    as_vs_was=True,
    asinstata=True,
)
stata_model_2.fit(df, Y="lngpinc", ID="id", Time="year", D="tau")
stata_model_2.summary()
stata_model_2.plot()
```

## Using machine learning to estimate the nuisance functions

`DIDMultiplegtStat` accepts custom scikit-learn-style estimators for its
nuisance functions. The examples below use a random forest regressor for the
outcome-change model and a random forest classifier for the stayer model.
Supplying these models overrides the built-in OLS and logit regressions,
regardless of the value of `asinstata`:

```python
# Uncomment this line when running the example in a Jupyter notebook:
# %pip install pandas statsmodels scikit-learn did-multiplegt-stat

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from did_multiplegt_stat import DIDMultiplegtStat

data_url = (
    "https://raw.githubusercontent.com/Credible-Answers/"
    "py_did_multiplegt_stat/main/tests/data/gazoline_did_multiplegt_stat.dta"
)
df = pd.read_stata(data_url)

# Random-forest version of Example 1
ml_model_1 = DIDMultiplegtStat(
    estimator=["as", "was"],
    order=1,
    placebo=3,
    as_vs_was=True,
    model_deltay=RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    ),
    model_stayer=RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    ),
)
ml_model_1.fit(df, Y="lngca", ID="id", Time="year", D="tau")
ml_model_1.summary()
ml_model_1.plot()

# Random-forest version of Example 2
ml_model_2 = DIDMultiplegtStat(
    estimator=["as", "was"],
    order=1,
    placebo=3,
    noextrapolation=True,
    as_vs_was=True,
    model_deltay=RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    ),
    model_stayer=RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    ),
)
ml_model_2.fit(df, Y="lngpinc", ID="id", Time="year", D="tau")
ml_model_2.summary()
ml_model_2.plot()
```

Any compatible estimators may be supplied: `model_deltay` must implement
`fit` and `predict`, while `model_stayer` must implement `fit` and
`predict_proba`.

## Backends: `asinstata`

Two regression backends are bundled:

| Backend                  | When to use                                              | Activate              |
|--------------------------|----------------------------------------------------------|-----------------------|
| **scikit-learn (default)** | Default behaviour. Faster, modern numerical stack.       | `asinstata=False`     |
| **Stata-faithful**       | Need byte-for-byte parity with the Stata ado-file.       | `asinstata=True`      |

Stata parity uses statsmodels OLS + a from-scratch Newton-Raphson logit that
matches Stata's `logit, asis` defaults; results agree to ~1e-7 relative error.

## Main options

The Python API follows the terminology used in the paper and Stata package:

- `estimator` (`as` / `was` / `iv-was`)
- `order` (scalar, 4-tuple, or 8-tuple for IV)
- `placebo(N)` (multi-period placebos)
- `exact_match`, `noextrapolation`
- `switchers` (`up` / `down`)
- `as_vs_was`
- `by`, `by_fd`, `by_baseline`
- `controls`, `weight`, `cluster`
- `other_treatments`
- `cross_fitting`, `trimming`, `on_placebo_sample`
- `bootstrap` + `seed`
- `twfe` (with `same_sample`, `full_sample`, `percentile`)
- `cross_validation` (k-fold CV for polynomial order)

The package uses doubly robust estimation by default. When `exact_match=True`,
it uses regression adjustment internally; users do not select RA or PS as a
separate estimation method.

See the [full Python documentation](https://credible-answers.github.io/py_did_multiplegt_stat/)
for the help-file style reference.

## Citation

If you use this software in academic work, please cite the underlying paper:

> de Chaisemartin, C., D'Haultfœuille, X., Pasquier, F., Sow, D., Vazquez-Bare, G.
> (2024). *Difference-in-Differences for Continuous Treatments and Instruments with Stayers.*
> arXiv:2201.06898.

A `CITATION.cff` is bundled for tooling integration.

## License

GPL-3.0-or-later — see the [license](https://github.com/Credible-Answers/py_did_multiplegt_stat/blob/main/LICENSE).

## Authors

Originally authored by the team behind the Stata package:

- Clément de Chaisemartin (Sciences Po)
- Diego Ciccia (Sciences Po)
- Xavier D'Haultfœuille (CREST-ENSAE)
- Felix Knau (Sciences Po)
- Felix Pasquier (CREST-ENSAE)
- Doulo Sow (Sciences Po)
- Gonzalo Vazquez-Bare (UCSB)

Python port: Anzony Quispe.

Contact: <chaisemartin.packages@gmail.com>

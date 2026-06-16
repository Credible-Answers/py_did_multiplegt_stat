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

## Quick start

```python
import pandas as pd
from did_multiplegt_stat import DIDMultiplegtStat

df = pd.read_stata("gazoline_did_multiplegt_stat.dta")

# AS + WAS with 3 placebo periods (mirrors Stata example 2)
model = DIDMultiplegtStat(
    estimator=["aoss", "waoss"],
    order=1,
    placebo=3,
    aoss_vs_waoss=True,
)
model.fit(df, Y="lngca", ID="id", Time="year", D="tau")
model.summary()
model.plot()
```

Functional API (one-shot, mirrors the Stata command):

```python
from did_multiplegt_stat import did_multiplegt_stat, summary_did_multiplegt_stat

res = did_multiplegt_stat(df, Y="lngca", ID="id", Time="year", D="tau",
                          estimator=["aoss", "waoss"], placebo=3,
                          aoss_vs_waoss=True)
summary_did_multiplegt_stat(res)
```

## Backends: `asinstata`

Two regression backends are bundled:

| Backend                  | When to use                                              | Activate              |
|--------------------------|----------------------------------------------------------|-----------------------|
| **scikit-learn (default)** | Default behaviour. Faster, modern numerical stack.       | `asinstata=False`     |
| **Stata-faithful**       | Need byte-for-byte parity with the Stata ado-file.       | `asinstata=True`      |
| **Custom sklearn-style** | Want a `RandomForest`, `LassoCV`, etc. as the nuisance.  | `model_deltay=...`, `model_stayer=...` |

Stata parity uses statsmodels OLS + a from-scratch Newton-Raphson logit that
matches Stata's `logit, asis` defaults; results agree to ~1e-7 relative error.

## What's supported

All Stata options are exposed, including:

- `estimator` (`aoss` / `waoss` / `ivwaoss`)
- `estimation_method` (`ra` / `ps` / `dr`)
- `order` (scalar, 4-tuple, or 8-tuple for IV)
- `placebo(N)` (multi-period placebos)
- `exact_match`, `noextrapolation`
- `switchers` (`up` / `down`)
- `aoss_vs_waoss`
- `by`, `by_fd`, `by_baseline`
- `controls`, `weight`, `cluster`
- `other_treatments`
- `cross_fitting`, `trimming`, `on_placebo_sample`
- `bootstrap` + `seed`
- `twfe` (with `same_sample`, `full_sample`, `percentile`)
- `cross_validation` (k-fold CV for polynomial order)

See the [full documentation](https://chaisemartin.github.io/did_multiplegt_stat/)
for the help-file style reference.

## Citation

If you use this software in academic work, please cite the underlying paper:

> de Chaisemartin, C., D'Haultfœuille, X., Pasquier, F., Sow, D., Vazquez-Bare, G.
> (2024). *Difference-in-Differences for Continuous Treatments and Instruments with Stayers.*
> arXiv:2201.06898.

A `CITATION.cff` is bundled for tooling integration.

## License

GPL-3.0-or-later — see [LICENSE](LICENSE).

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

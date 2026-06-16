# did_multiplegt_stat

Python implementation of the Stata `did_multiplegt_stat` package — heterogeneity-robust
difference-in-differences estimators with stayers, for binary, discrete, or continuous
treatments and instruments.

The package estimates:

- **AS** (Average Slope)
- **WAS** (Weighted Average Slope)
- **IV-WAS** (instrumental variant of WAS)

introduced in **de Chaisemartin, D'Haultfœuille, Pasquier, Sow, Vazquez-Bare (2024)**,
*Difference-in-Differences Estimators for Treatments Continuously Distributed at Every Period*
([arXiv:2201.06898](https://arxiv.org/abs/2201.06898)).

---

## Install

```bash
pip install did-multiplegt-stat
```

Optional extras:

```bash
pip install "did-multiplegt-stat[linearmodels]"   # alternative IV backend
pip install "did-multiplegt-stat[docs]"           # build docs locally
pip install "did-multiplegt-stat[dev]"            # tests + tooling
```

## 60-second example

```python
import pandas as pd
from did_multiplegt_stat import DIDMultiplegtStat

# Excerpt of Li et al. (2014) — gas taxes, prices, consumption for 48 US states.
df = pd.read_stata("gazoline_did_multiplegt_stat.dta")

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

## Two APIs

| | Functional | Class |
| - | - | - |
| Import | `from did_multiplegt_stat import did_multiplegt_stat` | `from did_multiplegt_stat import DIDMultiplegtStat` |
| Style | Mirrors the Stata command, one-shot | scikit-learn `fit/predict/summary` |
| Returns | `dict` with `args`, `results`, `first_stage`, etc. | Fitted estimator with `.table_`, `.results_`, `.first_stage_` |

See the **[Python API](api/class.md)** section for full method docs.

## Two backends

| Backend | When to use | Activate |
|---|---|---|
| scikit-learn *(default)* | Default — modern numerical stack, fast | `asinstata=False` |
| Stata-faithful | Need byte-for-byte parity with the Stata ado-file | `asinstata=True` |
| Custom sklearn-style | RandomForest / LassoCV / etc. as nuisance | `model_deltay=...`, `model_stayer=...` |

Stata parity uses statsmodels OLS and a from-scratch Newton-Raphson logit matching
Stata's `logit, asis` defaults. Numerical agreement is typically ~1e-7.

## Where to next

- **[Help file](help.md)** — the canonical reference, modelled on the Stata help.
- **[Syntax](syntax.md)** — quick reference for the command surface.
- **[Options](options.md)** — every option and what it does.
- **[Examples](examples.md)** — the three worked examples from the paper.
- **[Stata parity](parity.md)** — how Stata options map to Python kwargs.
- **[Python API](api/class.md)** — auto-generated method/attribute docs.

## Citation

```bibtex
@article{dechaisemartin2024did,
  title  = {Difference-in-Differences Estimators for Treatments Continuously Distributed at Every Period},
  author = {de Chaisemartin, Cl{\'e}ment and D'Haultf{\oe}uille, Xavier and
            Pasquier, Felix and Sow, Doulo and Vazquez-Bare, Gonzalo},
  year   = {2024},
  eprint = {2201.06898},
  archivePrefix = {arXiv}
}
```

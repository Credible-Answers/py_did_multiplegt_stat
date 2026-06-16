"""
did_multiplegt_stat - Heterogeneity-robust difference-in-differences estimators.

Python implementation of de Chaisemartin, D'Haultfœuille, Pasquier, Sow,
Vazquez-Bare (2024) "Difference-in-Differences for Continuous Treatments and
Instruments with Stayers" - a faithful port of the Stata ``did_multiplegt_stat``
ado-file.

Two interchangeable APIs are exposed:

* **Functional** -- :func:`did_multiplegt_stat`, mirroring the Stata command.
* **Class** -- :class:`DIDMultiplegtStat`, scikit-learn style.

By default, OLS / logit nuisance estimations use scikit-learn. Pass
``asinstata=True`` to use the Stata-faithful regressions (statsmodels OLS +
custom Newton-Raphson logit) that reproduce the Stata ado's numerical output
to ~1e-7 relative precision.

Quick example::

    import pandas as pd
    from did_multiplegt_stat import DIDMultiplegtStat

    df = pd.read_stata("gazoline_did_multiplegt_stat.dta")
    model = DIDMultiplegtStat(estimator=["aoss", "waoss"], placebo=3,
                              aoss_vs_waoss=True)
    model.fit(df, Y="lngca", ID="id", Time="year", D="tau")
    model.summary()
    model.plot()
"""

from __future__ import annotations

from .core import (
    did_multiplegt_stat,
    did_multiplegt_stat_main,
    did_multiplegt_stat_pairwise,
    did_multiplegt_stat_quantiles,
    summary_did_multiplegt_stat,
    print_did_multiplegt_stat,
    polynomials_generator,
    cross_validation_select,
    run_iv_regression,
)
from .estimator import DIDMultiplegtStat
from .plotting import plot_event_study, plot_by_groups, plot_comparison

__version__ = "0.1.0"

__all__ = [
    # Class API
    "DIDMultiplegtStat",
    # Functional API (matches Stata command surface)
    "did_multiplegt_stat",
    "summary_did_multiplegt_stat",
    "print_did_multiplegt_stat",
    # Internals exposed for advanced users / parity tests
    "did_multiplegt_stat_main",
    "did_multiplegt_stat_pairwise",
    "did_multiplegt_stat_quantiles",
    "polynomials_generator",
    "cross_validation_select",
    "run_iv_regression",
    # Plotting helpers
    "plot_event_study",
    "plot_by_groups",
    "plot_comparison",
]

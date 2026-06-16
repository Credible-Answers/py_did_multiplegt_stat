"""Smoke tests: importing the package and running a minimal fit doesn't error."""
from __future__ import annotations

import pandas as pd
import pytest


def test_package_imports():
    import did_multiplegt_stat as pkg

    assert pkg.__version__ == "0.1.0"
    # Public surface must be intact.
    for name in [
        "DIDMultiplegtStat",
        "did_multiplegt_stat",
        "summary_did_multiplegt_stat",
        "print_did_multiplegt_stat",
        "plot_event_study",
    ]:
        assert hasattr(pkg, name), f"missing public symbol: {name}"


def test_minimal_fit_class_api(gazoline: pd.DataFrame):
    from did_multiplegt_stat import DIDMultiplegtStat

    model = DIDMultiplegtStat(estimator=["aoss", "waoss"], order=1)
    model.fit(gazoline, Y="lngca", ID="id", Time="year", D="tau")

    assert model.is_fitted_
    assert model.table_ is not None
    assert "Estimate" in model.table_.columns


def test_minimal_fit_functional_api(gazoline: pd.DataFrame):
    from did_multiplegt_stat import did_multiplegt_stat

    res = did_multiplegt_stat(
        gazoline, Y="lngca", ID="id", Time="year", D="tau",
        estimator=["aoss", "waoss"], order=1,
    )
    assert res["_class"] == "did_multiplegt_stat"
    assert "results" in res
    assert "table" in res["results"]


@pytest.mark.parametrize("asinstata", [True, False])
def test_both_backends_run(gazoline: pd.DataFrame, asinstata: bool):
    from did_multiplegt_stat import did_multiplegt_stat

    res = did_multiplegt_stat(
        gazoline, Y="lngca", ID="id", Time="year", D="tau",
        estimator="waoss", order=1, asinstata=asinstata,
    )
    table = res["results"]["table"]
    # WAS-only request still allocates a 3-block table; locate WAS by index name.
    was_rows = [r for r in table.index if str(r).upper() == "WAS"]
    assert was_rows, f"WAS aggregate row not found in {table.index.tolist()[:5]}"
    # The WAS aggregate row should have a finite point estimate.
    assert pd.notna(table.loc[was_rows[0], "Estimate"]), "WAS estimate is NaN"

"""Port of `_run_test_custom.py`: custom sklearn-style models as nuisances."""
from __future__ import annotations

import pandas as pd
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _extract_main(result: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    tbl = result.get("results", {}).get("table")
    if isinstance(tbl, pd.DataFrame):
        for idx in tbl.index:
            out[idx] = tbl.iloc[tbl.index.get_loc(idx), 0]
    return out


def test_custom_random_forest(gazoline: pd.DataFrame):
    """RandomForest as nuisance: must produce finite estimates for AOSS+WAOSS."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    from did_multiplegt_stat import did_multiplegt_stat

    res = did_multiplegt_stat(
        gazoline, Y="lngca", ID="id", Time="year", D="tau",
        order=1, controls=["lngpinc"],
        model_deltay=RandomForestRegressor(n_estimators=20, random_state=42),
        model_stayer=RandomForestClassifier(n_estimators=20, random_state=42),
    )
    est = _extract_main(res)
    assert any(pd.notna(v) for v in est.values()), "All estimates NaN with RF backend"


def test_custom_lasso(gazoline: pd.DataFrame):
    """LassoCV as the OLS nuisance (logit still uses default sklearn)."""
    from sklearn.linear_model import LassoCV

    from did_multiplegt_stat import did_multiplegt_stat

    res = did_multiplegt_stat(
        gazoline, Y="lngca", ID="id", Time="year", D="tau",
        order=1, controls=["lngpinc"],
        model_deltay=LassoCV(cv=3),
    )
    est = _extract_main(res)
    assert any(pd.notna(v) for v in est.values()), "All estimates NaN with LassoCV"


def test_default_sklearn(gazoline: pd.DataFrame):
    """Default backend (asinstata=False) -- no custom model overrides."""
    from did_multiplegt_stat import did_multiplegt_stat

    res = did_multiplegt_stat(
        gazoline, Y="lngca", ID="id", Time="year", D="tau",
        order=1, controls=["lngpinc"],
    )
    est = _extract_main(res)
    assert any(pd.notna(v) for v in est.values())


def test_asinstata_backend(gazoline: pd.DataFrame):
    """Stata-faithful backend with asinstata=True -- must finish without error."""
    from did_multiplegt_stat import did_multiplegt_stat

    res = did_multiplegt_stat(
        gazoline, Y="lngca", ID="id", Time="year", D="tau",
        order=1, controls=["lngpinc"], asinstata=True,
    )
    est = _extract_main(res)
    assert any(pd.notna(v) for v in est.values())


def test_class_api_with_custom_models(gazoline: pd.DataFrame):
    """The DIDMultiplegtStat class accepts custom-model kwargs."""
    from sklearn.linear_model import LassoCV

    from did_multiplegt_stat import DIDMultiplegtStat

    model = DIDMultiplegtStat(
        estimator=["aoss", "waoss"], order=1, controls=["lngpinc"],
        model_deltay=LassoCV(cv=3),
    )
    model.fit(gazoline, Y="lngca", ID="id", Time="year", D="tau")
    assert model.is_fitted_
    df = model.to_dataframe()
    assert not df.empty

"""Coverage of the DIDMultiplegtStat scikit-learn style class API."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest


@pytest.fixture
def fitted_model(gazoline: pd.DataFrame):
    from did_multiplegt_stat import DIDMultiplegtStat

    model = DIDMultiplegtStat(estimator=["aoss", "waoss"], order=1, placebo=1)
    model.fit(gazoline, Y="lngca", ID="id", Time="year", D="tau")
    return model


def test_repr_before_fit():
    from did_multiplegt_stat import DIDMultiplegtStat

    model = DIDMultiplegtStat()
    assert "not fitted" in repr(model)


def test_repr_after_fit(fitted_model):
    assert "fitted" in repr(fitted_model)


def test_summary_runs(fitted_model, capsys):
    fitted_model.summary()
    out = capsys.readouterr().out
    assert "Estimation of AOSS" in out or "Estimation of WAOSS" in out


def test_to_dataframe(fitted_model):
    df = fitted_model.to_dataframe()
    assert not df.empty
    for col in ["Estimate", "SE", "LB CI", "UB CI", "Switchers", "Stayers"]:
        assert col in df.columns


def test_get_coefficients(fitted_model):
    coeffs = fitted_model.get_coefficients(estimator="aoss")
    assert isinstance(coeffs, pd.Series)
    assert len(coeffs) >= 1


def test_get_confidence_intervals(fitted_model):
    ci = fitted_model.get_confidence_intervals(estimator="waoss")
    assert "LB CI" in ci.columns
    assert "UB CI" in ci.columns


def test_get_set_params():
    from did_multiplegt_stat import DIDMultiplegtStat

    model = DIDMultiplegtStat(estimator="waoss", order=2, placebo=2)
    params = model.get_params()
    assert params["estimator"] == "waoss"
    assert params["order"] == 2
    assert params["placebo"] == 2

    model.set_params(order=3)
    assert model.order == 3


def test_set_params_rejects_unknown():
    from did_multiplegt_stat import DIDMultiplegtStat

    model = DIDMultiplegtStat()
    with pytest.raises(ValueError, match="Invalid parameter"):
        model.set_params(not_a_param=1)


def test_plot_returns_figure(fitted_model):
    fig = fitted_model.plot()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_before_fit_raises():
    from did_multiplegt_stat import DIDMultiplegtStat

    model = DIDMultiplegtStat()
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.plot()


def test_invalid_estimator_combo(gazoline: pd.DataFrame):
    from did_multiplegt_stat import DIDMultiplegtStat

    model = DIDMultiplegtStat(estimator=["aoss", "ivwaoss"])
    with pytest.raises(ValueError, match="Cannot combine"):
        model.fit(gazoline, Y="lngca", ID="id", Time="year", D="tau", Z="lngpinc")


def test_ivwaoss_without_z_raises(gazoline: pd.DataFrame):
    from did_multiplegt_stat import DIDMultiplegtStat

    model = DIDMultiplegtStat(estimator="ivwaoss")
    with pytest.raises(ValueError, match="IV variable Z"):
        model.fit(gazoline, Y="lngca", ID="id", Time="year", D="tau")

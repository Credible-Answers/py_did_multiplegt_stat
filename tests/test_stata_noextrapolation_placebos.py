"""Exact Stata parity for no-extrapolation with multiple placebos."""
from __future__ import annotations

import numpy as np
import pandas as pd


def test_stata_noextrapolation_three_placebos(gazoline: pd.DataFrame):
    """Match the ado-file output for the documented lngpinc example.

    Stata command::

        did_multiplegt_stat lngpinc id year tau, or(1) estimator(as was) \
            estimation_method(dr) placebo(3) noextra as_vs_was
    """
    from did_multiplegt_stat import DIDMultiplegtStat

    model = DIDMultiplegtStat(
        estimator=["as", "was"],
        order=1,
        placebo=3,
        as_vs_was=True,
        noextrapolation=True,
        asinstata=True,
    ).fit(gazoline, Y="lngpinc", ID="id", Time="year", D="tau")

    assert model.n_obs_ == 1603

    expected_main = {
        "AS": (0.0006422, 0.0024195, 355, 1248),
        "WAS": (0.0051255, 0.0009851, 355, 1248),
    }
    assert model.table_ is not None
    for row, (estimate, se, switchers, stayers) in expected_main.items():
        actual = model.table_.loc[row]
        np.testing.assert_allclose(actual["Estimate"], estimate, atol=5e-7, rtol=0)
        np.testing.assert_allclose(actual["SE"], se, atol=5e-7, rtol=0)
        assert actual["Switchers"] == switchers
        assert actual["Stayers"] == stayers

    expected_placebos = {
        1: {
            "as": (0.0003436, 0.0048656, 170, 881),
            "was": (0.0020187, 0.0013700, 170, 881),
        },
        2: {
            "as": (-0.0011831, 0.0064958, 189, 759),
            "was": (-0.0014137, 0.0011142, 189, 759),
        },
        3: {
            "as": (0.0014382, 0.0028566, 224, 853),
            "was": (0.0014295, 0.0008997, 224, 853),
        },
    }
    assert model.placebo_tables_ is not None
    for index, estimator_values in expected_placebos.items():
        table = model.placebo_tables_[index]
        for estimator, (estimate, se, switchers, stayers) in estimator_values.items():
            row = f"Placebo_{index}" if estimator == "as" else f"Placebo_{index}_WAS"
            actual = table.loc[row]
            np.testing.assert_allclose(actual["Estimate"], estimate, atol=5e-7, rtol=0)
            np.testing.assert_allclose(actual["SE"], se, atol=5e-7, rtol=0)
            assert actual["Switchers"] == switchers
            assert actual["Stayers"] == stayers

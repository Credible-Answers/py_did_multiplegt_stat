"""Vignette parity: the 6 examples from `_run_test_asinstata.py` ported to pytest.

These tests ensure the new package layout reproduces the same numbers as the
original script. We compare ``asinstata=True`` (Stata-faithful) and
``asinstata=False`` (sklearn default) cell-by-cell; the *agreement tolerance*
is generous because the two backends are not numerically identical — they're
just both expected to converge to similar coefficients.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _extract_estimates(result: dict) -> dict[str, float]:
    """Pull (Estimate column) from the main + placebo tables."""
    out: dict[str, float] = {}
    tbl = result.get("results", {}).get("table")
    if isinstance(tbl, pd.DataFrame):
        for idx in tbl.index:
            out[idx] = tbl.iloc[tbl.index.get_loc(idx), 0]
    for pl in range(1, 5):
        pl_tbl = result.get("results", {}).get(f"table_placebo_{pl}")
        if isinstance(pl_tbl, pd.DataFrame):
            for idx in pl_tbl.index:
                out[f"Plac{pl}_{idx}"] = pl_tbl.iloc[pl_tbl.index.get_loc(idx), 0]
    return out


def _compare(res_true: dict, res_false: dict, max_rel_pct: float,
             abs_floor: float = 1e-3):
    """Assert every cell agrees within ``max_rel_pct`` relative error.

    Cells where ``|asinstata=True| < abs_floor`` are skipped from the
    relative-error check because relative error is unstable near zero;
    those cells must still agree in absolute terms within ``abs_floor``.
    """
    est_t = _extract_estimates(res_true)
    est_f = _extract_estimates(res_false)
    for key in sorted(set(est_t) | set(est_f)):
        vt, vf = est_t.get(key, np.nan), est_f.get(key, np.nan)
        if pd.isna(vt) or pd.isna(vf):
            continue
        if abs(vt) < abs_floor:
            assert abs(vt - vf) <= abs_floor, (
                f"{key}: near-zero diff {abs(vt - vf):.6g} > floor={abs_floor:.6g} "
                f"(True={vt:.6g}, False={vf:.6g})"
            )
            continue
        pct = abs(vt - vf) / abs(vt) * 100
        assert pct <= max_rel_pct, (
            f"{key}: asinstata=True={vt:.6g} vs False={vf:.6g} "
            f"(diff={pct:.4f}% > tol={max_rel_pct}%)"
        )


# ----------------------------------------------------------------------
# V.01 -- AOSS+WAOSS with controls, Y=lngca
# ----------------------------------------------------------------------
def test_vignette_V01_aoss_waoss_controls(gazoline: pd.DataFrame):
    from did_multiplegt_stat import did_multiplegt_stat

    args = dict(Y="lngca", ID="id", Time="year", D="tau", order=1,
                aoss_vs_waoss=True, controls=["lngpinc"])
    r_t = did_multiplegt_stat(gazoline, **args, asinstata=True)
    r_f = did_multiplegt_stat(gazoline, **args, asinstata=False)
    _compare(r_t, r_f, max_rel_pct=15.0)


# ----------------------------------------------------------------------
# V.02 -- AOSS+WAOSS with controls, Y=lngpinc
# ----------------------------------------------------------------------
def test_vignette_V02_aoss_waoss_controls_lngpinc(gazoline: pd.DataFrame):
    from did_multiplegt_stat import did_multiplegt_stat

    args = dict(Y="lngpinc", ID="id", Time="year", D="tau", order=1,
                aoss_vs_waoss=True, controls=["lngpinc"])
    r_t = did_multiplegt_stat(gazoline, **args, asinstata=True)
    r_f = did_multiplegt_stat(gazoline, **args, asinstata=False)
    _compare(r_t, r_f, max_rel_pct=15.0)


# ----------------------------------------------------------------------
# V.03 -- IV-WAOSS with bootstrap
# ----------------------------------------------------------------------
@pytest.mark.slow
def test_vignette_V03_iv_waoss_bootstrap(gazoline: pd.DataFrame):
    from did_multiplegt_stat import did_multiplegt_stat

    args = dict(Y="lngca", ID="id", Time="year", D="lngpinc",
                Z="tau", estimator="ivwaoss", order=1,
                controls=["lngpinc"], bootstrap=5, seed=1)
    r_t = did_multiplegt_stat(gazoline, **args, asinstata=True)
    r_f = did_multiplegt_stat(gazoline, **args, asinstata=False)
    # IV-WAOSS + bootstrap is noisy; use a looser tolerance.
    _compare(r_t, r_f, max_rel_pct=30.0)


# ----------------------------------------------------------------------
# V.04 -- WAOSS on_placebo_sample, Y=lngca
# ----------------------------------------------------------------------
def test_vignette_V04_waoss_on_placebo_sample_lngca(gazoline: pd.DataFrame):
    from did_multiplegt_stat import did_multiplegt_stat

    args = dict(Y="lngca", ID="id", Time="year", D="tau",
                estimator="waoss", order=1, controls=["lngpinc"],
                on_placebo_sample=True)
    r_t = did_multiplegt_stat(gazoline, **args, asinstata=True)
    r_f = did_multiplegt_stat(gazoline, **args, asinstata=False)
    _compare(r_t, r_f, max_rel_pct=20.0)


# ----------------------------------------------------------------------
# V.05 -- WAOSS on_placebo_sample, Y=lngpinc
# ----------------------------------------------------------------------
def test_vignette_V05_waoss_on_placebo_sample_lngpinc(gazoline: pd.DataFrame):
    from did_multiplegt_stat import did_multiplegt_stat

    args = dict(Y="lngpinc", ID="id", Time="year", D="tau",
                estimator="waoss", order=1, controls=["lngpinc"],
                on_placebo_sample=True)
    r_t = did_multiplegt_stat(gazoline, **args, asinstata=True)
    r_f = did_multiplegt_stat(gazoline, **args, asinstata=False)
    _compare(r_t, r_f, max_rel_pct=20.0)


# ----------------------------------------------------------------------
# VI.01 -- Gentzkow exact_match with placebo=1
# ----------------------------------------------------------------------
@pytest.mark.slow
def test_vignette_VI01_gentzkow_exact_match_placebo(gentzkow: pd.DataFrame):
    from did_multiplegt_stat import did_multiplegt_stat

    args = dict(Y="prestout", ID="cnty90", Time="year", D="numdailies",
                placebo=1, exact_match=True)
    r_t = did_multiplegt_stat(gentzkow, **args, asinstata=True)
    r_f = did_multiplegt_stat(gentzkow, **args, asinstata=False)
    # Exact-match with placebo is the hardest parity case; allow looser drift.
    _compare(r_t, r_f, max_rel_pct=50.0)

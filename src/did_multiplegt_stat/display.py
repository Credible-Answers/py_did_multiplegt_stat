"""
Display and formatting functions for did_multiplegt_stat output.

This module provides Stata-style formatting for results display,
mimicking the output format of the original did_multiplegt_stat.ado.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List
import numpy as np
import pandas as pd


def _fmt_float7(x) -> str:
    """Format float to 7 decimal places."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{float(x):.7f}"


def _fmt_int0(x) -> str:
    """Format integer with comma separators."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{float(x):,.0f}"


def _fmt_float5(x) -> str:
    """Format float to 5 decimal places."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{float(x):.5f}"


def mat_print(mat, name: Optional[str] = None) -> None:
    """
    Pretty-print a matrix with floats (7 chars) and integers.

    Parameters
    ----------
    mat : np.ndarray or pd.DataFrame
        Matrix to print.
    name : str, optional
        Name to display above matrix.
    """
    if isinstance(mat, np.ndarray):
        df = pd.DataFrame(mat)
    else:
        df = mat.copy() if isinstance(mat, pd.DataFrame) else pd.DataFrame(mat)
    if df.shape[0] == 0:
        print("(empty)")
        return
    dis = df.copy().astype(object)
    for j in range(dis.shape[1]):
        col = pd.to_numeric(dis.iloc[:, j], errors="coerce")
        if j <= 3:
            dis.iloc[:, j] = col.map(_fmt_float7)
        else:
            dis.iloc[:, j] = col.map(_fmt_int0)
    with pd.option_context("display.max_rows", 200, "display.max_columns", 200, "display.width", 200):
        print(dis)


def tab_print(mat) -> None:
    """
    Pretty-print a table with all floats (5 chars).

    Parameters
    ----------
    mat : np.ndarray or pd.DataFrame
        Table to print.
    """
    if isinstance(mat, np.ndarray):
        df = pd.DataFrame(mat)
    else:
        df = mat.copy() if isinstance(mat, pd.DataFrame) else pd.DataFrame(mat)
    if df.shape[0] == 0:
        print("(empty)")
        return
    dis = df.copy().astype(object)
    for j in range(dis.shape[1]):
        col = pd.to_numeric(dis.iloc[:, j], errors="coerce")
        dis.iloc[:, j] = col.map(_fmt_float5)
    with pd.option_context("display.max_rows", 200, "display.max_columns", 200, "display.width", 200):
        print(dis)


def strdisplay(label: str, value) -> None:
    """
    Print a label=value pair with fixed-width formatting.

    Parameters
    ----------
    label : str
        The label (left side).
    value : float or str
        The value (right side).
    """
    ltot = 16
    label_out = (label + " " * max(0, ltot - len(label)))[:ltot]
    v = f"{float(value):.0f}" if not isinstance(value, str) else value
    value_out = (" " * max(0, ltot - len(str(v))) + str(v))[-ltot:]
    print(f"{label_out} = {value_out}")


def print_header(
    N: int,
    estimation_method: str,
    estimator_list: List[str],
    order: Optional[int] = None,
    exact_match: bool = False,
    noextrapolation: bool = False,
    controls: Optional[List[str]] = None,
    cross_fitting: int = 0,
    trimming: float = 0,
    n_clusters: Optional[int] = None,
    cluster: Optional[str] = None,
    by_level: Optional[str] = None,
) -> None:
    """
    Print the summary statistics header.

    Parameters
    ----------
    N : int
        Number of observations.
    estimation_method : str
        Estimation method ('ra', 'ps', 'dr').
    estimator_list : list of str
        List of estimators used.
    order : int, optional
        Polynomial order.
    exact_match : bool
        Whether exact matching was used.
    noextrapolation : bool
        Whether extrapolation was restricted.
    controls : list of str, optional
        Control variables.
    cross_fitting : int
        Number of cross-fitting folds.
    trimming : float
        Trimming threshold.
    n_clusters : int, optional
        Number of clusters.
    cluster : str, optional
        Cluster variable name.
    by_level : str, optional
        By-group level label.
    """
    if by_level is not None and by_level != "_no_by":
        print(f"\n{'#' * 70}")
        print(f" By level: {by_level}")

    print(f"\n{'-' * 35}")

    strdisplay("N", N)

    methods = {"ra": "Reg. Adjustment", "dr": "Doubly Robust", "ps": "Propensity Score"}
    for m in ("waoss", "ivwaoss"):
        if m in estimator_list:
            strdisplay(f"{m.upper()} Method", methods.get(estimation_method, estimation_method))

    if not exact_match and order is not None:
        strdisplay("Polynomial Order", order)

    if exact_match:
        strdisplay("Common Support", "Exact Matching")
    if noextrapolation:
        strdisplay("Common Support", "No Extrapolation")
    if controls:
        strdisplay("Controls", ", ".join(controls))
    if cross_fitting > 0:
        strdisplay("Cross-fitting", cross_fitting)
    if trimming > 0:
        strdisplay("Trimming", trimming)

    print(f"{'-' * 35}")

    if n_clusters is not None:
        print(f"(Std. errors adjusted for {n_clusters} clusters in {cluster})")


def print_estimator_section(
    estimator: str,
    table: pd.DataFrame,
    estims_map: Dict[str, int],
    pairs: int,
    disaggregate: bool = False,
) -> None:
    """
    Print a section for a specific estimator (AOSS/WAOSS/IV-WAOSS).

    Parameters
    ----------
    estimator : str
        Estimator type ('aoss', 'waoss', 'ivwaoss').
    table : pd.DataFrame
        Results table.
    estims_map : dict
        Mapping from estimator to row index.
    pairs : int
        Number of time period pairs.
    disaggregate : bool
        Whether to show disaggregated results.
    """
    print(f"\n{'-' * 70}")
    print(f"{' ' * 20}Estimation of {estimator.upper()}(s)")
    print(f"{'-' * 70}")

    if isinstance(table, pd.DataFrame):
        l_bound = estims_map[estimator] * pairs
        u_bound = l_bound + (pairs if disaggregate else 1)
        mat_sel = table.iloc[l_bound:u_bound]
        mat_print(mat_sel)


def print_placebo_section(
    estimator: str,
    placebo_tables: Dict[int, pd.DataFrame],
    estims_map: Dict[str, int],
    placebo_n: int,
) -> None:
    """
    Print placebo results for an estimator.

    Parameters
    ----------
    estimator : str
        Estimator type.
    placebo_tables : dict
        Dictionary mapping placebo index to table.
    estims_map : dict
        Mapping from estimator to row index.
    placebo_n : int
        Number of placebos.
    """
    estim_idx = estims_map[estimator]
    pl_rows = []

    for pl_idx in range(1, placebo_n + 1):
        table_p = placebo_tables.get(pl_idx)
        if isinstance(table_p, pd.DataFrame) and estim_idx < len(table_p):
            row = table_p.iloc[[estim_idx]].copy()
            suffix = "" if estimator == "aoss" else f"_{estimator}"
            row.index = [f"Placebo_{pl_idx}{suffix}"]
            # Skip if not computed
            if not (np.isnan(row.iloc[0]["Estimate"]) and row.iloc[0]["Switchers"] == 0):
                pl_rows.append(row)

    if pl_rows:
        pl_combined = pd.concat(pl_rows)
        print(f"\n{'-' * 70}")
        print(f"{' ' * 15}Estimation of {estimator.upper()}(s) - Placebo(s)")
        print(f"{'-' * 70}")
        mat_print(pl_combined)


def print_aoss_vs_waoss_section(diff_tab: pd.DataFrame) -> None:
    """Print the AOSS vs WAOSS difference test results."""
    print(f"\n{'-' * 70}")
    print(f"{' ' * 15}Difference test: AOSS and WAOSS")
    print(f"{'-' * 70}")
    print("H0: AOSS = WAOSS")
    tab_print(diff_tab)


def print_first_stage_section(fs_obj: Dict[str, Any]) -> None:
    """Print first-stage results header for IV-WAOSS."""
    print(f"\n{'=' * 80}")
    print(f"{' ' * 30}First stage estimation")
    print(f"{'=' * 80}")


def print_twfe_comparison(twfe_tab: pd.DataFrame) -> None:
    """Print TWFE comparison results."""
    print(f"\n{'-' * 70}")
    print(f"{' ' * 15}TWFE Comparison (Bootstrap)")
    print(f"{'-' * 70}")
    tab_print(twfe_tab)

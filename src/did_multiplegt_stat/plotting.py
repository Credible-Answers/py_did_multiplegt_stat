"""
Plotting functions for did_multiplegt_stat results.

This module provides event-study style plots that mimic the Stata ADO
output format, including confidence interval bands and by-group analysis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Default colors for by-group analysis (matching Stata)
BY_GROUP_COLORS = ["blue", "red", "green", "magenta", "gold", "lime", "cyan", "orange"]

# Default colors for estimators
ESTIMATOR_COLORS = {
    "aoss": "#1f77b4",      # Blue
    "waoss": "#ff7f0e",     # Orange
    "ivwaoss": "#2ca02c",   # Green
}


def _setup_figure(
    figsize: Tuple[float, float] = (10, 6),
    nrows: int = 1,
    ncols: int = 1,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Set up a matplotlib figure with subplots.

    Parameters
    ----------
    figsize : tuple
        Figure size in inches.
    nrows : int
        Number of subplot rows.
    ncols : int
        Number of subplot columns.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    axes : ndarray
        Array of axes.
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    return fig, axes


def _add_ci_bands(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    color: str,
    alpha: float = 0.2,
    label: Optional[str] = None,
) -> None:
    """
    Add confidence interval bands to an axis.

    Parameters
    ----------
    ax : Axes
        Matplotlib axis.
    x : ndarray
        X-axis values.
    y : ndarray
        Point estimates.
    lb : ndarray
        Lower bounds.
    ub : ndarray
        Upper bounds.
    color : str
        Color for the line and band.
    alpha : float
        Transparency for the band.
    label : str, optional
        Label for the legend.
    """
    # Plot point estimates with markers
    ax.plot(x, y, marker='o', color=color, linewidth=2, label=label, markersize=6)
    # Add CI band
    ax.fill_between(x, lb, ub, color=color, alpha=alpha)


def _add_zero_line(ax: plt.Axes) -> None:
    """Add a horizontal line at y=0."""
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)


def _format_axis(
    ax: plt.Axes,
    xlabel: str = "Relative Time",
    ylabel: str = "Effect Estimate",
    title: Optional[str] = None,
) -> None:
    """
    Format axis labels and title.

    Parameters
    ----------
    ax : Axes
        Matplotlib axis.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    title : str, optional
        Subplot title.
    """
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    ax.tick_params(axis='both', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _create_legend(ax: plt.Axes, loc: str = "best") -> None:
    """Add a legend to the axis."""
    ax.legend(loc=loc, framealpha=0.9, fontsize=10)


def plot_event_study(
    results: Dict[str, Any],
    estimator: Optional[str] = None,
    show_ci: bool = True,
    ci_alpha: float = 0.2,
    figsize: Tuple[float, float] = (10, 6),
    colors: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    xlabel: str = "Relative Time",
    ylabel: str = "Effect Estimate",
    show_zero_line: bool = True,
    separate_panels: bool = False,
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> Union[plt.Figure, Dict[str, plt.Figure]]:
    """
    Generate event-study style plots for DiD results.

    Parameters
    ----------
    results : dict
        Results dictionary from DIDMultiplegtStat.fit().
    estimator : str, optional
        Which estimator to plot ('aoss', 'waoss', 'ivwaoss').
        If None, plots all available estimators.
    show_ci : bool, default=True
        Display confidence interval bands.
    ci_alpha : float, default=0.2
        Transparency for CI bands.
    figsize : tuple, default=(10, 6)
        Figure size in inches.
    colors : dict, optional
        Custom colors for estimators.
    title : str, optional
        Custom title.
    xlabel : str, default="Relative Time"
        X-axis label.
    ylabel : str, default="Effect Estimate"
        Y-axis label.
    show_zero_line : bool, default=True
        Show horizontal line at y=0.
    separate_panels : bool, default=False
        Create separate subplots for each estimator.
    save_path : str, optional
        Path to save figure.
    dpi : int, default=150
        Resolution for saved figure.

    Returns
    -------
    fig : Figure or dict of Figures
        Matplotlib figure(s).
    """
    # Get estimator list from results
    args = results.get("args", {})
    estimator_list = args.get("estimator", ["aoss", "waoss"])
    if isinstance(estimator_list, str):
        estimator_list = [estimator_list]

    # Filter to requested estimator(s)
    if estimator is not None:
        estimator_list = [e for e in estimator_list if e == estimator]

    # Use custom colors if provided, otherwise defaults
    color_map = colors if colors else ESTIMATOR_COLORS

    # Get table data
    # Try to get by-level results first, then fall back to main results
    print_obj = results.get("results", results)
    table = print_obj.get("table")
    placebo_n = args.get("placebo", 0)
    disaggregate = args.get("disaggregate", False)
    pairs = int(print_obj.get("pairs", 1))

    if table is None or not isinstance(table, pd.DataFrame):
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data to plot", ha='center', va='center', fontsize=14)
        return fig

    estims_map = {"aoss": 0, "waoss": 1, "ivwaoss": 2}

    if separate_panels and len(estimator_list) > 1:
        # Create separate panels for each estimator
        n_estimators = len(estimator_list)
        fig, axes = _setup_figure(figsize=(figsize[0], figsize[1] * n_estimators),
                                   nrows=n_estimators, ncols=1)

        for i, est in enumerate(estimator_list):
            ax = axes[i, 0]
            _plot_single_estimator(
                ax=ax,
                table=table,
                estimator=est,
                estims_map=estims_map,
                pairs=pairs,
                disaggregate=disaggregate,
                placebo_n=placebo_n,
                placebo_tables=_get_placebo_tables(print_obj, placebo_n),
                color=color_map.get(est, "blue"),
                show_ci=show_ci,
                ci_alpha=ci_alpha,
                show_zero_line=show_zero_line,
            )
            _format_axis(ax, xlabel=xlabel, ylabel=ylabel, title=est.upper())

        plt.tight_layout()

    else:
        # Single panel with all estimators overlaid
        fig, ax = plt.subplots(figsize=figsize)

        for est in estimator_list:
            _plot_single_estimator(
                ax=ax,
                table=table,
                estimator=est,
                estims_map=estims_map,
                pairs=pairs,
                disaggregate=disaggregate,
                placebo_n=placebo_n,
                placebo_tables=_get_placebo_tables(print_obj, placebo_n),
                color=color_map.get(est, "blue"),
                show_ci=show_ci,
                ci_alpha=ci_alpha,
                show_zero_line=show_zero_line,
                label=est.upper(),
            )

        _format_axis(ax, xlabel=xlabel, ylabel=ylabel, title=title)
        if len(estimator_list) > 1:
            _create_legend(ax)
        if show_zero_line:
            _add_zero_line(ax)

        plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig


def _get_placebo_tables(print_obj: Dict[str, Any], placebo_n: int) -> Dict[int, pd.DataFrame]:
    """Extract placebo tables from results."""
    placebo_tables = {}
    for pl_idx in range(1, placebo_n + 1):
        table_p = print_obj.get(f"table_placebo_{pl_idx}", print_obj.get("table_placebo"))
        if isinstance(table_p, pd.DataFrame):
            placebo_tables[pl_idx] = table_p
    return placebo_tables


def _plot_single_estimator(
    ax: plt.Axes,
    table: pd.DataFrame,
    estimator: str,
    estims_map: Dict[str, int],
    pairs: int,
    disaggregate: bool,
    placebo_n: int,
    placebo_tables: Dict[int, pd.DataFrame],
    color: str,
    show_ci: bool,
    ci_alpha: float,
    show_zero_line: bool,
    label: Optional[str] = None,
) -> None:
    """Plot a single estimator's results."""
    # Get rows for this estimator
    l_bound = estims_map.get(estimator, 0) * pairs
    u_bound = l_bound + (pairs if disaggregate else 1)
    mat_sel = table.iloc[l_bound:u_bound]

    if mat_sel.empty:
        return

    # Build event-study data
    # Time 0 is the treatment period, negative is pre-treatment (placebos)
    x_vals = []
    y_vals = []
    lb_vals = []
    ub_vals = []

    # Add placebo points (negative time)
    estim_idx = estims_map.get(estimator, 0)
    for pl_idx in range(placebo_n, 0, -1):
        table_p = placebo_tables.get(pl_idx)
        if isinstance(table_p, pd.DataFrame) and estim_idx < len(table_p):
            row = table_p.iloc[estim_idx]
            if not np.isnan(row["Estimate"]):
                x_vals.append(-pl_idx)
                y_vals.append(row["Estimate"])
                lb_vals.append(row.get("LB CI", row["Estimate"] - 1.96 * row.get("SE", 0)))
                ub_vals.append(row.get("UB CI", row["Estimate"] + 1.96 * row.get("SE", 0)))

    # Add main estimate (time 0)
    if len(mat_sel) > 0:
        row = mat_sel.iloc[0]
        if not np.isnan(row["Estimate"]):
            x_vals.append(0)
            y_vals.append(row["Estimate"])
            lb_vals.append(row.get("LB CI", row["Estimate"] - 1.96 * row.get("SE", 0)))
            ub_vals.append(row.get("UB CI", row["Estimate"] + 1.96 * row.get("SE", 0)))

    # Add disaggregated estimates (positive time) if available
    if disaggregate and len(mat_sel) > 1:
        for t, (_, row) in enumerate(mat_sel.iloc[1:].iterrows(), start=1):
            if not np.isnan(row["Estimate"]):
                x_vals.append(t)
                y_vals.append(row["Estimate"])
                lb_vals.append(row.get("LB CI", row["Estimate"] - 1.96 * row.get("SE", 0)))
                ub_vals.append(row.get("UB CI", row["Estimate"] + 1.96 * row.get("SE", 0)))

    if not x_vals:
        return

    x = np.array(x_vals)
    y = np.array(y_vals)
    lb = np.array(lb_vals)
    ub = np.array(ub_vals)

    # Plot
    if show_ci:
        _add_ci_bands(ax, x, y, lb, ub, color=color, alpha=ci_alpha, label=label)
    else:
        ax.plot(x, y, marker='o', color=color, linewidth=2, label=label, markersize=6)


def plot_by_groups(
    results: Dict[str, Any],
    estimator: str = "aoss",
    show_ci: bool = True,
    ci_alpha: float = 0.15,
    figsize: Tuple[float, float] = (12, 6),
    colors: Optional[List[str]] = None,
    title: Optional[str] = None,
    xlabel: str = "Treatment Change",
    ylabel: str = "Effect Estimate",
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Generate plots for by-group analysis with multiple colored lines.

    Parameters
    ----------
    results : dict
        Results dictionary with by-group results.
    estimator : str, default="aoss"
        Which estimator to plot.
    show_ci : bool, default=True
        Display confidence interval bands.
    ci_alpha : float, default=0.15
        Transparency for CI bands.
    figsize : tuple, default=(12, 6)
        Figure size.
    colors : list, optional
        Custom colors for each group.
    title : str, optional
        Plot title.
    xlabel : str, default="Treatment Change"
        X-axis label.
    ylabel : str, default="Effect Estimate"
        Y-axis label.
    save_path : str, optional
        Path to save figure.
    dpi : int, default=150
        Resolution for saved figure.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    """
    by_levels = results.get("by_levels", [])
    color_list = colors if colors else BY_GROUP_COLORS

    fig, ax = plt.subplots(figsize=figsize)

    estims_map = {"aoss": 0, "waoss": 1, "ivwaoss": 2}
    estim_idx = estims_map.get(estimator, 0)

    for i, level in enumerate(by_levels):
        result_key = f"results_by_{i + 1}"
        print_obj = results.get(result_key)
        if print_obj is None:
            continue

        table = print_obj.get("table")
        if not isinstance(table, pd.DataFrame):
            continue

        pairs = int(print_obj.get("pairs", 1))
        l_bound = estim_idx * pairs
        u_bound = l_bound + 1  # Just first row for each by-group

        if l_bound >= len(table):
            continue

        row = table.iloc[l_bound]
        color = color_list[i % len(color_list)]

        y = row["Estimate"]
        lb = row.get("LB CI", y - 1.96 * row.get("SE", 0))
        ub = row.get("UB CI", y + 1.96 * row.get("SE", 0))

        # Plot as bar with error bars
        ax.bar(i, y, color=color, alpha=0.7, label=str(level))
        ax.errorbar(i, y, yerr=[[y - lb], [ub - y]], fmt='none',
                    color='black', capsize=5, capthick=1.5)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xticks(range(len(by_levels)))
    ax.set_xticklabels([str(l) for l in by_levels], rotation=45, ha='right')

    _format_axis(ax, xlabel=xlabel, ylabel=ylabel,
                 title=title or f"{estimator.upper()} by Group")
    ax.legend(title="Group", loc='best')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig


def plot_comparison(
    results: Dict[str, Any],
    estimators: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Estimator Comparison",
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Generate a side-by-side comparison plot of different estimators.

    Parameters
    ----------
    results : dict
        Results dictionary.
    estimators : list, optional
        Which estimators to compare.
    figsize : tuple, default=(10, 6)
        Figure size.
    title : str, default="Estimator Comparison"
        Plot title.
    save_path : str, optional
        Path to save figure.
    dpi : int, default=150
        Resolution.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    """
    args = results.get("args", {})
    estimator_list = estimators or args.get("estimator", ["aoss", "waoss"])
    if isinstance(estimator_list, str):
        estimator_list = [estimator_list]

    print_obj = results.get("results", results)
    table = print_obj.get("table")

    if not isinstance(table, pd.DataFrame):
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data to plot", ha='center', va='center')
        return fig

    estims_map = {"aoss": 0, "waoss": 1, "ivwaoss": 2}
    pairs = int(print_obj.get("pairs", 1))

    fig, ax = plt.subplots(figsize=figsize)

    x_pos = []
    y_vals = []
    errors = []
    labels = []
    colors = []

    for i, est in enumerate(estimator_list):
        l_bound = estims_map.get(est, 0) * pairs
        if l_bound >= len(table):
            continue

        row = table.iloc[l_bound]
        y = row["Estimate"]
        se = row.get("SE", 0)

        x_pos.append(i)
        y_vals.append(y)
        errors.append(1.96 * se)
        labels.append(est.upper())
        colors.append(ESTIMATOR_COLORS.get(est, "blue"))

    ax.bar(x_pos, y_vals, color=colors, alpha=0.7)
    ax.errorbar(x_pos, y_vals, yerr=errors, fmt='none',
                color='black', capsize=8, capthick=2)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)

    _format_axis(ax, xlabel="Estimator", ylabel="Effect Estimate", title=title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig

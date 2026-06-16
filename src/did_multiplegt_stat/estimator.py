"""
DIDMultiplegtStat: Scikit-learn style class for DiD estimation.

This module provides a class-based interface for the did_multiplegt_stat
estimator, following scikit-learn conventions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .core import did_multiplegt_stat as _did_multiplegt_stat
from .display import (
    print_header,
    print_estimator_section,
    print_placebo_section,
    print_aoss_vs_waoss_section,
    print_first_stage_section,
    print_twfe_comparison,
    mat_print,
    tab_print,
)
from .plotting import (
    plot_event_study,
    plot_by_groups,
    plot_comparison,
)


class DIDMultiplegtStat:
    """
    Difference-in-Differences estimator following de Chaisemartin & D'Haultfeuille (2024).

    Implements AS (Average Slope), WAS (Weighted Average Slope), and IV-WAS estimators
    with doubly-robust, regression-adjustment, and propensity-score methods.

    Parameters
    ----------
    estimator : str or list of str, optional
        Estimator type(s): 'aoss', 'waoss', 'ivwaoss'. Default: ['aoss', 'waoss'] or
        ['ivwaoss'] if Z is provided in fit().
    estimation_method : str, optional
        Method: 'ra' (regression adjustment), 'ps' (propensity score), 'dr' (doubly robust).
        Default: 'dr' without exact_match, 'ra' with exact_match.
    order : int or list of int, default=1
        Polynomial order. Can be single int or list of 4 (reg, logit_bis, logit_Plus,
        logit_Minus) or 8 (4 for first-stage + 4 for reduced-form for IV-WAS).
    noextrapolation : bool, default=False
        Restrict to common support without extrapolation.
    placebo : int, default=0
        Number of placebo tests.
    switchers : str, optional
        Restrict to 'up' (increasing treatment) or 'down' (decreasing treatment).
    disaggregate : bool, default=False
        Report period-specific estimates.
    aoss_vs_waoss : bool, default=False
        Test equality between AOSS and WAOSS.
    exact_match : bool, default=False
        Use exact matching on baseline treatment.
    by : list of str, optional
        Stratification variables.
    by_fd : int, optional
        Number of bins for first-difference quantiles.
    by_baseline : int, optional
        Number of bins for baseline treatment quantiles.
    other_treatments : list of str, optional
        Additional treatment variables to control for.
    cluster : str, optional
        Cluster variable for standard errors.
    weight : str, optional
        Observation weights variable.
    controls : list of str, optional
        Control variables.
    cross_fitting : int, default=0
        Number of cross-fitting folds.
    trimming : float, default=0
        Propensity score trimming threshold.
    on_placebo_sample : bool, default=False
        Estimate only on stayer sample.
    bootstrap : int, default=0
        Number of bootstrap replications.
    twfe : bool or dict, default=False
        Compare with TWFE regression.
    seed : int, default=0
        Random seed for reproducibility.
    cross_validation : dict, optional
        Cross-validation options for polynomial order selection.

    Attributes
    ----------
    results_ : dict
        Full results dictionary after fitting.
    table_ : pd.DataFrame
        Main results table with Estimate, SE, LB CI, UB CI, Switchers, Stayers.
    placebo_tables_ : dict
        Placebo test results by index.
    n_obs_ : int
        Number of observations.
    n_clusters_ : int or None
        Number of clusters (if clustered).
    by_levels_ : list or None
        Levels of by-group analysis.
    first_stage_ : DIDMultiplegtStat or None
        First-stage results for IV-WAS.
    is_fitted_ : bool
        Whether the model has been fitted.

    Examples
    --------
    >>> import pandas as pd
    >>> from stat_python import DIDMultiplegtStat
    >>>
    >>> # Basic usage
    >>> model = DIDMultiplegtStat(estimator=['aoss', 'waoss'])
    >>> model.fit(df, Y='outcome', ID='unit_id', Time='time', D='treatment')
    >>> model.summary()
    >>>
    >>> # With IV
    >>> model_iv = DIDMultiplegtStat(estimator='ivwaoss')
    >>> model_iv.fit(df, Y='outcome', ID='unit_id', Time='time', D='treatment', Z='instrument')
    >>> model_iv.plot()
    """

    def __init__(
        self,
        estimator: Optional[Union[str, Sequence[str]]] = None,
        estimation_method: Optional[str] = None,
        order: Union[int, List[int]] = 1,
        noextrapolation: bool = False,
        placebo: int = 0,
        switchers: Optional[str] = None,
        disaggregate: bool = False,
        aoss_vs_waoss: bool = False,
        exact_match: bool = False,
        by: Optional[Sequence[str]] = None,
        by_fd: Optional[int] = None,
        by_baseline: Optional[int] = None,
        other_treatments: Optional[Sequence[str]] = None,
        cluster: Optional[str] = None,
        weight: Optional[str] = None,
        controls: Optional[Sequence[str]] = None,
        cross_fitting: int = 0,
        trimming: float = 0,
        on_placebo_sample: bool = False,
        bootstrap: int = 0,
        twfe: Union[bool, Dict[str, Any]] = False,
        seed: int = 0,
        cross_validation: Optional[Dict[str, Any]] = None,
        iv_method: str = "manual",
        asinstata: bool = False,
        model_deltay: Optional[Any] = None,
        model_stayer: Optional[Any] = None,
    ) -> None:
        """Initialize the estimator with configuration parameters."""
        self.estimator = estimator
        self.estimation_method = estimation_method
        self.order = order
        self.noextrapolation = noextrapolation
        self.placebo = placebo
        self.switchers = switchers
        self.disaggregate = disaggregate
        self.aoss_vs_waoss = aoss_vs_waoss
        self.exact_match = exact_match
        self.by = by
        self.by_fd = by_fd
        self.by_baseline = by_baseline
        self.other_treatments = other_treatments
        self.cluster = cluster
        self.weight = weight
        self.controls = controls
        self.cross_fitting = cross_fitting
        self.trimming = trimming
        self.on_placebo_sample = on_placebo_sample
        self.bootstrap = bootstrap
        self.twfe = twfe
        self.seed = seed
        self.cross_validation = cross_validation
        self.iv_method = iv_method
        self.asinstata = asinstata
        self.model_deltay = model_deltay
        self.model_stayer = model_stayer

        # Fitted attributes (set after fit())
        self.results_: Optional[Dict[str, Any]] = None
        self.table_: Optional[pd.DataFrame] = None
        self.placebo_tables_: Optional[Dict[int, pd.DataFrame]] = None
        self.n_obs_: Optional[int] = None
        self.n_clusters_: Optional[int] = None
        self.by_levels_: Optional[List] = None
        self.first_stage_: Optional["DIDMultiplegtStat"] = None
        self.is_fitted_: bool = False

        # Data column names (set after fit())
        self._Y: Optional[str] = None
        self._ID: Optional[str] = None
        self._Time: Optional[str] = None
        self._D: Optional[str] = None
        self._Z: Optional[str] = None

    def fit(
        self,
        df: pd.DataFrame,
        Y: str,
        ID: str,
        Time: str,
        D: str,
        Z: Optional[str] = None,
    ) -> "DIDMultiplegtStat":
        """
        Fit the DiD estimator.

        Parameters
        ----------
        df : pd.DataFrame
            Panel data in long format.
        Y : str
            Column name for outcome variable.
        ID : str
            Column name for unit identifier.
        Time : str
            Column name for time variable.
        D : str
            Column name for treatment variable.
        Z : str, optional
            Column name for instrument variable (required for IV-WAS).

        Returns
        -------
        self : DIDMultiplegtStat
            Fitted estimator (scikit-learn convention).

        Raises
        ------
        ValueError
            If invalid parameter combinations are specified.
        """
        # Store column names
        self._Y = Y
        self._ID = ID
        self._Time = Time
        self._D = D
        self._Z = Z

        # Call the internal function
        self.results_ = _did_multiplegt_stat(
            df=df,
            Y=Y,
            ID=ID,
            Time=Time,
            D=D,
            Z=Z,
            estimator=self.estimator,
            estimation_method=self.estimation_method,
            order=self.order,
            noextrapolation=self.noextrapolation,
            placebo=self.placebo,
            switchers=self.switchers,
            disaggregate=self.disaggregate,
            aoss_vs_waoss=self.aoss_vs_waoss,
            exact_match=self.exact_match,
            by=self.by,
            by_fd=self.by_fd,
            by_baseline=self.by_baseline,
            other_treatments=self.other_treatments,
            cluster=self.cluster,
            weight=self.weight,
            controls=self.controls,
            cross_fitting=self.cross_fitting,
            trimming=self.trimming,
            on_placebo_sample=self.on_placebo_sample,
            bootstrap=self.bootstrap,
            twfe=self.twfe,
            seed=self.seed,
            cross_validation=self.cross_validation,
            iv_method=self.iv_method,
            asinstata=self.asinstata,
            model_deltay=self.model_deltay,
            model_stayer=self.model_stayer,
        )

        # Extract key results
        self._extract_results()
        self.is_fitted_ = True

        return self

    def _extract_results(self) -> None:
        """Extract key results from the full results dictionary."""
        if self.results_ is None:
            return

        # Get by-levels
        self.by_levels_ = self.results_.get("by_levels")

        # Determine which result key to use
        if self.by_levels_ is None:
            print_obj = self.results_.get("results", self.results_)
        else:
            # Use first by-level for main table
            print_obj = self.results_.get("results_by_1", self.results_.get("results", self.results_))

        # Extract table
        self.table_ = print_obj.get("table")

        # Extract N and clusters
        self.n_obs_ = print_obj.get("N")
        self.n_clusters_ = print_obj.get("n_clusters")

        # Extract placebo tables
        args = self.results_.get("args", {})
        placebo_n = args.get("placebo", 0)
        if placebo_n > 0:
            self.placebo_tables_ = {}
            for pl_idx in range(1, placebo_n + 1):
                table_p = print_obj.get(f"table_placebo_{pl_idx}", print_obj.get("table_placebo"))
                if isinstance(table_p, pd.DataFrame):
                    self.placebo_tables_[pl_idx] = table_p

        # Handle first-stage for IV-WAS
        fs_obj = self.results_.get("first_stage")
        if fs_obj is not None:
            self.first_stage_ = DIDMultiplegtStat()
            self.first_stage_.results_ = fs_obj
            self.first_stage_._extract_results()
            self.first_stage_.is_fitted_ = True

    def _check_is_fitted(self) -> None:
        """Check if the model has been fitted."""
        if not self.is_fitted_:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

    def summary(
        self,
        show_header: bool = True,
        show_placebo: bool = True,
        show_warnings: bool = True,
    ) -> None:
        """
        Print formatted summary mimicking Stata ADO output.

        Parameters
        ----------
        show_header : bool, default=True
            Display summary statistics header.
        show_placebo : bool, default=True
            Display placebo results if available.
        show_warnings : bool, default=True
            Display warnings about quasi-stayers, common support violations.
        """
        self._check_is_fitted()

        args = self.results_.get("args", {})
        estim_list = args.get("estimator", ["aoss", "waoss"])
        if isinstance(estim_list, str):
            estim_list = [estim_list]

        by_var = args.get("by")
        by_fd = args.get("by_fd")
        by_baseline = args.get("by_baseline")

        if by_var is None and by_fd is None and by_baseline is None:
            by_levs = ["_no_by"]
            by_obj = ["results"]
        else:
            by_levs = list(self.results_.get("by_levels", []))
            by_obj = [f"results_by_{j + 1}" for j in range(len(by_levs))]

        estims_map = {"aoss": 0, "waoss": 1, "ivwaoss": 2}

        for idx, key in enumerate(by_obj):
            print_obj = self.results_.get(key)
            if print_obj is None:
                continue

            by_level = by_levs[idx] if by_levs[idx] != "_no_by" else None

            if show_header:
                print_header(
                    N=print_obj.get("N", 0),
                    estimation_method=args.get("estimation_method", "dr"),
                    estimator_list=estim_list,
                    order=args.get("order"),
                    exact_match=args.get("exact_match", False),
                    noextrapolation=args.get("noextrapolation", False),
                    controls=args.get("controls"),
                    cross_fitting=args.get("cross_fitting", 0),
                    trimming=args.get("trimming", 0),
                    n_clusters=print_obj.get("n_clusters"),
                    cluster=args.get("cluster"),
                    by_level=by_level,
                )

            table = print_obj.get("table")
            pairs = int(print_obj.get("pairs", 1))

            for est in estim_list:
                print_estimator_section(
                    estimator=est,
                    table=table,
                    estims_map=estims_map,
                    pairs=pairs,
                    disaggregate=args.get("disaggregate", False),
                )

                # Placebo results
                placebo_n = args.get("placebo", 0)
                if show_placebo and placebo_n > 0:
                    placebo_tables = {}
                    for pl_idx in range(1, placebo_n + 1):
                        table_p = print_obj.get(f"table_placebo_{pl_idx}", print_obj.get("table_placebo"))
                        if isinstance(table_p, pd.DataFrame):
                            placebo_tables[pl_idx] = table_p
                    if placebo_tables:
                        print_placebo_section(
                            estimator=est,
                            placebo_tables=placebo_tables,
                            estims_map=estims_map,
                            placebo_n=placebo_n,
                        )

            # AOSS vs WAOSS test
            if args.get("aoss_vs_waoss"):
                diff_tab = print_obj.get("aoss_vs_waoss")
                if diff_tab is not None:
                    print_aoss_vs_waoss_section(diff_tab)

        # First-stage results (IV-WAOSS)
        if self.first_stage_ is not None:
            print_first_stage_section(self.first_stage_.results_)
            self.first_stage_.summary(show_header=show_header, show_placebo=show_placebo)
            print(f"{'=' * 80}")
            print(f"{' ' * 30}Reduced form estimation (above)")
            print(f"{'=' * 80}")

        # TWFE comparison
        twfe_tab = self.results_.get("twfe_comparison")
        if twfe_tab is not None:
            print_twfe_comparison(twfe_tab)

    def plot(
        self,
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
        Generate event-study style plots.

        Parameters
        ----------
        estimator : str, optional
            Which estimator to plot ('aoss', 'waoss', 'ivwaoss'). If None, plots all.
        show_ci : bool, default=True
            Display confidence interval bands.
        ci_alpha : float, default=0.2
            Transparency for CI bands.
        figsize : tuple, default=(10, 6)
            Figure size in inches.
        colors : dict, optional
            Custom colors for estimators {'aoss': 'blue', 'waoss': 'red', ...}.
        title : str, optional
            Custom title. Default: auto-generated based on estimator.
        xlabel : str, default="Relative Time"
            X-axis label.
        ylabel : str, default="Effect Estimate"
            Y-axis label.
        show_zero_line : bool, default=True
            Show horizontal line at y=0.
        separate_panels : bool, default=False
            Create separate subplots for AS/WAS/IV-WAS.
        save_path : str, optional
            Path to save figure.
        dpi : int, default=150
            Resolution for saved figure.

        Returns
        -------
        fig : matplotlib.figure.Figure or dict
            Single figure or dict of figures if separate_panels=True.
        """
        self._check_is_fitted()

        # Check for by-group analysis
        if self.by_levels_ is not None and len(self.by_levels_) > 1:
            return plot_by_groups(
                results=self.results_,
                estimator=estimator or "aoss",
                show_ci=show_ci,
                ci_alpha=ci_alpha,
                figsize=figsize,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                save_path=save_path,
                dpi=dpi,
            )

        return plot_event_study(
            results=self.results_,
            estimator=estimator,
            show_ci=show_ci,
            ci_alpha=ci_alpha,
            figsize=figsize,
            colors=colors,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            show_zero_line=show_zero_line,
            separate_panels=separate_panels,
            save_path=save_path,
            dpi=dpi,
        )

    def plot_comparison(
        self,
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
        estimators : list, optional
            Which estimators to compare. Default: all available.
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
        self._check_is_fitted()
        return plot_comparison(
            results=self.results_,
            estimators=estimators,
            figsize=figsize,
            title=title,
            save_path=save_path,
            dpi=dpi,
        )

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, return parameters for sub-objects.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            "estimator": self.estimator,
            "estimation_method": self.estimation_method,
            "order": self.order,
            "noextrapolation": self.noextrapolation,
            "placebo": self.placebo,
            "switchers": self.switchers,
            "disaggregate": self.disaggregate,
            "aoss_vs_waoss": self.aoss_vs_waoss,
            "exact_match": self.exact_match,
            "by": self.by,
            "by_fd": self.by_fd,
            "by_baseline": self.by_baseline,
            "other_treatments": self.other_treatments,
            "cluster": self.cluster,
            "weight": self.weight,
            "controls": self.controls,
            "cross_fitting": self.cross_fitting,
            "trimming": self.trimming,
            "on_placebo_sample": self.on_placebo_sample,
            "bootstrap": self.bootstrap,
            "twfe": self.twfe,
            "seed": self.seed,
            "cross_validation": self.cross_validation,
            "iv_method": self.iv_method,
            "asinstata": self.asinstata,
            "model_deltay": self.model_deltay,
            "model_stayer": self.model_stayer,
        }

    def set_params(self, **params) -> "DIDMultiplegtStat":
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : DIDMultiplegtStat
            Estimator instance.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return main results as a clean DataFrame.

        Returns
        -------
        df : pd.DataFrame
            Results table with Estimate, SE, LB CI, UB CI, Switchers, Stayers.
        """
        self._check_is_fitted()
        if self.table_ is None:
            return pd.DataFrame()
        return self.table_.copy()

    def get_coefficients(
        self,
        estimator: Optional[str] = None,
    ) -> pd.Series:
        """
        Get coefficient estimates for specified estimator.

        Parameters
        ----------
        estimator : str, optional
            Which estimator ('aoss', 'waoss', 'ivwaoss'). Default: first available.

        Returns
        -------
        coeffs : pd.Series
            Coefficient estimates.
        """
        self._check_is_fitted()

        if self.table_ is None:
            return pd.Series()

        args = self.results_.get("args", {})
        estim_list = args.get("estimator", ["aoss", "waoss"])
        if isinstance(estim_list, str):
            estim_list = [estim_list]

        if estimator is None:
            estimator = estim_list[0]

        estims_map = {"aoss": 0, "waoss": 1, "ivwaoss": 2}
        pairs = int(self.results_.get("results", self.results_).get("pairs", 1))

        l_bound = estims_map.get(estimator, 0) * pairs
        u_bound = l_bound + pairs

        if l_bound >= len(self.table_):
            return pd.Series()

        return self.table_.iloc[l_bound:u_bound]["Estimate"]

    def get_confidence_intervals(
        self,
        estimator: Optional[str] = None,
        level: float = 0.95,
    ) -> pd.DataFrame:
        """
        Get confidence intervals at specified level.

        Parameters
        ----------
        estimator : str, optional
            Which estimator. Default: first available.
        level : float, default=0.95
            Confidence level (e.g., 0.95 for 95% CI).

        Returns
        -------
        ci : pd.DataFrame
            DataFrame with columns 'LB CI' and 'UB CI'.
        """
        self._check_is_fitted()

        if self.table_ is None:
            return pd.DataFrame()

        args = self.results_.get("args", {})
        estim_list = args.get("estimator", ["aoss", "waoss"])
        if isinstance(estim_list, str):
            estim_list = [estim_list]

        if estimator is None:
            estimator = estim_list[0]

        estims_map = {"aoss": 0, "waoss": 1, "ivwaoss": 2}
        pairs = int(self.results_.get("results", self.results_).get("pairs", 1))

        l_bound = estims_map.get(estimator, 0) * pairs
        u_bound = l_bound + pairs

        if l_bound >= len(self.table_):
            return pd.DataFrame()

        # Note: Currently returns stored 95% CI; for different levels,
        # would need to recompute from SE
        return self.table_.iloc[l_bound:u_bound][["LB CI", "UB CI"]]

    def __repr__(self) -> str:
        """Return string representation of the estimator."""
        fitted_str = "fitted" if self.is_fitted_ else "not fitted"
        est_str = self.estimator if self.estimator else "auto"
        return f"DIDMultiplegtStat(estimator={est_str}, {fitted_str})"

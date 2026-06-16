"""
did_multiplegt_stat.py
Complete Python implementation of de Chaisemartin & D'Haultfœuille (2024).
Faithful translation from Stata's did_multiplegt_stat.ado.

Features:
  - AS (Average Slope), WAS (Weighted Average Slope), IV-WAS
  - Doubly-robust, regression-adjustment, propensity-score estimation
  - controls(), placebo(N), bootstrap(), twfe()
  - cross_validation(), cross_fitting(), trimming()
  - by_baseline(), on_placebo_sample, multi-order
  - by(), by_fd(), exact_match, noextrapolation
  - cluster-robust SEs, influence functions
"""

from __future__ import annotations

import math
import os
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm as _scipy_norm

# Optional IV regression packages - imported lazily to avoid hard dependencies
_LINEARMODELS_AVAILABLE = False
_ECONTOOLS_AVAILABLE = False

try:
    from linearmodels.iv import IV2SLS
    _LINEARMODELS_AVAILABLE = True
except ImportError:
    pass

try:
    import econtools.metrics as ecmt
    _ECONTOOLS_AVAILABLE = True
except ImportError:
    pass


# ============================================================
# IV REGRESSION HELPER FUNCTIONS
# ============================================================

def _iv_manual_2sls(df: pd.DataFrame, y_col: str, d_col: str, z_col: str,
                    exog_cols: list, controls: list = None) -> float:
    """
    Manual 2SLS implementation using two OLS regressions.
    This is the original implementation.

    Returns the coefficient on the endogenous variable D.
    """
    all_exog = exog_cols.copy() if exog_cols else []
    if controls:
        all_exog.extend(controls)

    # Quote column names that contain special characters
    def quote_col(c):
        if any(ch in c for ch in [' ', '-', '.', '(', ')', '+']):
            return f"Q('{c}')"
        return c

    z_quoted = quote_col(z_col)
    d_quoted = quote_col(d_col)
    y_quoted = quote_col(y_col)
    exog_quoted = [quote_col(c) for c in all_exog]

    # First stage: D ~ Z + exog
    if exog_quoted:
        fs_formula = f"{d_quoted} ~ {z_quoted} + " + " + ".join(exog_quoted)
    else:
        fs_formula = f"{d_quoted} ~ {z_quoted}"

    fs_model = smf.ols(fs_formula, data=df).fit()
    df = df.copy()
    df["_D_hat_XX"] = fs_model.fittedvalues

    # Second stage: Y ~ D_hat + exog
    if exog_quoted:
        ss_formula = f"{y_quoted} ~ _D_hat_XX + " + " + ".join(exog_quoted)
    else:
        ss_formula = f"{y_quoted} ~ _D_hat_XX"

    ss_model = smf.ols(ss_formula, data=df).fit()

    return ss_model.params.get("_D_hat_XX", np.nan)


def _iv_linearmodels(df: pd.DataFrame, y_col: str, d_col: str, z_col: str,
                     exog_cols: list, controls: list = None) -> float:
    """
    IV regression using linearmodels.IV2SLS.
    Properly computes standard errors accounting for first-stage estimation.

    Returns the coefficient on the endogenous variable D.
    """
    if not _LINEARMODELS_AVAILABLE:
        raise ImportError("linearmodels package not installed. Install with: pip install linearmodels")

    all_exog = ["const"] + exog_cols.copy()
    if controls:
        all_exog.extend(controls)

    # Prepare data
    df = df.copy()
    df["const"] = 1.0

    # Build formula: y ~ exog + [endog ~ instrument]
    exog_str = " + ".join(all_exog)
    formula = f"{y_col} ~ {exog_str} + [{d_col} ~ {z_col}]"

    try:
        mod = IV2SLS.from_formula(formula, data=df)
        res = mod.fit(cov_type="unadjusted")
        return res.params.get(d_col, np.nan)
    except Exception:
        # Fall back to manual 2SLS if linearmodels fails
        return _iv_manual_2sls(df, y_col, d_col, z_col, exog_cols, controls)


def _iv_econtools(df: pd.DataFrame, y_col: str, d_col: str, z_col: str,
                  exog_cols: list, controls: list = None) -> float:
    """
    IV regression using econtools.metrics.ivreg.
    Stata-like syntax and output.

    Returns the coefficient on the endogenous variable D.
    """
    if not _ECONTOOLS_AVAILABLE:
        raise ImportError("econtools package not installed. Install with: pip install econtools")

    all_exog = exog_cols.copy()
    if controls:
        all_exog.extend(controls)

    df = df.copy()

    try:
        # econtools.ivreg signature:
        # ivreg(df, y_name, x_name, z_name, w_name, addcons=True)
        # x_name = endogenous regressor(s)
        # z_name = instrument(s)
        # w_name = exogenous regressors
        results = ecmt.ivreg(df, y_col, [d_col], [z_col], all_exog, addcons=True)
        return results.beta.get(d_col, np.nan)
    except Exception:
        # Fall back to manual 2SLS if econtools fails
        return _iv_manual_2sls(df, y_col, d_col, z_col, exog_cols, controls)


def run_iv_regression(df: pd.DataFrame, y_col: str, d_col: str, z_col: str,
                      exog_cols: list, controls: list = None,
                      method: str = "manual") -> float:
    """
    Run IV regression using the specified method.

    Parameters
    ----------
    df : DataFrame
        Data for regression
    y_col : str
        Dependent variable column name
    d_col : str
        Endogenous variable column name
    z_col : str
        Instrument variable column name
    exog_cols : list
        List of exogenous regressor column names (e.g., fixed effects)
    controls : list, optional
        Additional control variables
    method : str
        IV method: "manual" (default), "linearmodels", or "econtools"

    Returns
    -------
    float
        Coefficient on the endogenous variable D
    """
    method = method.lower()

    if method == "manual":
        return _iv_manual_2sls(df, y_col, d_col, z_col, exog_cols, controls)
    elif method == "linearmodels":
        return _iv_linearmodels(df, y_col, d_col, z_col, exog_cols, controls)
    elif method == "econtools":
        return _iv_econtools(df, y_col, d_col, z_col, exog_cols, controls)
    else:
        raise ValueError(f"Unknown iv_method: {method}. Choose from: 'manual', 'linearmodels', 'econtools'")


# ============================================================
# Stata mt64s RNG — replicates Stata's `set seed` + `rnormal()`
# Reference: MT19937-64 by Matsumoto & Nishimura
# ============================================================

class _StataMT64:
    """Pure-Python replica of Stata's mt64s (MT19937-64) RNG."""
    _NN = 312
    _MM = 156
    _MATRIX_A = 0xB5026F5AA96619E9
    _UM = 0xFFFFFFFF80000000
    _LM = 0x7FFFFFFF
    _M64 = 0xFFFFFFFFFFFFFFFF

    def __init__(self, seed: int):
        mt = self._mt = [0] * self._NN
        mt[0] = seed & self._M64
        for i in range(1, self._NN):
            mt[i] = (6364136223846793005 * (mt[i - 1] ^ (mt[i - 1] >> 62)) + i) & self._M64
        self._mti = self._NN

    def _int64(self) -> int:
        mt = self._mt
        mag01 = (0, self._MATRIX_A)
        if self._mti >= self._NN:
            NN, MM, UM, LM = self._NN, self._MM, self._UM, self._LM
            for i in range(NN - MM):
                x = (mt[i] & UM) | (mt[i + 1] & LM)
                mt[i] = mt[i + MM] ^ (x >> 1) ^ mag01[x & 1]
            for i in range(NN - MM, NN - 1):
                x = (mt[i] & UM) | (mt[i + 1] & LM)
                mt[i] = mt[i + (MM - NN)] ^ (x >> 1) ^ mag01[x & 1]
            x = (mt[NN - 1] & UM) | (mt[0] & LM)
            mt[NN - 1] = mt[MM - 1] ^ (x >> 1) ^ mag01[x & 1]
            self._mti = 0
        x = mt[self._mti]
        self._mti += 1
        x ^= (x >> 29) & 0x5555555555555555
        x ^= (x << 17) & 0x71D67FFFEDA60000
        x ^= (x << 37) & 0xFFF7EEE000000000
        x ^= (x >> 43)
        return x & self._M64

    def runiform(self) -> float:
        """Stata's runiform(): open interval (0,1), genrand64_real3."""
        return ((self._int64() >> 12) + 0.5) * (1.0 / 4503599627370496.0)

    def rnormal(self) -> float:
        """Stata's rnormal() = invnormal(runiform())."""
        return float(_scipy_norm.ppf(self.runiform()))

    def rnormal_array(self, n: int) -> np.ndarray:
        return np.array([self.rnormal() for _ in range(n)])


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def _as_numpy(a) -> np.ndarray:
    if isinstance(a, (pd.Series, pd.Index)):
        return a.to_numpy(dtype=float)
    return np.asarray(a, dtype=float)


def wtd_mean(x, weights=None, na_rm: bool = True) -> float:
    x = _as_numpy(x)
    if weights is None:
        return float(np.nanmean(x) if na_rm else np.mean(x))
    w = _as_numpy(weights)
    if na_rm:
        s = ~np.isnan(x + w)
        x, w = x[s], w[s]
    sw = np.sum(w)
    return float(np.sum(w * x) / sw) if sw != 0 else float("nan")


def wtd_var(x, weights=None, normwt: bool = False, na_rm: bool = True, method: str = "unbiased") -> float:
    x = _as_numpy(x)
    if weights is None or len(_as_numpy(weights)) == 0:
        if na_rm:
            x = x[~np.isnan(x)]
        return float(np.var(x, ddof=1))
    w = _as_numpy(weights)
    if na_rm:
        s = ~np.isnan(x + w)
        x, w = x[s], w[s]
    if normwt:
        sw0 = np.sum(w)
        if sw0 != 0:
            w = w * (len(x) / sw0)
    if normwt or method.lower() == "ml":
        sw = np.sum(w)
        if sw == 0:
            return float("nan")
        w_norm = w / sw
        xbar = np.sum(w_norm * x)
        v_ml = np.sum(w_norm * (x - xbar) ** 2)
        if method.lower() == "ml":
            return float(v_ml)
        denom = 1.0 - np.sum(w_norm ** 2)
        return float(v_ml / denom) if denom != 0 else float("inf")
    sw = np.sum(w)
    xbar = np.sum(w * x) / sw if sw != 0 else float("nan")
    return float(np.sum(w * (x - xbar) ** 2) / (sw - 1))


def Mean(var: str, df: pd.DataFrame, w: str = "weight_XX") -> float:
    if var not in df.columns:
        return float("nan")
    _ensure_numeric(df, var)
    x = df[var].to_numpy(dtype=float)
    if w not in df.columns:
        v = x[~np.isnan(x)]
        return float(np.mean(v)) if len(v) > 0 else float("nan")
    _ensure_numeric(df, w)
    ww = df[w].to_numpy(dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(ww)
    if not np.any(mask):
        return float("nan")
    return float(np.average(x[mask], weights=ww[mask]))


def Sd(var: str, df: pd.DataFrame, w: str = "weight_XX") -> float:
    if var not in df.columns:
        return float("nan")
    return float(np.sqrt(wtd_var(df[var], df[w] if w in df.columns else None, normwt=False, na_rm=True)))


def wSum(df: pd.DataFrame, w: str = "weight_XX") -> float:
    if w not in df.columns:
        return float(len(df))
    return float(np.nansum(_as_numpy(df[w])))


def Sum(var: str, df: pd.DataFrame, w: str = "weight_XX") -> float:
    if var not in df.columns:
        return 0.0
    x = _as_numpy(df[var])
    if w not in df.columns:
        return float(np.nansum(x))
    ww = _as_numpy(df[w])
    s = ~np.isnan(x + ww)
    return float(np.sum(x[s] * ww[s])) if np.any(s) else 0.0


def _ensure_numeric(df: pd.DataFrame, col: str) -> None:
    if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
        df[col] = pd.to_numeric(df[col], errors="coerce")


def _nanmin_or_nan(s: pd.Series) -> float:
    arr = s.to_numpy(dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmin(arr))


def _nanmax_or_nan(s: pd.Series) -> float:
    arr = s.to_numpy(dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmax(arr))


def _nanmax_or_minus_inf(s: pd.Series) -> float:
    arr = s.to_numpy(dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("-inf")
    return float(np.nanmax(arr))


def _se_from_phi(phi: pd.Series) -> float:
    phi = pd.to_numeric(phi, errors="coerce").dropna()
    n = len(phi)
    if n <= 1:
        return np.nan
    return float(phi.std(ddof=1) / math.sqrt(n))


def _se_cluster_from_phi(ids_df: pd.DataFrame, cluster_col: str, phi_col: str, N_bar_c: float) -> float:
    """Aggregated cluster SE: Stata sums Phi within cluster, divides by N_bar_c,
    then computes sd/sqrt(n_clusters).  Equivalent to std(sum)/N_bar_c/sqrt(n)."""
    tmp = ids_df[[cluster_col, phi_col]].copy()
    tmp[phi_col] = pd.to_numeric(tmp[phi_col], errors="coerce")
    s = tmp.groupby(cluster_col)[phi_col].sum(min_count=1).dropna()
    if len(s) <= 1:
        return np.nan
    if not np.isfinite(N_bar_c) or N_bar_c <= 0:
        return np.nan
    # Stata: bysort cluster: Phi_c = total(Phi); Phi_c = Phi_c / N_bar_c;
    #        sum Phi_c; sd_delta = r(sd)/sqrt(r(sum_w))
    # = std(sum_phi) / N_bar_c / sqrt(n_clusters)
    return float(s.std(ddof=1) / float(N_bar_c) / math.sqrt(len(s)))


def by_check(df: pd.DataFrame, ID: str, by_var: str) -> bool:
    tmp = df[[ID, by_var]].dropna()
    if tmp.empty:
        return True
    n_unique = tmp.groupby(ID)[by_var].nunique(dropna=True)
    return bool((n_unique <= 1).all())


# ============================================================
# PANEL BALANCING
# ============================================================

def _balance_panel_fill(df: pd.DataFrame, id_col: str, t_col: str) -> pd.DataFrame:
    df = df.copy()
    times = pd.Series(df[t_col].dropna().unique()).sort_values().to_list()
    ids = pd.Series(df[id_col].dropna().unique()).to_list()
    idx = pd.MultiIndex.from_product([ids, times], names=[id_col, t_col])
    df0 = df.set_index([id_col, t_col])
    df_bal = df0.reindex(idx).reset_index()
    orig = df[[id_col, t_col]].copy()
    orig["_orig_row_XX"] = 1
    df_bal = df_bal.merge(orig, on=[id_col, t_col], how="left")
    df_bal["tsfilled_XX"] = df_bal["_orig_row_XX"].isna().astype(int)
    df_bal = df_bal.drop(columns=["_orig_row_XX"])
    t_map = {t: i + 1 for i, t in enumerate(times)}
    df_bal[t_col] = df_bal[t_col].map(t_map).astype(int)
    return df_bal


# ============================================================
# LOGIT / PREDICTION HELPERS
# ============================================================

def _expit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


class _CustomLogitResult:
    """Result wrapper for custom logit, compatible with lpredict()."""

    def __init__(self, params: np.ndarray, col_names: list):
        self.params = pd.Series(params, index=col_names)
        self._params_array = params
        self._col_names = col_names

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        X_cols = []
        for col_name in self._col_names:
            if col_name == "Intercept":
                X_cols.append(np.ones(n))
            elif col_name in df.columns:
                X_cols.append(df[col_name].to_numpy(dtype=float))
            else:
                X_cols.append(np.full(n, np.nan))
        X = np.column_stack(X_cols)
        xb = X @ self._params_array
        return _expit(xb)


def _is_binomial_glm(model) -> bool:
    try:
        if isinstance(model, _CustomLogitResult):
            return True
        if getattr(model, "is_logit", False):
            return True
        inner = getattr(model, "model", None)
        if inner is None:
            return False
        fam = getattr(inner, "family", None)
        if isinstance(fam, sm.families.Binomial):
            return True
        if isinstance(inner, sm.Logit):
            return True
        return False
    except Exception:
        return False


class _SvdWlsResult:
    """Result of SVD-based WLS that handles rank-deficient design matrices.

    Stata's `reg` uses Gauss-Jordan elimination and automatically drops collinear
    variables. With exact_match + high polynomial order, the design matrix is
    often rank-deficient in float32. This class mimics Stata's behavior by using
    SVD with a rank tolerance matching float32 precision (~1e-7).
    """

    def __init__(self, beta: np.ndarray, col_names: list, intercept_idx: int,
                 formula: str, condition_number: float = np.inf):
        self.params = pd.Series(beta, index=col_names)
        self._beta = beta
        self._col_names = col_names
        self._intercept_idx = intercept_idx
        self._formula = formula
        self.condition_number = condition_number

    def predict(self, df_new: pd.DataFrame) -> np.ndarray:
        # Build design matrix manually to handle NaN rows (patsy drops them)
        n = len(df_new)
        X_cols = []
        for col_name in self._col_names:
            if col_name == "Intercept":
                X_cols.append(np.ones(n))
            elif col_name in df_new.columns:
                X_cols.append(df_new[col_name].to_numpy(dtype=float))
            else:
                X_cols.append(np.full(n, np.nan))
        X = np.column_stack(X_cols)
        return X @ self._beta  # NaN in X → NaN in prediction


def _svd_wls(formula: str, df: pd.DataFrame, weights: pd.Series,
             rcond: float = 1e-7, use_float32: bool = False,
             use_rmcoll: bool = False):
    """Weighted least squares via sweep (Gauss-Jordan) or SVD with rank tolerance.

    Matches Stata's `reg` behavior: drops collinear terms automatically.

    When use_float32=True (exact_match), uses Stata's sweep algorithm:
    1. Data is cast to float32 (matching Stata's default float storage)
    2. Cross-product X'X is computed in float64 (matching Stata's internal math)
    3. Sweep (Gauss-Jordan) drops individual variables whose pivot falls below
       a tolerance, matching Stata's automatic collinearity handling in `reg`.

    When use_rmcoll=True (only with use_float32=True), uses Stata's _rmcoll
    algorithm: reverse variable order (right-to-left) + c(epsfloat) tolerance.
    This matches Stata's collinearity detection for severely ill-conditioned
    polynomial designs.  The intercept is protected from being dropped.

    When use_float32=False, uses SVD-based lstsq (standard WLS).
    """
    import patsy
    y_dm, X_dm = patsy.dmatrices(formula, data=df, return_type="dataframe")
    y = y_dm.to_numpy(dtype=float).ravel()
    X = X_dm.to_numpy(dtype=float)
    # patsy may drop NaN rows; align weights with the surviving rows
    w = weights.reindex(X_dm.index).to_numpy(dtype=float)

    W_sqrt = np.sqrt(np.clip(w, 0, None))
    Xw = X * W_sqrt[:, None]
    yw = y * W_sqrt

    if use_float32:
        # Stata stores data in float (float32). Cast to match.
        Xw = Xw.astype(np.float32).astype(np.float64)
        yw = yw.astype(np.float32).astype(np.float64)
        if not np.all(np.isfinite(Xw)) or not np.all(np.isfinite(yw)):
            raise ValueError("Regression failed: float32 overflow in design matrix")

        n, p = Xw.shape

        col_names = list(X_dm.columns)
        icept_idx = None
        for ci, cn in enumerate(col_names):
            if cn == "Intercept":
                icept_idx = ci
                break

        non_icept = [i for i in range(p) if i != icept_idx] if icept_idx is not None else list(range(p))

        if use_rmcoll and icept_idx is not None:
            # _rmcoll mode: reverse variable order (right-to-left processing)
            # with c(epsfloat) tolerance, matching Stata's _rmcoll collinearity
            # detection.  Intercept is last and protected from being dropped.
            reorder = non_icept[::-1] + [icept_idx]
            _SWEEP_TOL = float(np.finfo(np.float32).eps)  # ~1.19e-7
            _protect_last = True
        elif icept_idx is not None and icept_idx < p - 1:
            # Standard mode: forward variable order, intercept last, tight tol
            reorder = non_icept + [icept_idx]
            _SWEEP_TOL = 1e-15
            _protect_last = False
        else:
            reorder = None
            _SWEEP_TOL = 1e-15
            _protect_last = False

        if reorder is not None:
            Xw = Xw[:, reorder]

        # Build augmented cross-product matrix [X'X, X'y; y'X, y'y]
        XtX = Xw.T @ Xw
        Xty = Xw.T @ yw
        yty = float(yw @ yw)
        A = np.zeros((p + 1, p + 1))
        A[:p, :p] = XtX
        A[:p, p] = Xty
        A[p, :p] = Xty
        A[p, p] = yty

        orig_diag = np.diag(XtX).copy()
        dropped = np.zeros(p, dtype=bool)

        for k in range(p):
            pivot = A[k, k]
            threshold = _SWEEP_TOL * abs(orig_diag[k])
            is_protected = _protect_last and (k == p - 1)
            if (abs(pivot) < threshold or abs(orig_diag[k]) == 0) and not is_protected:
                dropped[k] = True
                A[k, :] = 0.0
                A[:, k] = 0.0
                continue
            pivot_inv = 1.0 / pivot
            row_k = A[k, :].copy()
            col_k = A[:, k].copy()
            A -= np.outer(col_k, row_k) * pivot_inv
            A[k, :] = row_k * pivot_inv
            A[:, k] = -col_k * pivot_inv
            A[k, k] = -pivot_inv

        beta = A[:p, p].copy()
        beta[dropped] = 0.0

        # Reorder beta back to patsy's original column order (intercept first)
        if reorder is not None:
            inv_reorder = [0] * p
            for new_pos, old_pos in enumerate(reorder):
                inv_reorder[old_pos] = new_pos
            beta = beta[inv_reorder]

        rank = int(np.sum(~dropped))
        if rank == 0:
            raise ValueError("Regression failed: all variables dropped by sweep")
        cond = np.inf
    else:
        beta, _, rank, sv = np.linalg.lstsq(Xw, yw, rcond=rcond)
        cond = float(sv[0] / sv[-1]) if len(sv) > 0 and sv[-1] > 0 else np.inf

    return _SvdWlsResult(beta, list(X_dm.columns), 0, formula, cond)


def stata_logit(formula: str, df: pd.DataFrame, wcol: str = "weight_XX",
                maxit: int = 16000, nrtol: float = 1e-5):
    """Logit matching Stata's ``logit ..., asis`` behaviour.

    From-scratch Modified Newton-Raphson (no statsmodels dependency).
    Convergence: nrtolerance g'(-H)^{-1}g < 1e-5 (Stata ml default).

    - Starting values: slopes=0, constant=ln(p/(1-p))  (Stata default)
    - nrtolerance:  g'(-H)^{-1}g < 1e-5               (Stata ml default)
    - Max iterations: 16 000                             (Stata c(maxiter))
    - Step halving with "back up" on failure.
    """
    import patsy
    y_dm, X_dm = patsy.dmatrices(formula, data=df, return_type="dataframe")
    y = y_dm.to_numpy(dtype=float).ravel()
    X = X_dm.to_numpy(dtype=float)
    col_names = list(X_dm.columns)
    n, k = X.shape

    # Stata starting values: slopes = 0, constant = ln(p/(1-p))
    p_mean = np.clip(y.mean(), 1e-15, 1 - 1e-15)
    params = np.zeros(k)
    if "Intercept" in col_names:
        params[col_names.index("Intercept")] = np.log(p_mean / (1 - p_mean))

    for _it in range(maxit):
        xb = X @ params
        pr = _expit(xb)

        # Gradient: X'(y - p)
        g = X.T @ (y - pr)

        # Hessian: -X' diag(p*(1-p)) X
        w = pr * (1.0 - pr)
        H = -(X.T * w) @ X

        # Newton step: solve (-H) * step = g
        try:
            step = np.linalg.solve(-H, g)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)):
            break

        # nrtolerance criterion: g'(-H)^{-1}g = g'step
        nrc = float(np.dot(g, step))
        if nrc < nrtol:
            break

        # Log-likelihood at current params
        ll_old = float(np.sum(y * xb - np.logaddexp(0.0, xb)))

        # Step halving: accept if LL does not decrease
        alpha = 1.0
        step_improved = False
        for _ in range(20):
            cand = params + alpha * step
            if np.all(np.isfinite(cand)):
                xb_cand = X @ cand
                ll_new = float(np.sum(y * xb_cand - np.logaddexp(0.0, xb_cand)))
                if np.isfinite(ll_new) and ll_new >= ll_old - 1e-10:
                    step_improved = True
                    break
            alpha *= 0.5

        if step_improved:
            params = params + alpha * step
        # else: "backed up" — stay at current params, try next iteration

    return _CustomLogitResult(params, col_names)


def lpredict(df: pd.DataFrame, outcol: str, fitted_model, prob: bool = False,
             sensitivity: float = 1e-10) -> pd.DataFrame:
    df = df.copy()
    try:
        pred = np.asarray(fitted_model.predict(df), dtype=float)
    except Exception:
        pred = np.full(len(df), np.nan)
    df[outcol] = pred
    is_binom = _is_binomial_glm(fitted_model)
    if prob or is_binom:
        p = df[outcol].to_numpy(dtype=float)
        p = np.where(np.isnan(p), 1.0, p)
        p = np.where(p < sensitivity, 0.0, p)
        df[outcol] = p
    return df


# ============================================================
# REGRESSION MODEL WRAPPERS (Stata vs scikit-learn)
# ============================================================
#
# Unified interface for OLS / logit estimation:
#   model.fit(formula, df, wcol=...)  -> returns self
#   model.predict(df)                 -> returns predictions (probabilities for logit)
#
# Stata wrappers (asinstata=True) reproduce the existing Stata-faithful
# behaviour (smf.ols, custom Newton-Raphson logit, smf.glm for CV).
# Sklearn wrappers (asinstata=False) parse the patsy formula and forward to
# scikit-learn's LinearRegression / LogisticRegression.  The wrappers expose a
# `.predict(df)` that builds the design matrix from a DataFrame so they remain
# drop-in compatible with `lpredict()`.  Logit wrappers set `is_logit=True`,
# which `_is_binomial_glm()` uses to apply the sensitivity / NaN handling.


def _patsy_design(formula: str, df: pd.DataFrame):
    """Build (y, X, col_names, has_intercept, cols_no_intercept) from a formula."""
    import patsy
    y_dm, X_dm = patsy.dmatrices(formula, data=df, return_type="dataframe")
    col_names = list(X_dm.columns)
    has_intercept = "Intercept" in col_names
    if has_intercept:
        int_idx = col_names.index("Intercept")
        cols_no_int = [c for i, c in enumerate(col_names) if i != int_idx]
        X = np.delete(X_dm.to_numpy(dtype=float), int_idx, axis=1)
    else:
        cols_no_int = col_names
        X = X_dm.to_numpy(dtype=float)
    y = y_dm.to_numpy(dtype=float).ravel()
    return y, X, col_names, has_intercept, cols_no_int


def _build_X_for_predict(df: pd.DataFrame, cols_no_int: list) -> np.ndarray:
    """Build design matrix from df for prediction, matching given column names.
    Missing columns -> NaN (predictions become NaN)."""
    n = len(df)
    if not cols_no_int:
        return np.zeros((n, 0))
    cols = []
    for c in cols_no_int:
        if c in df.columns:
            cols.append(df[c].to_numpy(dtype=float))
        else:
            cols.append(np.full(n, np.nan))
    return np.column_stack(cols)


class _StataOLSModel:
    """OLS wrapper using statsmodels.formula.api.ols (Stata-style)."""
    is_logit = False

    def fit(self, formula: str, df: pd.DataFrame, wcol: str | None = None):
        self._mod = smf.ols(formula, data=df).fit()
        self.params = self._mod.params
        return self

    def predict(self, df: pd.DataFrame):
        return self._mod.predict(df)


class _StataLogitModel:
    """Logit wrapper using the custom from-scratch Newton-Raphson `stata_logit`."""
    is_logit = True

    def fit(self, formula: str, df: pd.DataFrame, wcol: str = "weight_XX"):
        self._mod = stata_logit(formula, df, wcol=wcol)
        self.params = self._mod.params
        return self

    def predict(self, df: pd.DataFrame):
        return self._mod.predict(df)


class _StataGLMLogitModel:
    """Logit wrapper using statsmodels GLM (used in cross_validation_select)."""
    is_logit = True

    def fit(self, formula: str, df: pd.DataFrame, wcol: str | None = None):
        self._mod = smf.glm(formula, data=df, family=sm.families.Binomial()).fit(
            maxiter=300, disp=0)
        return self

    def predict(self, df: pd.DataFrame):
        return self._mod.predict(df)


class _SklearnOLSModel:
    """OLS wrapper using sklearn.linear_model.LinearRegression."""
    is_logit = False

    def fit(self, formula: str, df: pd.DataFrame, wcol: str | None = None):
        from sklearn.linear_model import LinearRegression
        y, X, col_names, has_int, cols_no_int = _patsy_design(formula, df)
        self._cols_no_int = cols_no_int
        self._has_intercept = has_int
        self._lr = LinearRegression(fit_intercept=has_int)
        self._lr.fit(X, y)
        # Build params Series for parity with statsmodels
        params = {}
        if has_int:
            params["Intercept"] = float(self._lr.intercept_)
        for c, b in zip(cols_no_int, self._lr.coef_, strict=False):
            params[c] = float(b)
        self.params = pd.Series(params)
        return self

    def predict(self, df: pd.DataFrame):
        X = _build_X_for_predict(df, self._cols_no_int)
        # sklearn doesn't propagate NaN; do it manually
        nan_rows = np.isnan(X).any(axis=1) if X.shape[1] > 0 else np.zeros(len(df), dtype=bool)
        X_safe = np.where(np.isnan(X), 0.0, X)
        pred = self._lr.predict(X_safe)
        pred = np.where(nan_rows, np.nan, pred)
        return pred


class _SklearnLogitModel:
    """Logit wrapper using sklearn.linear_model.LogisticRegression (unpenalised)."""
    is_logit = True

    def fit(self, formula: str, df: pd.DataFrame, wcol: str | None = None):
        from sklearn.linear_model import LogisticRegression
        y, X, col_names, has_int, cols_no_int = _patsy_design(formula, df)
        if len(np.unique(y[~np.isnan(y)])) <= 1:
            # Stata's `logit` errors with "outcome does not vary"; replicate.
            raise ValueError("Logit: outcome does not vary")
        self._cols_no_int = cols_no_int
        self._has_intercept = has_int
        # Use C=np.inf (no regularisation) — equivalent to MLE.  Forward-
        # compatible with sklearn 1.8+ which deprecates penalty=None in favour
        # of C=np.inf.  For older sklearn versions C=1e15 is effectively the
        # same and avoids any warning.
        try:
            self._lr = LogisticRegression(
                fit_intercept=has_int, C=np.inf, solver="lbfgs", max_iter=1000)
        except (ValueError, TypeError):
            self._lr = LogisticRegression(
                fit_intercept=has_int, C=1e15, solver="lbfgs", max_iter=1000)
        self._lr.fit(X, y.astype(int))
        params = {}
        if has_int:
            params["Intercept"] = float(self._lr.intercept_[0])
        for c, b in zip(cols_no_int, self._lr.coef_[0], strict=False):
            params[c] = float(b)
        self.params = pd.Series(params)
        return self

    def predict(self, df: pd.DataFrame):
        X = _build_X_for_predict(df, self._cols_no_int)
        nan_rows = np.isnan(X).any(axis=1) if X.shape[1] > 0 else np.zeros(len(df), dtype=bool)
        X_safe = np.where(np.isnan(X), 0.0, X)
        # P(y=1)
        proba = self._lr.predict_proba(X_safe)[:, 1]
        proba = np.where(nan_rows, np.nan, proba)
        return proba


class _CustomModelWrapper:
    """Wraps a user-provided sklearn-style model into the formula-based interface.

    The wrapped model must implement:
      - .fit(X, y)          — train on numpy arrays
      - .predict(X)         — predict (regression) or predict_proba(X) (classification)

    For classifiers, set ``is_classifier=True`` so that ``.predict()`` returns
    P(y=1) via ``predict_proba`` (falling back to ``.predict()`` otherwise).

    Each ``.fit()`` call clones the template via ``copy.deepcopy`` so the
    original object is never mutated.
    """

    def __init__(self, model_template, is_classifier: bool = False):
        self._template = model_template
        self._is_classifier = is_classifier
        self.is_logit = is_classifier

    def fit(self, formula: str, df: pd.DataFrame, wcol: str | None = None):
        import copy
        self._model = copy.deepcopy(self._template)
        y, X, col_names, has_int, cols_no_int = _patsy_design(formula, df)
        self._cols_no_int = cols_no_int
        self._has_intercept = has_int
        if self._is_classifier:
            y = y.astype(int)
        self._model.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame):
        X = _build_X_for_predict(df, self._cols_no_int)
        nan_rows = (np.isnan(X).any(axis=1) if X.shape[1] > 0
                    else np.zeros(len(df), dtype=bool))
        X_safe = np.where(np.isnan(X), 0.0, X)
        if self._is_classifier and hasattr(self._model, "predict_proba"):
            pred = self._model.predict_proba(X_safe)[:, 1]
        else:
            pred = np.asarray(self._model.predict(X_safe), dtype=float)
        pred = np.where(nan_rows, np.nan, pred)
        return pred


def _make_ols(asinstata: bool = True, custom_model=None):
    """Factory: OLS regression wrapper."""
    if custom_model is not None:
        return _CustomModelWrapper(custom_model, is_classifier=False)
    return _StataOLSModel() if asinstata else _SklearnOLSModel()


def _make_logit(asinstata: bool = True, custom_model=None):
    """Factory: logit wrapper (Stata custom Newton-Raphson vs sklearn)."""
    if custom_model is not None:
        return _CustomModelWrapper(custom_model, is_classifier=True)
    return _StataLogitModel() if asinstata else _SklearnLogitModel()


def _make_glm_logit(asinstata: bool = True, custom_model=None):
    """Factory: logit wrapper used in cross-validation (smf.glm vs sklearn)."""
    if custom_model is not None:
        return _CustomModelWrapper(custom_model, is_classifier=True)
    return _StataGLMLogitModel() if asinstata else _SklearnLogitModel()


# ============================================================
# POLYNOMIALS GENERATOR (with controls support)
# ============================================================

def polynomials_generator(df: pd.DataFrame, order: int, var_prefix: str = "D",
                          controls: list[str] | None = None,
                          other_treatments: list[str] | None = None):
    """
    Generate polynomial terms matching Stata's polynomials_generator.

    Creates: D1^1..D1^order, control^1..control^order,
    and when order>1: control*D1 interactions, control1*control2 cross-interactions.

    Uses raw (uncentered) polynomial terms matching Stata exactly.

    Returns: (df_with_new_cols, formula_terms_string)
    """
    var_col = f"{var_prefix}1_XX"
    pol_terms = []
    ctrl_terms = []

    # Stata uses raw (uncentered) polynomial terms: D1^k, ctrl^k
    # Stata's `gen` stores as float32; replicate truncation for parity.
    d1_f32 = df[var_col].to_numpy(dtype=np.float32)
    d1_f64 = d1_f32.astype(np.float64)

    for k in range(1, order + 1):
        col_name = f"D1_XX_{k}_XX"
        df[col_name] = (d1_f64 ** k).astype(np.float32).astype(np.float64)
        pol_terms.append(col_name)

        if controls:
            for ctrl in controls:
                ctrl_col = f"{ctrl}_{k}_XX"
                _ensure_numeric(df, ctrl)
                c_f32 = df[ctrl].to_numpy(dtype=np.float32).astype(np.float64)
                df[ctrl_col] = (c_f32 ** k).astype(np.float32).astype(np.float64)
                ctrl_terms.append(ctrl_col)

    if order > 1 and controls:
        # Stata: c.control#c.D1_XX (interaction = raw product)
        for ctrl in controls:
            interaction_col = f"{ctrl}_x_{var_prefix}1_XX"
            _ensure_numeric(df, ctrl)
            c_f32 = df[ctrl].to_numpy(dtype=np.float32).astype(np.float64)
            df[interaction_col] = (c_f32 * d1_f64).astype(np.float32).astype(np.float64)
            ctrl_terms.append(interaction_col)

        # Stata: c.control1#c.control2 (cross-interactions)
        if len(controls) > 1:
            for i in range(len(controls)):
                for j in range(i + 1, len(controls)):
                    c1, c2 = controls[i], controls[j]
                    cross_col = f"{c1}_x_{c2}_XX"
                    _ensure_numeric(df, c1)
                    _ensure_numeric(df, c2)
                    c1_f32 = df[c1].to_numpy(dtype=np.float32).astype(np.float64)
                    c2_f32 = df[c2].to_numpy(dtype=np.float32).astype(np.float64)
                    df[cross_col] = (c1_f32 * c2_f32).astype(np.float32).astype(np.float64)
                    ctrl_terms.append(cross_col)

    ot_terms = []
    if other_treatments:
        interact_parts = ["D1_XX_1_XX"]
        for v in other_treatments:
            interact_parts.append(v)
        if len(interact_parts) == 2:
            ot_col = f"D1_XX_1_XX_x_{other_treatments[0]}"
            _ensure_numeric(df, other_treatments[0])
            df[ot_col] = df["D1_XX_1_XX"].to_numpy(dtype=float) * df[other_treatments[0]].to_numpy(dtype=float)
            ot_terms.extend(other_treatments)
            ot_terms.append(ot_col)
        elif len(interact_parts) > 2:
            for v in other_treatments:
                _ensure_numeric(df, v)
                ot_terms.append(v)
            ot_col = "ot_interact_XX"
            df[ot_col] = df["D1_XX_1_XX"].to_numpy(dtype=float)
            for v in other_treatments:
                df[ot_col] = df[ot_col] * df[v].to_numpy(dtype=float)
            ot_terms.append(ot_col)

    all_terms = pol_terms + ot_terms + ctrl_terms
    formula_str = " + ".join(all_terms) if all_terms else "D1_XX_1_XX"
    return df, formula_str


# ============================================================
# CROSS VALIDATION
# ============================================================

def cross_validation_select(df: pd.DataFrame, outcome: str,
                            model_type: str = "reg",
                            algorithm: str = "kfolds",
                            tolerance: float = 0.01, max_k: int = 5,
                            seed: int = 0, kfolds: int = 5,
                            controls: list[str] | None = None,
                            first_stage: bool = False,
                            reduced_form: bool = False,
                            asinstata: bool = True,
                            model_deltay=None,
                            model_stayer=None) -> int:
    """
    K-fold cross-validation to select polynomial order.
    Returns the optimal order (integer).
    """
    df = df.copy()

    var_prefix = "Z" if (first_stage or reduced_form) else "D"

    _ensure_numeric(df, "ID_XX")
    _ensure_numeric(df, "T_XX")

    max_T = int(df["T_XX"].max())

    for t_val in df["T_XX"].dropna().unique():
        t_int = int(t_val)
        df[f"T_XX_FE_{t_int}"] = (df["T_XX"] == t_val).astype(float)

    lag_col = f"L{var_prefix}_XX"
    df = df.sort_values(["ID_XX", "T_XX"])
    df[lag_col] = df.groupby("ID_XX")[f"{var_prefix}_XX" if f"{var_prefix}_XX" in df.columns else "D_XX"].shift(1)

    for k in range(1, max_k + 1):
        col = f"Lag1Dt_{k}XX"
        df[col] = df[lag_col].to_numpy(dtype=float) ** k
        for t_int in range(1, max_T + 1):
            fe_col = f"T_XX_FE_{t_int}"
            if fe_col in df.columns:
                bis_col = f"bis{fe_col}_Lag1Dt_{k}XX"
                df[bis_col] = df[fe_col].to_numpy(dtype=float) * df[col].to_numpy(dtype=float)

    if seed != 0:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState(12345)

    df["random_shuffler_XX"] = rng.randn(len(df))
    df = df.sort_values("random_shuffler_XX").reset_index(drop=True)
    df["fold_identifier_XX"] = rng.randint(1, kfolds + 1, size=len(df))

    cv_scores = {}
    set_chosen_order_linear = False

    for k in range(1, max_k + 1):
        covariates = []
        for k2 in range(1, k + 1):
            for t_int in range(1, max_T + 1):
                bis_col = f"bisT_XX_FE_{t_int}_Lag1Dt_{k2}XX"
                if bis_col in df.columns:
                    covariates.append(bis_col)

        if not covariates:
            continue

        formula_rhs = " + ".join(covariates)

        # Include controls in CV formula (Stata: cv_covariates)
        if controls:
            for ctrl in controls:
                _ensure_numeric(df, ctrl)
            formula_rhs += " + " + " + ".join(controls)

        e_sq_list = []
        convergence_failures = 0

        for fold in range(1, kfolds + 1):
            train = df[df["fold_identifier_XX"] != fold].copy()
            test = df[df["fold_identifier_XX"] == fold].copy()

            if model_type in ("", "reg"):
                try:
                    formula = f"{outcome} ~ {formula_rhs}"
                    train_valid = train[train[outcome].notna()].copy()
                    if len(train_valid) < len(covariates) + 1:
                        convergence_failures += 1
                        continue
                    mod = _make_ols(asinstata, model_deltay).fit(formula, train_valid)
                    pred = np.asarray(mod.predict(test), dtype=float)
                    resid = test[outcome].to_numpy(dtype=float) - pred
                    e_sq = np.nanmean(resid ** 2)
                    e_sq_list.append(e_sq)
                except Exception:
                    convergence_failures += 1
            else:
                try:
                    formula = f"{outcome} ~ {formula_rhs}"
                    train_valid = train[train[outcome].notna()].copy()
                    mod = _make_glm_logit(asinstata, model_stayer).fit(formula, train_valid)
                    pred = np.asarray(mod.predict(test), dtype=float)
                    y_test = test[outcome].to_numpy(dtype=float)
                    pred = np.clip(pred, 1e-10, 1 - 1e-10)
                    log_loss = -np.nanmean(y_test * np.log(pred) + (1 - y_test) * np.log(1 - pred))
                    e_sq_list.append(log_loss)
                except Exception:
                    convergence_failures += 1

        if convergence_failures == kfolds:
            set_chosen_order_linear = True
            break

        cv_scores[k] = np.mean(e_sq_list) if e_sq_list else float("inf")

        if k > 1 and k - 1 in cv_scores and cv_scores[k - 1] != 0:
            diff = cv_scores[k] / cv_scores[k - 1] - 1
            if diff > -tolerance and not np.isnan(diff):
                return k - 1

    if set_chosen_order_linear:
        return 1

    if cv_scores:
        return min(cv_scores, key=cv_scores.get)
    return 1


# ============================================================
# PAIRWISE ESTIMATOR
# ============================================================

def did_multiplegt_stat_pairwise(
    df: pd.DataFrame,
    Y: str, ID: str, Time: str, D: str,
    Z: str | None,
    estimator: Any,
    order: int,
    noextrapolation: bool,
    weight: str | None,
    switchers: str | None,
    pairwise: int,
    IDs: Any,
    aoss: int, waoss: int, ivwaoss: int,
    estimation_method: str,
    scalars: dict[str, Any],
    placebo: int,
    exact_match: bool,
    cluster: str | None,
    by_fd_opt: int | None,
    other_treatments: list[str] | None,
    controls: list[str] | None = None,
    cross_fitting: int = 0,
    trimming: float = 0,
    on_placebo_sample: bool = False,
    order_reg: int | None = None,
    order_logit_bis: int | None = None,
    order_logit_Plus: int | None = None,
    order_logit_Minus: int | None = None,
    cf_folds_file: str | None = None,
    asinstata: bool = True,
    model_deltay=None,
    model_stayer=None,
) -> dict[str, Any]:
    """Pairwise DiD estimation between consecutive time periods."""

    df = df.copy()

    # Standardize column names
    for new, old in [("ID_XX", ID), ("T_XX", Time), ("Y_XX", Y), ("D_XX", D)]:
        if new not in df.columns and old in df.columns:
            df[new] = df[old]
    if Z is not None and "Z_XX" not in df.columns and Z in df.columns:
        df["Z_XX"] = df[Z]
    if "tsfilled_XX" not in df.columns:
        df["tsfilled_XX"] = 0.0
    if weight is not None and "weight_XX" not in df.columns and weight in df.columns:
        df["weight_XX"] = df[weight]
    if cluster is not None and "cluster_XX" not in df.columns and cluster in df.columns:
        df["cluster_XX"] = df[cluster]

    pl = "_pl" if placebo > 0 else ""
    placebo_index = placebo

    # Resolve multi-order: 4 separate orders for reg, logit_bis, logit_Plus, logit_Minus
    o_reg = order_reg if order_reg is not None else order
    o_logit_bis = order_logit_bis if order_logit_bis is not None else order
    o_logit_Plus = order_logit_Plus if order_logit_Plus is not None else order
    o_logit_Minus = order_logit_Minus if order_logit_Minus is not None else order

    # --- 1) Subset time window ---
    if placebo_index == 0:
        df = df[df["T_XX"].isin([pairwise - 1, pairwise])]
    else:
        periods = sorted(set([pairwise - placebo_index - 1, pairwise - placebo_index,
                              pairwise - 1, pairwise]))
        df = df[df["T_XX"].isin(periods)]

        # Lag controls to baseline period (Stata: lines 3082-3094)
        if controls:
            baseline_t = pairwise - placebo_index - 1
            for ctrl in controls:
                _ensure_numeric(df, ctrl)
                baseline_vals = df.loc[df["T_XX"] == baseline_t, ["ID_XX", ctrl]].copy()
                baseline_vals = baseline_vals.groupby("ID_XX")[ctrl].mean().reset_index()
                baseline_vals = baseline_vals.rename(columns={ctrl: f"_ctrl_base_{ctrl}_XX"})
                df = df.merge(baseline_vals, on="ID_XX", how="left")
                df[ctrl] = df[f"_ctrl_base_{ctrl}_XX"]
                df.drop(columns=[f"_ctrl_base_{ctrl}_XX"], inplace=True)

    # --- 2) Gap detection ---
    if len(df) == 0:
        gap_XX = 1.0
    else:
        df["tsfilled_min_XX"] = df.groupby("T_XX")["tsfilled_XX"].transform(_nanmin_or_nan)
        gap_XX = float(_nanmax_or_nan(df["tsfilled_min_XX"]))

    # --- 3) Relabel time ---
    if len(df) > 0:
        tvals = np.sort(df["T_XX"].dropna().unique())
        tmap = {t: i + 1 for i, t in enumerate(tvals)}
        df["T_XX"] = df["T_XX"].map(tmap).astype(float)

    # --- 4) First differences ---
    _ensure_numeric(df, "ID_XX")
    df = df.sort_values(["ID_XX", "T_XX"], kind="mergesort").reset_index(drop=True)
    g = df.groupby("ID_XX")
    df["delta_Y_XX"] = g["Y_XX"].diff()
    df["delta_D_XX"] = g["D_XX"].diff()
    if ivwaoss == 1:
        df["delta_Z_XX"] = g["Z_XX"].diff()

    if other_treatments:
        for v in other_treatments:
            df[f"_fdtmp_{v}"] = df.groupby("ID_XX")[v].diff()
        for v in other_treatments:
            df[f"fd_{v}_XX"] = df.groupby("ID_XX")[f"_fdtmp_{v}"].transform(
                lambda s: float(np.nansum(s.to_numpy(dtype=float))))
            df.drop(columns=[f"_fdtmp_{v}"], inplace=True)

    if "partition_XX" in df.columns:
        df["partition_lead_XX"] = df.groupby("ID_XX")["partition_XX"].shift(-1)
        df.drop(columns=["partition_XX"], inplace=True)

    # --- 5) Make delta_Y constant per ID ---
    if placebo_index == 0:
        df["delta_Y_XX"] = df.groupby("ID_XX")["delta_Y_XX"].transform("mean")
    else:
        df["delta_temp"] = np.where(df["T_XX"] == 2, df["delta_Y_XX"], np.nan)
        df["delta_Y_XX"] = df.groupby("ID_XX")["delta_temp"].transform("mean")
        df.drop(columns=["delta_temp"], inplace=True)

    # --- 6) Placebo restriction ---
    if placebo_index > 0 and (aoss == 1 or waoss == 1):
        df["inSamplePlacebo_temp_XX"] = np.where(
            (df["delta_D_XX"] == 0) & (df["T_XX"] == 2), 1.0, 0.0)
        df.loc[df["delta_D_XX"].isna(), "inSamplePlacebo_temp_XX"] = np.nan
        df["inSamplePlacebo_XX"] = df.groupby("ID_XX")["inSamplePlacebo_temp_XX"].transform(_nanmax_or_minus_inf)
        df = df[df["T_XX"] != 1]
        if placebo_index > 1:
            df = df[df["T_XX"] != 2]
        cutoff_t = 3 if placebo_index == 1 else 4
        df["delta_D_XX"] = np.where(df["T_XX"] != cutoff_t, np.nan, df["delta_D_XX"])

    if placebo_index > 0 and ivwaoss == 1:
        df["inSamplePlacebo_IV_temp_XX"] = np.where(
            (df["delta_Z_XX"] == 0) & (df["T_XX"] == 2), 1.0, 0.0)
        df.loc[df["delta_Z_XX"].isna(), "inSamplePlacebo_IV_temp_XX"] = np.nan
        df["inSamplePlacebo_XX"] = df.groupby("ID_XX")["inSamplePlacebo_IV_temp_XX"].transform(_nanmax_or_minus_inf)
        df = df[df["T_XX"] != 1]
        if placebo_index > 1:
            df = df[df["T_XX"] != 2]
        cutoff_t = 3 if placebo_index == 1 else 4
        df["delta_Z_XX"] = np.where(df["T_XX"] != cutoff_t, np.nan, df["delta_Z_XX"])
        df["delta_D_XX"] = np.where(df["T_XX"] != cutoff_t, np.nan, df["delta_D_XX"])

    # --- 7) Early exit if empty ---
    if len(df) == 0:
        if aoss == 1:
            scalars[f"P_{pairwise}{pl}_XX"] = 0.0
        if waoss == 1:
            scalars[f"E_abs_delta_D_{pairwise}{pl}_XX"] = 0.0
        if ivwaoss == 1:
            scalars[f"denom_delta_IV_{pairwise}{pl}_XX"] = 0.0
        scalars[f"non_missing_{pairwise}{pl}_XX"] = 0.0
        for v in ("Switchers", "Stayers"):
            for n in (1, 2, 3):
                scalars[f"N_{v}_{n}_{pairwise}{pl}_XX"] = 0.0
        for i in (1, 2, 3):
            scalars[f"delta_{i}_{pairwise}{pl}_XX"] = 0.0
            scalars[f"sd_delta_{i}_{pairwise}{pl}_XX"] = np.nan
        return {"scalars": scalars, "to_add": None}

    # --- 8) Make delta_D (and delta_Z) constant per ID ---
    df["delta_D_XX"] = df.groupby("ID_XX")["delta_D_XX"].transform("mean")
    if ivwaoss == 1:
        df["delta_Z_XX"] = df.groupby("ID_XX")["delta_Z_XX"].transform("mean")
        df["SI_XX"] = np.sign(df["delta_Z_XX"]).astype(float)
        df["Z1_XX"] = df["Z_XX"]

    # --- 9) Switcher indicators ---
    df[f"used_in_{pairwise}{pl}_XX"] = (
        (~df["delta_Y_XX"].isna()) & (~df["delta_D_XX"].isna())).astype(float)
    if ivwaoss == 1:
        df[f"used_in_IV_{pairwise}{pl}_XX"] = (
            (df[f"used_in_{pairwise}{pl}_XX"] == 1.0) & (~df["delta_Z_XX"].isna())).astype(float)
        df = df[df[f"used_in_IV_{pairwise}{pl}_XX"] == 1.0]

    df["S_XX"] = np.sign(df["delta_D_XX"]).astype(float)
    if waoss == 1 or aoss == 1:
        df["abs_delta_D_XX"] = df["S_XX"] * df["delta_D_XX"]
        if switchers == "up":
            df = df[df["S_XX"] != -1.0]
        elif switchers == "down":
            df = df[df["S_XX"] != 1.0]
    if ivwaoss == 1:
        if switchers == "up":
            df = df[df["SI_XX"] != -1.0]
        elif switchers == "down":
            df = df[df["SI_XX"] != 1.0]
        df["abs_delta_Z_XX"] = df["SI_XX"] * df["delta_Z_XX"]

    # --- 10) Forward shift weights (Stata line 3309: replace weights_XX = F.weights_XX) ---
    # Use weight from time t (the "second" period) instead of t-1 for the pair (t-1, t)
    df = df.sort_values(["ID_XX", "T_XX"], kind="mergesort")
    df["weight_XX"] = df.groupby("ID_XX")["weight_XX"].shift(-1)
    if "weight_c_XX" in df.columns:
        df["weight_c_XX"] = df.groupby("ID_XX")["weight_c_XX"].shift(-1)

    # Keep first row of pair ---
    df = df[df["T_XX"] != df["T_XX"].max()]
    df["D1_XX"] = df["D_XX"]
    df.drop(columns=["D_XX"], inplace=True, errors="ignore")
    df["Ht_XX"] = ((~df["delta_D_XX"].isna()) & (~df["delta_Y_XX"].isna())).astype(float)
    df.loc[df["Ht_XX"] == 0, "S_XX"] = np.nan
    if ivwaoss == 1:
        df["Ht_XX"] = ((df["Ht_XX"] == 1.0) & (~df["delta_Z_XX"].isna())).astype(float)
        df.loc[df["Ht_XX"] == 0, "SI_XX"] = np.nan

    if by_fd_opt is not None and "partition_lead_XX" in df.columns:
        df = df[(df["partition_lead_XX"] == 0) | (df["partition_lead_XX"] == by_fd_opt)]

    # --- 11) Set missing if placebo/other_treatments fail ---
    vars_to_set_missing = ["S_XX", "delta_D_XX", "delta_Y_XX", "D1_XX"]
    if aoss == 1 or waoss == 1:
        vars_to_set_missing += ["abs_delta_D_XX"]
    else:
        vars_to_set_missing += ["Z1_XX", "SI_XX"]
    # Stata line 3347: controls are included in vars_to_set_to_missing
    if controls:
        vars_to_set_missing += [c for c in controls if c in df.columns]

    if placebo_index > 0 and "inSamplePlacebo_XX" in df.columns:
        mask_bad = (df["inSamplePlacebo_XX"] == 0)
        for v in vars_to_set_missing:
            if v in df.columns:
                df.loc[mask_bad, v] = np.nan
        # Stata line 3363: replace Ht_XX = 0 (not missing!) if inSamplePlacebo_XX==0
        df.loc[mask_bad, "Ht_XX"] = 0.0

    if other_treatments:
        for ot in other_treatments:
            colfd = f"fd_{ot}_XX"
            if colfd in df.columns:
                mask_bad = (df[colfd] != 0)
                for v in vars_to_set_missing:
                    if v in df.columns:
                        df.loc[mask_bad, v] = np.nan
                # Stata: replace Ht_XX = 0 (not missing!) for other_treatments exclusion
                df.loc[mask_bad, "Ht_XX"] = 0.0

    # --- on_placebo_sample: keep only stayers ---
    if on_placebo_sample and "deltaDt_XX" in df.columns:
        # Stata lines 3072-3075, 3527-3528: keep only units where deltaDt_XX == 0
        # After step 10 (keep first row = pairwise-1), deltaDt_XX = D(pairwise-1) - D(pairwise-2)
        # i.e., units that were stayers in the previous period transition
        df = df[df["deltaDt_XX"] == 0].copy()

    # --- 12) No-extrapolation ---
    scalars.setdefault("N_drop_total_XX", 0.0)
    scalars.setdefault("N_drop_total_C_XX", 0.0)
    if noextrapolation:
        if aoss == 1 or waoss == 1:
            stayers = df[df["S_XX"] == 0]
            if len(stayers):
                max_D1 = float(np.nanmax(stayers["D1_XX"]))
                min_D1 = float(np.nanmin(stayers["D1_XX"]))
            else:
                max_D1, min_D1 = float("-inf"), float("inf")
            d1 = df["D1_XX"].to_numpy(dtype=float)
            df["outofBounds_XX"] = (~np.isnan(d1)) & ((d1 < min_D1) | (d1 > max_D1))
            N_drop = float(np.nansum(df["outofBounds_XX"].astype(float)))
            scalars[f"N_drop_{pairwise}{pl}_XX"] = N_drop
            df = df[~df["outofBounds_XX"]]
            if N_drop > 0 and placebo_index == 0 and gap_XX == 0:
                scalars["N_drop_total_XX"] += N_drop

        if ivwaoss == 1:
            stayers = df[df["SI_XX"] == 0]
            if len(stayers):
                max_Z1 = float(np.nanmax(stayers["Z1_XX"]))
                min_Z1 = float(np.nanmin(stayers["Z1_XX"]))
            else:
                max_Z1, min_Z1 = float("-inf"), float("inf")
            z1 = df["Z1_XX"].to_numpy(dtype=float)
            df["outofBoundsIV_XX"] = (~np.isnan(z1)) & ((z1 < min_Z1) | (z1 > max_Z1))
            N_IVdrop = float(np.nansum(df["outofBoundsIV_XX"].astype(float)))
            df = df[~df["outofBoundsIV_XX"]]
            if N_IVdrop > 0 and placebo_index == 0 and gap_XX == 0:
                scalars["N_drop_total_XX"] += N_IVdrop

    # --- 13) Exact matching ---
    if exact_match:
        if aoss == 1 or waoss == 1:
            group_cols = ["D1_XX"] + (other_treatments or [])
            g = df.groupby(group_cols, dropna=False)
            df["has_match_min_XX"] = g["abs_delta_D_XX"].transform(_nanmin_or_nan)
            df["has_match_max_XX"] = g["abs_delta_D_XX"].transform(_nanmax_or_minus_inf)
            df["s_has_match_XX"] = np.where(~df["S_XX"].isna(), (df["has_match_min_XX"] == 0).astype(float), -1.0)
            df.loc[df["S_XX"] == 0, "s_has_match_XX"] = -1.0
            df["c_has_match_XX"] = np.where(~df["S_XX"].isna(), (df["has_match_max_XX"] > 0).astype(float), -1.0)
            df.loc[(df["S_XX"] != 0) & (~df["S_XX"].isna()), "c_has_match_XX"] = -1.0
        else:
            group_cols = ["Z1_XX"] + (other_treatments or [])
            g = df.groupby(group_cols, dropna=False)
            df["has_match_min_XX"] = g["abs_delta_Z_XX"].transform(_nanmin_or_nan)
            df["has_match_max_XX"] = g["abs_delta_Z_XX"].transform(_nanmax_or_minus_inf)
            df["s_has_match_XX"] = np.where(~df["SI_XX"].isna(), (df["has_match_min_XX"] == 0).astype(float), -1.0)
            df.loc[df["SI_XX"] == 0, "s_has_match_XX"] = -1.0
            df["c_has_match_XX"] = np.where(~df["SI_XX"].isna(), (df["has_match_max_XX"] > 0).astype(float), -1.0)
            df.loc[(df["SI_XX"] != 0) & (~df["SI_XX"].isna()), "c_has_match_XX"] = -1.0

        mask_bad = (df["s_has_match_XX"] == 0) | (df["c_has_match_XX"] == 0)
        for v in vars_to_set_missing:
            if v in df.columns:
                df.loc[mask_bad, v] = np.nan
        # Stata: replace Ht_XX = 0 (not missing) for unmatched observations
        df.loc[mask_bad, "Ht_XX"] = 0.0

        if "D1_XX" in df.columns:
            nun = int(df["D1_XX"].nunique(dropna=True))
            if nun >= 1:
                # Stata: levelsof D1_XX; local order = r(r) — always K.
                # Generate K polynomial terms (matching Stata).
                # For mean_pred: use K-1 terms with smf.ols (full rank, good
                # point estimates). For ESbis/ES/SDD (SE only): use K terms
                # with sweep (float32) to match Stata's collinearity handling.
                o_reg = nun
                o_logit_bis = max(1, nun - 1)
                o_logit_Plus = max(1, nun - 1)
                o_logit_Minus = max(1, nun - 1)

        # Stata generates polynomial terms D1^1..D1^k as float variables.
        # When |D1|^k > Stata's maxfloat (1.7014e+38), the term is stored as
        # missing, and `reg` excludes those observations.  We replicate this
        # by identifying which D1 values would overflow and excluding them
        # from group-mean computations (setting predictions to NaN).
        _STATA_MAXFLOAT = 1.7014117331926443e+38
        _d1_all = df["D1_XX"].dropna().unique()
        _em_valid_d1 = set()
        for _d in _d1_all:
            _da = abs(float(_d))
            if _da <= 1.0 or _da ** o_reg <= _STATA_MAXFLOAT:
                _em_valid_d1.add(float(_d))
        # Boolean mask: True for observations with valid (non-overflow) D1
        _em_d1_ok = df["D1_XX"].apply(
            lambda x: float(x) in _em_valid_d1 if pd.notna(x) else False)

        df.drop(columns=[c for c in ["has_match_min_XX", "has_match_max_XX"] if c in df.columns], inplace=True)

    # --- 14) Bookkeeping ---
    if "weight_XX" not in df.columns:
        df["weight_XX"] = 1.0
    _ensure_numeric(df, "weight_XX")
    df["weight_XX"] = df["weight_XX"].fillna(0.0)

    scalars[f"W{pl}_XX"] = float(np.nansum(df["weight_XX"].to_numpy(dtype=float)))
    scalars[f"N{pl}_XX"] = float(len(df))

    if waoss == 1 or aoss == 1:
        scalars[f"N_Switchers{pl}_XX"] = float(((df["S_XX"] != 0) & (~df["S_XX"].isna())).sum())
        scalars[f"N_Stayers{pl}_XX"] = float((df["S_XX"] == 0).sum())
    if ivwaoss == 1:
        scalars[f"N_Switchers_IV{pl}_XX"] = float(((df["SI_XX"] != 0) & (~df["SI_XX"].isna())).sum())
        scalars[f"N_Stayers_IV{pl}_XX"] = float((df["SI_XX"] == 0).sum())

    # --- 14b) Stata: weight-as-control ---
    # Stata (lines 3601-3604): sum weights_XX; if (r(sd)!=0) local controls `controls' `weights'
    # When weights have non-zero variance, add them to controls for polynomial terms.
    # This applies regardless of exact_match.
    em_controls = list(controls) if controls else []
    w_arr = df["weight_XX"].to_numpy(dtype=float)
    if np.nanstd(w_arr) > 0:
        em_controls = em_controls + ["weight_XX"]

    # --- 15) Build polynomial terms using polynomials_generator ---
    _poly_controls = em_controls
    df, reg_pol_terms = polynomials_generator(df, o_reg, var_prefix="D",
                                              controls=_poly_controls, other_treatments=other_treatments)
    # For exact_match main effect: K-1 polynomial terms for mean_pred/SDD
    # (full rank with smf.ols → perfect interpolation = group means).
    # K terms (reg_pol_terms) are used for ESbis/ES regressions (sweep) and placebos.
    if exact_match and o_reg >= 2:
        _km1 = max(1, o_reg - 1)
        _, reg_pol_terms_km1 = polynomials_generator(df, _km1, var_prefix="D",
                                                      controls=_poly_controls, other_treatments=other_treatments)
    else:
        pass
    if not exact_match:
        _, logit_bis_pol = polynomials_generator(df, o_logit_bis, var_prefix="D",
                                                 controls=_poly_controls, other_treatments=other_treatments)
        _, logit_Plus_pol = polynomials_generator(df, o_logit_Plus, var_prefix="D",
                                                  controls=_poly_controls, other_treatments=other_treatments)
        _, logit_Minus_pol = polynomials_generator(df, o_logit_Minus, var_prefix="D",
                                                   controls=_poly_controls, other_treatments=other_treatments)
    else:
        logit_bis_pol = reg_pol_terms
        logit_Plus_pol = reg_pol_terms
        logit_Minus_pol = reg_pol_terms

    if ivwaoss == 1:
        df, IV_reg_pol_terms = polynomials_generator(df, o_reg, var_prefix="Z",
                                                     controls=_poly_controls, other_treatments=other_treatments)
        if not exact_match:
            _, IV_logit_bis_pol = polynomials_generator(df, o_logit_bis, var_prefix="Z",
                                                        controls=controls, other_treatments=other_treatments)
            _, IV_logit_Plus_pol = polynomials_generator(df, o_logit_Plus, var_prefix="Z",
                                                         controls=controls, other_treatments=other_treatments)
            _, IV_logit_Minus_pol = polynomials_generator(df, o_logit_Minus, var_prefix="Z",
                                                          controls=controls, other_treatments=other_treatments)
        else:
            IV_logit_bis_pol = IV_reg_pol_terms
            IV_logit_Plus_pol = IV_reg_pol_terms
            IV_logit_Minus_pol = IV_reg_pol_terms

    df["S_bis_XX"] = np.where(df["S_XX"].isna(), np.nan, (df["S_XX"] != 0).astype(float))

    # --- Cross-fitting: create sample IDs ---
    if cross_fitting > 0 and (waoss == 1 or aoss == 1):
        _cf_loaded = False
        if cf_folds_file is not None:
            # Load fold IDs from external file (exported by Stata .ado)
            try:
                _cf_df = pd.read_csv(cf_folds_file)
                _est_type = "aswas"
                _mask = (
                    (_cf_df["pairwise"] == pairwise) &
                    (_cf_df["placebo_index"] == placebo) &
                    (_cf_df["estimator_type"] == _est_type)
                )
                _cf_sub = _cf_df.loc[_mask, ["ID_XX", "cf_sample_id"]].copy()
                if len(_cf_sub) > 0:
                    _cf_sub["ID_XX"] = _cf_sub["ID_XX"].astype(df["ID_XX"].dtype)
                    _cf_sub["cf_sample_id"] = _cf_sub["cf_sample_id"].astype(int)
                    df = df.merge(_cf_sub, on="ID_XX", how="left")
                    _cf_loaded = True
            except Exception as e:
                warnings.warn(f"Could not load cf_folds_file: {e}. Falling back to internal RNG.", stacklevel=2)
        if not _cf_loaded:
            # Use Stata-compatible mt64s RNG so fold assignments match Stata exactly
            _rng_stata = _StataMT64(1234)
            df["rnd_sorter_XX"] = _rng_stata.rnormal_array(len(df))
            df = df.sort_values(["D1_XX", "rnd_sorter_XX"]).reset_index(drop=True)
            df["cf_sample_id"] = 1 + np.arange(len(df)) % cross_fitting

    # --- 16) Feasibility check ---
    if aoss == 1 or waoss == 1:
        feasible_est = (gap_XX == 0 and scalars[f"N_Switchers{pl}_XX"] > 0 and scalars[f"N_Stayers{pl}_XX"] > 1)
    else:
        feasible_est = (gap_XX == 0 and scalars[f"N_Switchers_IV{pl}_XX"] > 0 and scalars[f"N_Stayers_IV{pl}_XX"] > 1)

    scalars[f"P_Ht_{pairwise}{pl}_XX"] = Mean("Ht_XX", df)

    # --- 16b) Stata: cap reg feasibility check ---
    # Stata runs `cap reg deltaY_XX reg_vars_pol_XX if S_XX==0` and checks _rc==0.
    _cap_reg_model = None
    if feasible_est and (aoss == 1 or waoss == 1):
        df0_test = df[(df["S_XX"] == 0) & df["delta_Y_XX"].notna()].copy()
        if exact_match:
            # Use K terms with float32 sweep to match Stata's `reg` behavior:
            # Stata uses K polynomial terms + intercept, and drops collinear
            # columns via Gauss-Jordan elimination.  With high-degree polynomials
            # the result may NOT equal group means (R² < 1).
            ra_test_formula = f"delta_Y_XX ~ {reg_pol_terms}"
            try:
                _w_test = pd.Series(np.ones(len(df0_test)), index=df0_test.index)
                # First try standard sweep (forward order, tight tolerance)
                _cap_reg_model = _svd_wls(ra_test_formula, df0_test, _w_test,
                                          use_float32=True)
                # Check if the design is severely ill-conditioned (cond > 1e16).
                # When it is, the standard sweep keeps all columns but produces
                # numerically unstable predictions.  Switch to Stata's _rmcoll
                # algorithm (reverse sweep + epsfloat tolerance) which drops the
                # right columns for such cases.
                # When o_reg >= 6, the polynomial design matrix is severely
                # ill-conditioned (cond > 1e20).  Stata's _rmcoll drops the
                # most collinear columns via right-to-left sweep with
                # c(epsfloat) tolerance.  For lower orders the standard forward
                # sweep gives stable predictions matching Stata.
                _n_stayer_unique = int(df0_test["D1_XX"].nunique(dropna=True))
                if placebo_index > 0 and o_reg >= 6 and _n_stayer_unique >= 6:
                    _cap_reg_model = _svd_wls(ra_test_formula, df0_test, _w_test,
                                              use_float32=True, use_rmcoll=True)
            except Exception:
                feasible_est = False
                if placebo_index > 0 and os.environ.get("DMS_DEBUG_PLACEBO"):
                    _nun_st = int(df0_test["D1_XX"].nunique(dropna=True)) if len(df0_test) > 0 else 0
                    print(f"  PAIR p={pairwise} plac={placebo_index}: INFEASIBLE (sweep failed) n_st={len(df0_test)} nun_st={_nun_st}")
        else:
            ra_test_formula = f"delta_Y_XX ~ {reg_pol_terms}"
            try:
                _cap_reg_model = _make_ols(asinstata, model_deltay).fit(ra_test_formula, df0_test)
            except Exception:
                feasible_est = False
    if not feasible_est and placebo_index > 0 and os.environ.get("DMS_DEBUG_PLACEBO"):
        _sw = scalars.get(f"N_Switchers{pl}_XX", 0)
        _st = scalars.get(f"N_Stayers{pl}_XX", 0)
        if gap_XX != 0:
            print(f"  PAIR p={pairwise} plac={placebo_index}: INFEASIBLE (gap={gap_XX}) sw={_sw:.0f} st={_st:.0f}")
        elif _sw == 0:
            print(f"  PAIR p={pairwise} plac={placebo_index}: INFEASIBLE (no switchers) sw={_sw:.0f} st={_st:.0f}")
        elif _st <= 1:
            print(f"  PAIR p={pairwise} plac={placebo_index}: INFEASIBLE (few stayers) sw={_sw:.0f} st={_st:.0f}")

    # --- 17) Cluster preparation ---
    cluster_col = None
    if cluster is not None:
        if "cluster_XX" in df.columns:
            cluster_col = "cluster_XX"
        elif cluster in df.columns:
            df["cluster_XX"] = df[cluster]
            cluster_col = "cluster_XX"
        if cluster_col is not None:
            same_as_id = False
            try:
                same_as_id = df[cluster_col].astype("string").fillna("<NA>").equals(
                    df["ID_XX"].astype("string").fillna("<NA>"))
            except Exception:
                pass
            if same_as_id:
                cluster_col = None
                cluster = None
            else:
                if "weight_c_XX" not in df.columns:
                    _ensure_numeric(df, "weight_XX")
                    df["weight_c_XX"] = df.groupby([cluster_col, "T_XX"])["weight_XX"].transform("sum")
                _ensure_numeric(df, "weight_c_XX")
                df["_first_in_id"] = df.groupby("ID_XX").cumcount().eq(0).astype(float)
                df["_Nc"] = df.groupby(cluster_col)["_first_in_id"].transform(
                    lambda s: float(np.nansum(s.to_numpy(dtype=float))))
                scalars[f"N_bar_c_{pairwise}{pl}_XX"] = float(np.nanmean(df["_Nc"].to_numpy(dtype=float)))
                df.drop(columns=["_first_in_id", "_Nc"], inplace=True, errors="ignore")

    # ============================================================
    # 18) ESTIMATION
    # ============================================================
    if feasible_est:

        # Helper for cross-fitted regression
        def _cf_regression(df_in, formula, dep_var, pred_col, subset_mask=None,
                           use_logit=False, wcol="weight_XX", fallback_model=None):
            """Run cross-fitted regression: train leaving out each fold, predict on held-out.

            fallback_model: initial model to use when a fold's estimation fails.
            Stata's ``capture logit`` + ``predict`` reuses the last stored e(b)
            when the logit errors (e.g. "outcome does not vary").  Python's GLM
            does not raise an exception for all-zero response — it converges to
            extreme values.  To match Stata, we detect no-variation in the dep
            var before fitting and fall back to the last successful model.
            """
            df_in[f"cf_{pred_col}"] = np.nan
            last_model = fallback_model  # Stata: e(b) starts as full-sample model
            for cf_id in range(1, cross_fitting + 1):
                train_mask = df_in["cf_sample_id"] != cf_id
                test_mask = df_in["cf_sample_id"] == cf_id
                if subset_mask is not None:
                    train_mask = train_mask & subset_mask
                train = df_in[train_mask].copy()
                test = df_in[test_mask].copy()
                if len(train) < 2:
                    if last_model is not None:
                        test = lpredict(test, f"cf_{pred_col}", last_model, prob=use_logit)
                        df_in.loc[test.index, f"cf_{pred_col}"] = test[f"cf_{pred_col}"]
                    continue
                # Stata: logit errors with "outcome does not vary" (r(2000))
                # when the dep var is constant.  capture eats the error, predict
                # reuses last e(b).  Python's GLM silently converges to extreme
                # values instead of raising, so we must detect this explicitly.
                _dep_vals = train[dep_var].dropna()
                if use_logit and len(_dep_vals) > 0 and _dep_vals.nunique() <= 1:
                    if last_model is not None:
                        test = lpredict(test, f"cf_{pred_col}", last_model, prob=use_logit)
                        df_in.loc[test.index, f"cf_{pred_col}"] = test[f"cf_{pred_col}"]
                    continue
                try:
                    if use_logit:
                        mod = _make_logit(asinstata, model_stayer).fit(formula, train, wcol=wcol)
                    else:
                        # Stata: reg ... (NO weights for polynomial regressions)
                        mod = _make_ols(asinstata, model_deltay).fit(formula, train)
                    last_model = mod
                    test = lpredict(test, f"cf_{pred_col}", mod, prob=use_logit)
                    df_in.loc[test.index, f"cf_{pred_col}"] = test[f"cf_{pred_col}"]
                except Exception:
                    if last_model is not None:
                        test = lpredict(test, f"cf_{pred_col}", last_model, prob=use_logit)
                        df_in.loc[test.index, f"cf_{pred_col}"] = test[f"cf_{pred_col}"]
            return df_in

        # --- 18A) Common prelims AOSS/WAOSS ---
        if waoss == 1 or aoss == 1:
            df0 = df[df["S_XX"] == 0].copy()
            if _cap_reg_model is not None:
                ra_model = _cap_reg_model
                df = lpredict(df, "mean_pred_XX", ra_model)
            elif exact_match:
                # Group-means fallback: equivalent to saturated polynomial OLS
                ra_formula = f"delta_Y_XX ~ {reg_pol_terms}"
                _gm_dy = df0.groupby("D1_XX")["delta_Y_XX"].mean()
                df["mean_pred_XX"] = df["D1_XX"].map(_gm_dy).astype(float)
                df.loc[~_em_d1_ok, "mean_pred_XX"] = np.nan
            else:
                ra_formula = f"delta_Y_XX ~ {reg_pol_terms}"
                try:
                    ra_model = _make_ols(asinstata, model_deltay).fit(ra_formula, df0)
                    df = lpredict(df, "mean_pred_XX", ra_model)
                except Exception:
                    df["mean_pred_XX"] = 0.0

            df["inner_sum_delta_1_2_XX"] = df["delta_Y_XX"] - df["mean_pred_XX"]
            df["S0_XX"] = 1.0 - df["S_bis_XX"]
            # Stata: SbisV_XX = Sbis_XX * weights_XX (used in Phi and ES)
            df["SbisV_XX"] = df["S_bis_XX"] * df["weight_XX"]

            ra_formula = f"delta_Y_XX ~ {reg_pol_terms}"
            # Cross-fitting for E[ΔY|D1,S=0]
            if cross_fitting > 0:
                df = _cf_regression(df, ra_formula, "delta_Y_XX", "mean_pred_XX",
                                    subset_mask=(df["S_XX"] == 0))
                df["cf_inner_sum_delta_1_2_XX"] = df["delta_Y_XX"] - df["cf_mean_pred_XX"]

            if not exact_match:
                ps0_formula = f"S0_XX ~ {logit_bis_pol}"
                ps0_model = None
                try:
                    ps0_model = _make_logit(asinstata, model_stayer).fit(ps0_formula, df)
                    df = lpredict(df, "PS_0_D_1_XX", ps0_model, prob=True)
                except Exception:
                    df["PS_0_D_1_XX"] = 0.5

                # Stata: replace PS0D1_XX=0 if PS0D1_XX<=10^(-10)
                df.loc[df["PS_0_D_1_XX"] <= 1e-10, "PS_0_D_1_XX"] = 0.0

                # Cross-fitting for P(S=0|D1)
                if cross_fitting > 0:
                    df = _cf_regression(df, ps0_formula, "S0_XX", "PS_0_D_1_XX",
                                        use_logit=True, fallback_model=ps0_model)
                    df.loc[df["cf_PS_0_D_1_XX"] <= 1e-10, "cf_PS_0_D_1_XX"] = 0.0
                    # Trimming on cross-fitted PS
                    if trimming > 0:
                        df["trimmed_out_XX"] = (df["cf_PS_0_D_1_XX"] < trimming) & df["cf_PS_0_D_1_XX"].notna()
                        df.loc[df["trimmed_out_XX"], "cf_PS_0_D_1_XX"] = np.nan
                elif trimming > 0:
                    # Stata: trimming without cross_fitting is not allowed (ignored)
                    pass
            else:
                # exact_match: Stata uses reg Sbis/S on K polynomial terms (all obs)
                # Use K terms with float32 sweep to match Stata's collinearity handling.
                esbis_formula = f"S_bis_XX ~ {reg_pol_terms}"
                es_formula = f"S_XX ~ {reg_pol_terms}"
                try:
                    _w_es = pd.Series(np.ones(len(df)), index=df.index)
                    _esbis_m = _svd_wls(esbis_formula, df, _w_es, use_float32=True)
                    df = lpredict(df, "ES_bis_XX_D_1", _esbis_m)
                except Exception:
                    _df_ok = df[_em_d1_ok]
                    _gm_sbis = _df_ok.groupby("D1_XX")["S_bis_XX"].mean()
                    df["ES_bis_XX_D_1"] = df["D1_XX"].map(_gm_sbis).astype(float)
                try:
                    _es_m = _svd_wls(es_formula, df, _w_es, use_float32=True)
                    df = lpredict(df, "ES_XX_D_1", _es_m)
                except Exception:
                    _df_ok = df[_em_d1_ok]
                    _gm_s = _df_ok.groupby("D1_XX")["S_XX"].mean()
                    df["ES_XX_D_1"] = df["D1_XX"].map(_gm_s).astype(float)

                if cross_fitting > 0:
                    esbis_formula = f"S_bis_XX ~ {reg_pol_terms}"
                    es_formula = f"S_XX ~ {reg_pol_terms}"
                    df = _cf_regression(df, esbis_formula, "S_bis_XX", "ES_bis_XX_D_1")
                    df = _cf_regression(df, es_formula, "S_XX", "ES_XX_D_1")

            scalars[f"PS_0{pl}_XX"] = Mean("S0_XX", df)

        # Choose which inner_sum to use (CF or full)
        inner_sum_col = "cf_inner_sum_delta_1_2_XX" if (cross_fitting > 0 and "cf_inner_sum_delta_1_2_XX" in df.columns) else "inner_sum_delta_1_2_XX"
        ps0_col = "cf_PS_0_D_1_XX" if (cross_fitting > 0 and "cf_PS_0_D_1_XX" in df.columns and not exact_match) else "PS_0_D_1_XX"
        esbis_col = "cf_ES_bis_XX_D_1" if (cross_fitting > 0 and "cf_ES_bis_XX_D_1" in df.columns and exact_match) else "ES_bis_XX_D_1"
        es_col = "cf_ES_XX_D_1" if (cross_fitting > 0 and "cf_ES_XX_D_1" in df.columns and exact_match) else "ES_XX_D_1"

        # --- 18B) AOSS ---
        if aoss == 1:
            # Stata: sum SbisV_XX; scalar ES = r(mean)
            # ES is the unweighted mean of SbisV (= Sbis * weights)
            _sbv = df["SbisV_XX"].to_numpy(dtype=float)
            _sbv_nm = _sbv[~np.isnan(_sbv)]
            ES = float(np.mean(_sbv_nm)) if len(_sbv_nm) > 0 else 0.0
            scalars[f"ES{pl}_XX"] = ES
            scalars[f"P_{pairwise}{pl}_XX"] = ES * scalars[f"P_Ht_{pairwise}{pl}_XX"]
            scalars[f"PS_sum{pl}_XX"] = scalars.get(f"PS_sum{pl}_XX", 0.0) + scalars[f"P_{pairwise}{pl}_XX"]

            df["S_over_delta_D_XX"] = df["S_bis_XX"] / df["delta_D_XX"]
            df.loc[df["S_bis_XX"] == 0, "S_over_delta_D_XX"] = 0.0

            # Stata: reg S_over_deltaD_XX vars (unweighted for exact_match)
            # Use K terms with float32 sweep to match Stata's collinearity handling.
            _sdd_pol = reg_pol_terms  # K terms for both exact_match and normal
            sdd_formula = f"S_over_delta_D_XX ~ {_sdd_pol}"
            if exact_match:
                try:
                    _w_sdd = pd.Series(np.ones(len(df)), index=df.index)
                    _sdd_m = _svd_wls(sdd_formula, df, _w_sdd, use_float32=True)
                    df = lpredict(df, "mean_S_over_delta_D_XX", _sdd_m)
                except Exception:
                    _df_ok_sdd = df[_em_d1_ok]
                    _gm_sdd = _df_ok_sdd.groupby("D1_XX")["S_over_delta_D_XX"].mean()
                    df["mean_S_over_delta_D_XX"] = df["D1_XX"].map(_gm_sdd).astype(float)
                    df.loc[~_em_d1_ok, "mean_S_over_delta_D_XX"] = np.nan
            else:
                # Stata: reg S_over_deltaD_XX ... (NO weights)
                sdd_model = _make_ols(asinstata, model_deltay).fit(sdd_formula, df)
                df = lpredict(df, "mean_S_over_delta_D_XX", sdd_model)

            # Cross-fitting for E[S/ΔD|D1]
            if cross_fitting > 0:
                df = _cf_regression(df, sdd_formula, "S_over_delta_D_XX", "mean_S_over_delta_D_XX")
                sdd_cf_col = "cf_mean_S_over_delta_D_XX"
            else:
                sdd_cf_col = "mean_S_over_delta_D_XX"

            if exact_match:
                # Stata: delta_1 = mean(inner/deltaD * V) / ES
                # = mean(inner_sumdelta1_VXX) / mean(SbisV_XX)
                df["_inner_sum_d1_XX"] = df[inner_sum_col] / df["delta_D_XX"]
                df.loc[df["delta_D_XX"] == 0, "_inner_sum_d1_XX"] = 0.0
                df["_inner_sum_d1_V_XX"] = df["_inner_sum_d1_XX"] * df["weight_XX"]
                _d1v = df["_inner_sum_d1_V_XX"].to_numpy(dtype=float)
                _d1v_nm = _d1v[~np.isnan(_d1v)]
                d1_num = float(np.mean(_d1v_nm)) if len(_d1v_nm) > 0 else 0.0
                scalars[f"delta_1_{pairwise}{pl}_XX"] = d1_num / ES if ES != 0 else 0.0
                if placebo_index > 0 and os.environ.get("DMS_DEBUG_PLACEBO"):
                    _n_sw = int(((df["S_XX"] != 0) & df["S_XX"].notna()).sum())
                    _n_st = int((df["S_XX"] == 0).sum())
                    _P_Ht = scalars[f"P_Ht_{pairwise}{pl}_XX"]
                    _P_p = scalars[f"P_{pairwise}{pl}_XX"]
                    _n_ht1 = int((df["Ht_XX"] == 1).sum())
                    _n_ht0 = int((df["Ht_XX"] == 0).sum())
                    _n_d1_unique = int(df["D1_XX"].nunique(dropna=True))
                    _cap_ok = "YES" if _cap_reg_model is not None else "NO"
                    print(f"  PAIR p={pairwise} plac={placebo_index}: d1={d1_num/ES if ES!=0 else 0:.8f} ES={ES:.4f} P_Ht={_P_Ht:.4f} P_p={_P_p:.6f} N={len(df)} sw={_n_sw} st={_n_st} Ht1={_n_ht1} Ht0={_n_ht0} order={_n_d1_unique} cap_reg={_cap_ok}")
                    if os.environ.get("DMS_DEBUG_PLACEBO") == "DETAIL" and pairwise == int(os.environ.get("DMS_DEBUG_PAIR", "3")):
                        _dbg = df[["ID_XX","D1_XX","delta_Y_XX","delta_D_XX","S_XX","S_bis_XX","Ht_XX","weight_XX","mean_pred_XX","_inner_sum_d1_XX","_inner_sum_d1_V_XX"]].copy()
                        _dbg = _dbg.sort_values("ID_XX")
                        print(f"    --- Per-ID data for p={pairwise} plac={placebo_index} ---")
                        for _, r in _dbg.iterrows():
                            print(f"    ID={int(r['ID_XX']):3d} D1={r['D1_XX']:8.2f} dY={r['delta_Y_XX']:10.6f} dD={r['delta_D_XX']:8.4f} S={r['S_XX']:4.0f} Sbis={r['S_bis_XX']:4.0f} Ht={r['Ht_XX']:2.0f} w={r['weight_XX']:6.2f} mp={r['mean_pred_XX']:10.6f} is_d1={r['_inner_sum_d1_XX']:12.8f} is_d1V={r['_inner_sum_d1_V_XX']:12.8f}")
            else:
                ps0_safe = df[ps0_col].replace(0, np.nan)
                df["dr_delta1_DR_XX"] = np.where(
                    df["S_bis_XX"] == 0,
                    -(df[sdd_cf_col] / ps0_safe) * df[inner_sum_col],
                    df["S_over_delta_D_XX"] * df[inner_sum_col])
                if cross_fitting > 0:
                    # Per-fold aggregation matching Stata:
                    # delta1_k = sum(dr * weight in fold k) / sum(S_bis * weight in fold k)
                    _dr = df["dr_delta1_DR_XX"].to_numpy(dtype=float)
                    _w = df["weight_XX"].to_numpy(dtype=float)
                    _sb = df["S_bis_XX"].to_numpy(dtype=float)
                    _cf = df["cf_sample_id"].to_numpy(dtype=float)
                    cf_sum_w = 0.0
                    delta1_num = 0.0
                    for cf_id in range(1, cross_fitting + 1):
                        fm = _cf == cf_id
                        fm_ok = fm & ~np.isnan(_dr)
                        sbv = _sb[fm_ok] * _w[fm_ok]
                        cf_N_k = float(np.sum(sbv))
                        cf_ES_k = float(np.mean(sbv)) if np.any(fm_ok) else 0.0
                        if cf_ES_k == 0:
                            cf_delta1_k = 0.0
                        else:
                            num_k = float(np.sum(_dr[fm_ok] * _w[fm_ok]))
                            cf_delta1_k = num_k / cf_N_k
                        cf_sum_w += cf_N_k
                        delta1_num += cf_N_k * cf_delta1_k
                    scalars[f"delta_1_{pairwise}{pl}_XX"] = delta1_num / cf_sum_w if cf_sum_w != 0 else 0.0
                    if os.environ.get("DMS_DEBUG_CF"):
                        _dr_ok = _dr[~np.isnan(_dr)]
                        _meanDR = float(np.mean(_dr_ok)) if len(_dr_ok) > 0 else 0.0
                        _Pt = scalars.get(f"P_{pairwise}{pl}_XX", 0.0)
                        print(f"  [CF-AS] p={pairwise} plac={placebo_index}: delta1={delta1_num/cf_sum_w if cf_sum_w!=0 else 0:.10f} Pt={_Pt:.8f} sumw={cf_sum_w:.2f} N={len(df)} sw={int(df['S_bis_XX'].sum())}")
                        # Ablation: compute delta1 with different nuisance param combos
                        if os.environ.get("DMS_DEBUG_CF") in ("ABLATION", "DETAIL"):
                            _fs_ps0 = df["PS_0_D_1_XX"].replace(0, np.nan).to_numpy(dtype=float)
                            _fs_is = df["inner_sum_delta_1_2_XX"].to_numpy(dtype=float)
                            _fs_sdd = df["mean_S_over_delta_D_XX"].to_numpy(dtype=float)
                            _cf_ps0_v = df["cf_PS_0_D_1_XX"].replace(0, np.nan).to_numpy(dtype=float)
                            _cf_is_v = df["cf_inner_sum_delta_1_2_XX"].to_numpy(dtype=float)
                            _cf_sdd_v = df["cf_mean_S_over_delta_D_XX"].to_numpy(dtype=float)
                            _sbis = df["S_bis_XX"].to_numpy(dtype=float)
                            _sodd = df["S_over_delta_D_XX"].to_numpy(dtype=float)
                            _combos = [
                                ("all_CF   ", _cf_sdd_v, _cf_ps0_v, _cf_is_v),
                                ("full_p   ", _cf_sdd_v, _fs_ps0, _cf_is_v),
                                ("full_g   ", _fs_sdd, _cf_ps0_v, _cf_is_v),
                                ("full_mu  ", _cf_sdd_v, _cf_ps0_v, _fs_is),
                                ("full_p_g ", _fs_sdd, _fs_ps0, _cf_is_v),
                                ("full_p_mu", _cf_sdd_v, _fs_ps0, _fs_is),
                                ("full_g_mu", _fs_sdd, _cf_ps0_v, _fs_is),
                                ("all_FULL ", _fs_sdd, _fs_ps0, _fs_is),
                            ]
                            for _cname, _csdd, _cps0, _cis in _combos:
                                _drt = np.where(_sbis == 0, -(_csdd / _cps0) * _cis, _sodd * _cis)
                                _drt = _drt.astype(float)
                                _asw = 0.0
                                _asn = 0.0
                                for cf_id in range(1, cross_fitting + 1):
                                    fm = _cf == cf_id
                                    fm_ok = fm & ~np.isnan(_drt)
                                    _asbv = _sb[fm_ok] * _w[fm_ok]
                                    _aNk = float(np.sum(_asbv))
                                    _aESk = float(np.mean(_asbv)) if np.any(fm_ok) else 0.0
                                    if _aESk == 0:
                                        _adk = 0.0
                                    else:
                                        _adk = float(np.sum(_drt[fm_ok] * _w[fm_ok])) / _aNk
                                    _asw += _aNk
                                    _asn += _aNk * _adk
                                _ad1 = _asn / _asw if _asw != 0 else 0.0
                                print(f"    ABLATION p={pairwise} {_cname}: delta1={_ad1:.10f}")
                        if os.environ.get("DMS_DEBUG_CF") == "DETAIL" and str(pairwise) == os.environ.get("DMS_DEBUG_PAIR", "3"):
                            print(f"    --- Per-fold AS detail for pair {pairwise} ---")
                            for cf_id in range(1, cross_fitting + 1):
                                fm = _cf == cf_id
                                fm_ok = fm & ~np.isnan(_dr)
                                _n_fold = int(np.sum(fm))
                                _n_ok = int(np.sum(fm_ok))
                                _sbv = _sb[fm_ok] * _w[fm_ok]
                                _nk = float(np.sum(_sbv))
                                _esk = float(np.mean(_sbv)) if _n_ok > 0 else 0.0
                                if _esk == 0:
                                    _dk = 0.0
                                    _numk = 0.0
                                else:
                                    _numk = float(np.sum(_dr[fm_ok] * _w[fm_ok]))
                                    _dk = _numk / _nk
                                print(f"      fold {cf_id}: N={_n_fold} ok={_n_ok} Nk={_nk:.1f} ESk={_esk:.4f} numk={_numk:.8f} dk={_dk:.10f}")
                            # Also dump per-obs data
                            print("    --- Per-obs data (first 20 obs) ---")
                            _dbg_df = df[["ID_XX","D1_XX","S_bis_XX","cf_sample_id",
                                          inner_sum_col, ps0_col, sdd_cf_col,
                                          "inner_sum_delta_1_2_XX","PS_0_D_1_XX","mean_S_over_delta_D_XX",
                                          "S_over_delta_D_XX","dr_delta1_DR_XX","weight_XX"]].head(20)
                            for _, r in _dbg_df.iterrows():
                                print(f"      ID={int(r['ID_XX']):3d} D1={r['D1_XX']:.4f} S={r['S_bis_XX']:.0f} fold={int(r['cf_sample_id'])} "
                                      f"cf_is={r[inner_sum_col]:.8f} cf_ps0={r[ps0_col]:.8f} cf_sdd={r[sdd_cf_col]:.8f} "
                                      f"fs_is={r['inner_sum_delta_1_2_XX']:.8f} fs_ps0={r['PS_0_D_1_XX']:.8f} fs_sdd={r['mean_S_over_delta_D_XX']:.8f} "
                                      f"DR={r['dr_delta1_DR_XX']:.8f}")
                else:
                    scalars[f"delta_1_{pairwise}{pl}_XX"] = Mean("dr_delta1_DR_XX", df)
                    if os.environ.get("DMS_DEBUG_CF"):
                        print(f"  [noCF-AS] p={pairwise} plac={placebo_index}: delta1={scalars[f'delta_1_{pairwise}{pl}_XX']:.10f} N={len(df)}")

            df["inner_sum_delta_1_XX"] = df["inner_sum_delta_1_2_XX"] / df["delta_D_XX"]
            df.loc[df["delta_D_XX"] == 0, "inner_sum_delta_1_XX"] = np.nan

            # Stata: Phi_1 always uses full-sample (non-CF) nuisance params
            phi1_ps0_col = "PS_0_D_1_XX" if cross_fitting > 0 else ps0_col
            phi1_sdd_col = "mean_S_over_delta_D_XX" if cross_fitting > 0 else sdd_cf_col
            phi1_inner_col = "inner_sum_delta_1_2_XX" if cross_fitting > 0 else inner_sum_col
            phi1_esbis_col = "ES_bis_XX_D_1" if cross_fitting > 0 else esbis_col
            if not exact_match:
                ps0_safe = df[phi1_ps0_col].replace(0, np.nan)
                # Split to avoid NaN*0=NaN for switchers when ps0_safe is NaN
                adj = np.where(df["S_bis_XX"] == 0, 1.0 / ps0_safe, 0.0)
                # Stata: gen Phi1 = weights_XX * (...) * inner_sum
                raw_phi = df["weight_XX"] * (df["S_over_delta_D_XX"] - df[phi1_sdd_col] * adj) * df[phi1_inner_col]
            else:
                denom_exact = (1.0 - df[phi1_esbis_col]).replace(0, np.nan)
                adj = np.where(df["S_bis_XX"] == 0, 1.0 / denom_exact, 0.0)
                # Stata: gen Phi1 = weights_XX * (...) * inner_sum
                raw_phi = df["weight_XX"] * (df["S_over_delta_D_XX"] - df[phi1_sdd_col] * adj) * df[phi1_inner_col]

            denom_phi = ES * scalars[f"P_Ht_{pairwise}{pl}_XX"]
            # Stata: replace Phi1 = [Phi1 - delta1 * SbisV_XX] / [ES * PHt]
            df[f"Phi_1_{pairwise}{pl}_XX"] = (
                raw_phi - scalars[f"delta_1_{pairwise}{pl}_XX"] * df["SbisV_XX"]
            ) / denom_phi if denom_phi != 0 else np.nan
            df.loc[df["Ht_XX"] == 0, f"Phi_1_{pairwise}{pl}_XX"] = 0.0
            # Stata line 3903: replace Phi1 = . if S_XX==.
            df.loc[df["S_XX"].isna(), f"Phi_1_{pairwise}{pl}_XX"] = np.nan

            # SE — Stata line 3905/3915: sum Phi1 (unweighted), sd_delta1 = r(sd)/sqrt(r(N))
            phi_col = f"Phi_1_{pairwise}{pl}_XX"
            if cluster_col is not None:
                df["_phi_c"] = df.groupby(cluster_col)[phi_col].transform(
                    lambda s: float(np.nansum(s.to_numpy(dtype=float))))
                df["_first_clus"] = df.groupby(cluster_col).cumcount().eq(0)
                df["_phi_c"] = np.where(df["_first_clus"], df["_phi_c"], np.nan) / scalars[f"N_bar_c_{pairwise}{pl}_XX"]
                # Stata line 3912: sum Phi1_cXX [iw=weights_cXX]; line 3915: r(sd)/sqrt(r(N))
                # r(N) = count of non-missing clusters, NOT sum of weights
                n_c = int(df["_phi_c"].notna().sum())
                scalars[f"sd_delta_1_{pairwise}{pl}_XX"] = Sd("_phi_c", df, w="weight_c_XX") / np.sqrt(n_c) if n_c > 0 else np.nan
                df.drop(columns=["_phi_c", "_first_clus"], inplace=True)
            else:
                # Stata line 3905: sum Phi1 (unweighted), then r(sd)/sqrt(r(N))
                _phi_arr = df[phi_col].to_numpy(dtype=float)
                _phi_arr = _phi_arr[~np.isnan(_phi_arr)]
                n_phi = len(_phi_arr)
                sd_phi = float(np.std(_phi_arr, ddof=1)) if n_phi > 1 else 0.0
                scalars[f"sd_delta_1_{pairwise}{pl}_XX"] = sd_phi / np.sqrt(n_phi) if n_phi > 0 else np.nan

            se = scalars[f"sd_delta_1_{pairwise}{pl}_XX"]
            scalars[f"LB_1_{pairwise}{pl}_XX"] = scalars[f"delta_1_{pairwise}{pl}_XX"] - 1.96 * se
            scalars[f"UB_1_{pairwise}{pl}_XX"] = scalars[f"delta_1_{pairwise}{pl}_XX"] + 1.96 * se
            # Stata: gen S_pairwise = SbisV_XX; replace = 0 if Ht_XX==0
            df[f"S_{pairwise}{pl}_XX"] = df["SbisV_XX"]
            df.loc[df["Ht_XX"] == 0, f"S_{pairwise}{pl}_XX"] = 0.0

        # --- 18C) WAOSS ---
        if waoss == 1:
            # Stata: gen absdeltaDV = absdeltaD * weights; sum absdeltaDV; EabsdeltaD = r(mean)
            # = unweighted mean of (|deltaD| * V). For uniform V this equals Mean(|deltaD|).
            df["absdeltaDV_XX"] = df["abs_delta_D_XX"] * df["weight_XX"]
            _adv = df["absdeltaDV_XX"].to_numpy(dtype=float)
            _adv_nm = _adv[~np.isnan(_adv)]
            scalars[f"E_abs_delta_D{pl}_XX"] = float(np.mean(_adv_nm)) if len(_adv_nm) > 0 else 0.0
            scalars[f"E_abs_delta_D_{pairwise}{pl}_XX"] = scalars[f"E_abs_delta_D{pl}_XX"] * scalars[f"P_Ht_{pairwise}{pl}_XX"]
            scalars[f"E_abs_delta_D_sum{pl}_XX"] = scalars.get(f"E_abs_delta_D_sum{pl}_XX", 0.0) + scalars[f"E_abs_delta_D_{pairwise}{pl}_XX"]

            for suffix in ("Minus", "Plus"):
                target_S = 1.0 if suffix == "Plus" else -1.0
                df["Ster_XX"] = np.where(df["S_XX"].isna(), np.nan, (df["S_XX"] == target_S).astype(float))
                df["prod_sgn_delta_D_delta_D_XX"] = df["S_XX"] * df["delta_D_XX"]
                sum_prod = Sum("prod_sgn_delta_D_delta_D_XX", df[df["Ster_XX"] == 1])
                scalars[f"w_{suffix}_{pairwise}{pl}_XX"] = sum_prod / scalars[f"N{pl}_XX"] if scalars[f"N{pl}_XX"] > 0 else 0.0

                denom = Sum("delta_D_XX", df[df["Ster_XX"] == 1])
                scalars[f"denom_delta_2_{suffix}_{pairwise}{pl}_XX"] = denom

                if estimation_method == "ra":
                    if denom == 0:
                        denom = 1.0
                    num = Sum("inner_sum_delta_1_2_XX", df[df["Ster_XX"] == 1])
                    scalars[f"delta_2_{suffix}_{pairwise}{pl}_XX"] = num / denom

                nb_sw = float(df[df["Ster_XX"] == 1].shape[0])
                scalars[f"nb_Switchers_{suffix}{pl}_XX"] = nb_sw
                scalars[f"PS_{suffix}1{pl}_XX"] = nb_sw / scalars[f"N{pl}_XX"] if scalars[f"N{pl}_XX"] > 0 else 0.0

                logit_pol = logit_Plus_pol if suffix == "Plus" else logit_Minus_pol

                if not exact_match:
                    if scalars[f"PS_{suffix}1{pl}_XX"] == 0:
                        scalars[f"delta_2_{suffix}_{pairwise}{pl}_XX"] = 0.0
                        df[f"PS_1_{suffix}_D_1_XX"] = 0.0
                    else:
                        ps1_formula = f"Ster_XX ~ {logit_pol}"
                        ps1_model = None
                        try:
                            ps1_model = _make_logit(asinstata, model_stayer).fit(ps1_formula, df)
                            df = lpredict(df, f"PS_1_{suffix}_D_1_XX", ps1_model, prob=True)
                        except Exception:
                            df[f"PS_1_{suffix}_D_1_XX"] = scalars[f"PS_{suffix}1{pl}_XX"]

                        # Cross-fitting for P(S+/S-|D1)
                        if cross_fitting > 0:
                            # Pass full-sample model as fallback: Stata's capture+predict
                            # reuses last e(b) when logit fails ("outcome does not vary").
                            df = _cf_regression(df, ps1_formula, "Ster_XX", f"PS_1_{suffix}_D_1_XX",
                                                use_logit=True, fallback_model=ps1_model)

                        if estimation_method == "ps":
                            df[f"delta_Y_P_{suffix}_XX"] = (
                                df["delta_Y_XX"]
                                * (df[f"PS_1_{suffix}_D_1_XX"] / df[ps0_col].replace(0, np.nan))
                                * (scalars[f"PS_0{pl}_XX"] / scalars[f"PS_{suffix}1{pl}_XX"]))
                            mean_delta_Y_P = Mean(f"delta_Y_P_{suffix}_XX", df[df["S_XX"] == 0])
                            mean_delta_Y = Mean("delta_Y_XX", df[df["Ster_XX"] == 1])
                            mean_delta_D = Mean("delta_D_XX", df[df["Ster_XX"] == 1])
                            scalars[f"delta_2_{suffix}_{pairwise}{pl}_XX"] = (mean_delta_Y - mean_delta_Y_P) / mean_delta_D if mean_delta_D != 0 else 0.0

            if estimation_method in ("ra", "ps"):
                w_plus = scalars.get(f"w_Plus_{pairwise}{pl}_XX", 0.0)
                w_minus = scalars.get(f"w_Minus_{pairwise}{pl}_XX", 0.0)
                denomw = w_plus + w_minus
                scalars[f"W_Plus_{pairwise}{pl}_XX"] = (w_plus / denomw) if denomw != 0 else 0.0

            # Choose PS columns for CF or full
            ps1_plus_col = "cf_PS_1_Plus_D_1_XX" if (cross_fitting > 0 and "cf_PS_1_Plus_D_1_XX" in df.columns) else "PS_1_Plus_D_1_XX"
            ps1_minus_col = "cf_PS_1_Minus_D_1_XX" if (cross_fitting > 0 and "cf_PS_1_Minus_D_1_XX" in df.columns) else "PS_1_Minus_D_1_XX"

            if not exact_match:
                ps_plus = df.get(ps1_plus_col, pd.Series(0.0, index=df.index))
                ps_minus = df.get(ps1_minus_col, pd.Series(0.0, index=df.index))
                ps0_safe = df[ps0_col].replace(0, np.nan)
                # Split stayer/switcher computation to avoid NaN*0=NaN when ps0_safe is NaN
                # Stata does: gen dr=... if Sbis==0 / replace dr=... if Sbis==1
                dr_stayer = -((ps_plus - ps_minus) / ps0_safe) * df[inner_sum_col]
                dr_switcher = df["S_XX"] * df[inner_sum_col]
                df["dr_delta_Y_XX"] = np.where(
                    df["S_bis_XX"] == 0, dr_stayer, dr_switcher)
                df.loc[df["S_bis_XX"].isna(), "dr_delta_Y_XX"] = np.nan
                scalars[f"denom_dr_delta_2{pl}_XX"] = Sum("dr_delta_Y_XX", df)

            if estimation_method in ("ra", "ps"):
                Wp = scalars[f"W_Plus_{pairwise}{pl}_XX"]
                scalars[f"delta_2_{pairwise}{pl}_XX"] = (
                    Wp * scalars.get(f"delta_2_Plus_{pairwise}{pl}_XX", 0.0)
                    + (1.0 - Wp) * scalars.get(f"delta_2_Minus_{pairwise}{pl}_XX", 0.0))
            elif estimation_method == "dr":
                if cross_fitting > 0:
                    # Per-fold aggregation matching Stata:
                    # For each fold, compute num_k / denom_k.
                    # Skip folds where denom_k == 0 (no switchers).
                    _dr = df["dr_delta_Y_XX"].to_numpy(dtype=float)
                    _w = df["weight_XX"].to_numpy(dtype=float)
                    _a = df["abs_delta_D_XX"].to_numpy(dtype=float)
                    _cf = df["cf_sample_id"].to_numpy(dtype=float)
                    cf_sum_w = 0.0
                    delta2_num = 0.0
                    for cf_id in range(1, cross_fitting + 1):
                        fm = _cf == cf_id
                        fm_ok = fm & ~np.isnan(_dr)
                        num_k = float(np.sum(_dr[fm_ok] * _w[fm_ok])) if np.any(fm_ok) else 0.0
                        denom_k = float(np.sum(_a[fm_ok] * _w[fm_ok])) if np.any(fm_ok) else 0.0
                        if denom_k != 0:
                            cf_sum_w += denom_k
                            delta2_num += num_k
                    scalars[f"delta_2_{pairwise}{pl}_XX"] = delta2_num / cf_sum_w if cf_sum_w != 0 else 0.0
                    if os.environ.get("DMS_DEBUG_CF"):
                        print(f"  [CF-WAS] p={pairwise} plac={placebo_index}: delta2={delta2_num/cf_sum_w if cf_sum_w!=0 else 0:.10f} num={delta2_num:.8f} sumw={cf_sum_w:.2f}")
                        if os.environ.get("DMS_DEBUG_CF") in ("ABLATION", "DETAIL"):
                            _sbis = df["S_bis_XX"].to_numpy(dtype=float)
                            _sxx = df["S_XX"].to_numpy(dtype=float)
                            _fs_ps_plus = df.get("PS_1_Plus_D_1_XX", pd.Series(0.0, index=df.index)).to_numpy(dtype=float)
                            _fs_ps_minus = df.get("PS_1_Minus_D_1_XX", pd.Series(0.0, index=df.index)).to_numpy(dtype=float)
                            _fs_ps0 = df["PS_0_D_1_XX"].to_numpy(dtype=float)
                            _fs_ps0[_fs_ps0 == 0] = np.nan
                            _fs_inner = df["inner_sum_delta_1_2_XX"].to_numpy(dtype=float)
                            _cf_ps_plus = df.get("cf_PS_1_Plus_D_1_XX", pd.Series(0.0, index=df.index)).to_numpy(dtype=float)
                            _cf_ps_minus = df.get("cf_PS_1_Minus_D_1_XX", pd.Series(0.0, index=df.index)).to_numpy(dtype=float)
                            _cf_ps0 = df["cf_PS_0_D_1_XX"].to_numpy(dtype=float)
                            _cf_ps0[_cf_ps0 == 0] = np.nan
                            _cf_inner = df["cf_inner_sum_delta_1_2_XX"].to_numpy(dtype=float)
                            _combos_was = [
                                ("all_CF    ", _cf_ps_plus, _cf_ps_minus, _cf_ps0, _cf_inner),
                                ("full_PS1  ", _fs_ps_plus, _fs_ps_minus, _cf_ps0, _cf_inner),
                                ("full_PS0  ", _cf_ps_plus, _cf_ps_minus, _fs_ps0, _cf_inner),
                                ("full_inner", _cf_ps_plus, _cf_ps_minus, _cf_ps0, _fs_inner),
                                ("full_PS   ", _fs_ps_plus, _fs_ps_minus, _fs_ps0, _cf_inner),
                                ("full_PS+in", _fs_ps_plus, _fs_ps_minus, _fs_ps0, _fs_inner),
                                ("full_P0+in", _cf_ps_plus, _cf_ps_minus, _fs_ps0, _fs_inner),
                                ("all_FULL  ", _fs_ps_plus, _fs_ps_minus, _fs_ps0, _fs_inner),
                            ]
                            for _cname, _cpp, _cpm, _cp0, _cin in _combos_was:
                                _drt_st = -((_cpp - _cpm) / _cp0) * _cin
                                _drt_sw = _sxx * _cin
                                _drt = np.where(_sbis == 0, _drt_st, _drt_sw)
                                _drt = np.where(np.isnan(_sbis), np.nan, _drt)
                                _asw2 = 0.0
                                _asn2 = 0.0
                                for cf_id in range(1, cross_fitting + 1):
                                    fm = _cf == cf_id
                                    fm_ok = fm & ~np.isnan(_drt)
                                    _ank = float(np.sum(_a[fm_ok] * _w[fm_ok])) if np.any(fm_ok) else 0.0
                                    _annk = float(np.sum(_drt[fm_ok] * _w[fm_ok])) if np.any(fm_ok) else 0.0
                                    if _ank != 0:
                                        _asw2 += _ank
                                        _asn2 += _annk
                                _ad2 = _asn2 / _asw2 if _asw2 != 0 else 0.0
                                print(f"    ABLATION-WAS p={pairwise} {_cname}: delta2={_ad2:.10f}")
                else:
                    sum_abs = Sum("abs_delta_D_XX", df)
                    scalars[f"delta_2_{pairwise}{pl}_XX"] = scalars[f"denom_dr_delta_2{pl}_XX"] / sum_abs if sum_abs != 0 else 0.0

            if not exact_match:
                # Stata: Phi_2 always uses full-sample (non-CF) nuisance params
                # Stata line 4142: gen Phi2 = weights_XX*(dr_deltaYV_XX - delta2*absdeltaDV_XX)
                # dr_deltaYV_XX and absdeltaDV_XX already include weights_XX,
                # but SE uses unweighted sd(Phi2)/sqrt(N). To match Stata's
                # numerical outcome, keep w^1 here + unweighted SE below.
                if cross_fitting > 0:
                    ps_plus_fs = df.get("PS_1_Plus_D_1_XX", pd.Series(0.0, index=df.index))
                    ps_minus_fs = df.get("PS_1_Minus_D_1_XX", pd.Series(0.0, index=df.index))
                    ps0_fs_safe = df["PS_0_D_1_XX"].replace(0, np.nan)
                    dr_stayer_fs = -((ps_plus_fs - ps_minus_fs) / ps0_fs_safe) * df["inner_sum_delta_1_2_XX"]
                    dr_switcher_fs = df["S_XX"] * df["inner_sum_delta_1_2_XX"]
                    dr_fs = np.where(df["S_bis_XX"] == 0, dr_stayer_fs, dr_switcher_fs)
                    dr_fs = np.where(df["S_bis_XX"].isna(), np.nan, dr_fs)
                    df[f"Phi_2_{pairwise}{pl}_XX"] = df["weight_XX"] * (dr_fs - scalars[f"delta_2_{pairwise}{pl}_XX"] * df["abs_delta_D_XX"])
                else:
                    df[f"Phi_2_{pairwise}{pl}_XX"] = df["weight_XX"] * (df["dr_delta_Y_XX"] - scalars[f"delta_2_{pairwise}{pl}_XX"] * df["abs_delta_D_XX"])
            else:
                # Stata: gen Phi2 = weights_XX * [(S - ES*adj) * inner - delta2 * absdeltaD]
                denom_em = (1.0 - df[esbis_col]).replace(0, np.nan)
                phi2_adj = np.where(df["S_bis_XX"] == 0,
                    -df[es_col] / denom_em, df["S_XX"])
                df[f"Phi_2_{pairwise}{pl}_XX"] = df["weight_XX"] * (
                    phi2_adj * df[inner_sum_col]
                    - scalars[f"delta_2_{pairwise}{pl}_XX"] * df["abs_delta_D_XX"])

            denom_if = scalars[f"P_Ht_{pairwise}{pl}_XX"] * scalars[f"E_abs_delta_D{pl}_XX"]
            df[f"Phi_2_{pairwise}{pl}_XX"] = df[f"Phi_2_{pairwise}{pl}_XX"] / denom_if if denom_if != 0 else np.nan
            df.loc[df["Ht_XX"] == 0, f"Phi_2_{pairwise}{pl}_XX"] = 0.0
            # Stata line 4152: replace Phi2 = . if S_XX==.
            df.loc[df["S_XX"].isna(), f"Phi_2_{pairwise}{pl}_XX"] = np.nan

            # SE — Stata line 4165: sd_delta2 = r(sd)/sqrt(r(sum_w))
            # Stata uses UNWEIGHTED sd of Phi2 (which already has w^2 baked in)
            phi_col = f"Phi_2_{pairwise}{pl}_XX"
            if cluster_col is not None:
                df["_phi_c"] = df.groupby(cluster_col)[phi_col].transform(
                    lambda s: float(np.nansum(s.to_numpy(dtype=float))))
                df["_first_clus"] = df.groupby(cluster_col).cumcount().eq(0)
                df["_phi_c"] = np.where(df["_first_clus"], df["_phi_c"], np.nan) / scalars[f"N_bar_c_{pairwise}{pl}_XX"]
                # Stata line 4162: sum Phi2_cXX (unweighted), then r(sd)/sqrt(r(sum_w))
                _pc = df.loc[~df["_phi_c"].isna(), "_phi_c"].to_numpy(dtype=float)
                _pc = _pc[~np.isnan(_pc)]
                n_c = len(_pc)
                sd_c = float(np.std(_pc, ddof=1)) if n_c > 1 else 0.0
                scalars[f"sd_delta_2_{pairwise}{pl}_XX"] = sd_c / np.sqrt(n_c) if n_c > 0 else np.nan
                df.drop(columns=["_phi_c", "_first_clus"], inplace=True)
            else:
                # Stata line 4156: sum Phi2 (unweighted), then r(sd)/sqrt(r(sum_w))
                _phi_arr = df[phi_col].to_numpy(dtype=float)
                _phi_arr = _phi_arr[~np.isnan(_phi_arr)]
                n_phi = len(_phi_arr)
                sd_phi = float(np.std(_phi_arr, ddof=1)) if n_phi > 1 else 0.0
                scalars[f"sd_delta_2_{pairwise}{pl}_XX"] = sd_phi / np.sqrt(n_phi) if n_phi > 0 else np.nan

            se = scalars[f"sd_delta_2_{pairwise}{pl}_XX"]
            scalars[f"LB_2_{pairwise}{pl}_XX"] = scalars[f"delta_2_{pairwise}{pl}_XX"] - 1.96 * se
            scalars[f"UB_2_{pairwise}{pl}_XX"] = scalars[f"delta_2_{pairwise}{pl}_XX"] + 1.96 * se
            # Stata: gen absdeltaD_pairwise = absdeltaD_XX; replace = 0 if Ht==0; replace = absdeltaD * weights
            df[f"abs_delta_D_{pairwise}{pl}_XX"] = np.where(df["Ht_XX"] == 0, 0.0, df["abs_delta_D_XX"] * df["weight_XX"])

        # --- 18D) IV-WAOSS ---
        if ivwaoss == 1:
            scalars[f"E_abs_delta_Z{pl}_XX"] = Mean("abs_delta_Z_XX", df)
            df["SI_bis_XX"] = ((df["SI_XX"] != 0) & (~df["SI_XX"].isna())).astype(float)
            df["SI_Plus_XX"] = np.where(df["SI_XX"].isna(), np.nan, (df["SI_XX"] == 1).astype(float))
            df["SI_Minus_XX"] = np.where(df["SI_XX"].isna(), np.nan, (df["SI_XX"] == -1).astype(float))
            df["S_IV_0_XX"] = 1.0 - df["SI_bis_XX"]

            if not exact_match:
                psiv0_formula = f"S_IV_0_XX ~ {IV_logit_bis_pol}"
                try:
                    psiv0_model = _make_logit(asinstata, model_stayer).fit(psiv0_formula, df)
                    df = lpredict(df, "PS_IV_0_Z_1_XX", psiv0_model, prob=True)
                except Exception:
                    df["PS_IV_0_Z_1_XX"] = 0.5
                # PS bounding (Stata: replace PS_IV0Z1_XX=0 if <=10^(-10))
                df.loc[df["PS_IV_0_Z_1_XX"] <= 1e-10, "PS_IV_0_Z_1_XX"] = 0.0
            else:
                # exact_match: group means by Z1 (equivalent to saturated polynomial)
                _gm_sibis = df.groupby("Z1_XX")["SI_bis_XX"].mean()
                df["ES_I_bis_XX_Z_1"] = df["Z1_XX"].map(_gm_sibis).astype(float)
                _gm_si = df.groupby("Z1_XX")["SI_XX"].mean()
                df["ES_I_XX_Z_1"] = df["Z1_XX"].map(_gm_si).astype(float)

            scalars[f"PS_IV_0{pl}_XX"] = Mean("S_IV_0_XX", df)

            for suffix in ("Minus", "Plus"):
                flag = "SI_Minus_XX" if suffix == "Minus" else "SI_Plus_XX"
                nb = float((df[flag] == 1).sum())
                scalars[f"nb_Switchers_I_{suffix}{pl}_XX"] = nb
                scalars[f"PS_I_{suffix}_1{pl}_XX"] = nb / scalars[f"N{pl}_XX"] if scalars[f"N{pl}_XX"] > 0 else 0.0
                if scalars[f"PS_I_{suffix}_1{pl}_XX"] == 0:
                    df[f"PS_I_{suffix}_1_Z_1_XX"] = 0.0
                else:
                    if not exact_match:
                        iv_logit_pol = IV_logit_Plus_pol if suffix == "Plus" else IV_logit_Minus_pol
                        psis_formula = f"{flag} ~ {iv_logit_pol}"
                        try:
                            psis_model = _make_logit(asinstata, model_stayer).fit(psis_formula, df)
                            df = lpredict(df, f"PS_I_{suffix}_1_Z_1_XX", psis_model, prob=True)
                        except Exception:
                            df[f"PS_I_{suffix}_1_Z_1_XX"] = scalars[f"PS_I_{suffix}_1{pl}_XX"]

            df["prod_sgn_delta_Z_delta_Y_XX"] = df["SI_XX"] * df["delta_Y_XX"]
            df["prod_sgn_delta_Z_delta_D_XX"] = df["SI_XX"] * df["delta_D_XX"]

            df_temp = df[df["SI_XX"] == 0].copy()
            mY_formula = f"delta_Y_XX ~ {IV_reg_pol_terms}"
            try:
                if exact_match:
                    _ones_t = pd.Series(1.0, index=df_temp.index)
                    mY_model = _svd_wls(mY_formula, df_temp, _ones_t, rcond=1e-7, use_float32=True)
                else:
                    # Stata: reg deltaY_XX ... if SI_XX==0  (NO weights)
                    mY_model = _make_ols(asinstata, model_deltay).fit(mY_formula, df_temp)
                df = lpredict(df, "mean_delta_Y_pred_IV_XX", mY_model)
            except Exception:
                df["mean_delta_Y_pred_IV_XX"] = 0.0
            df["inner_sum_IV_num_XX"] = df["delta_Y_XX"] - df["mean_delta_Y_pred_IV_XX"]

            mD_formula = f"delta_D_XX ~ {IV_reg_pol_terms}"
            try:
                if exact_match:
                    _ones_t = pd.Series(1.0, index=df_temp.index)
                    mD_model = _svd_wls(mD_formula, df_temp, _ones_t, rcond=1e-7, use_float32=True)
                else:
                    # Stata: reg deltaD_XX ... if SI_XX==0  (NO weights)
                    mD_model = _make_ols(asinstata, model_deltay).fit(mD_formula, df_temp)
                df = lpredict(df, "mean_delta_D_pred_IV_XX", mD_model)
            except Exception:
                df["mean_delta_D_pred_IV_XX"] = 0.0
            df["inner_sum_IV_denom_XX"] = df["delta_D_XX"] - df["mean_delta_D_pred_IV_XX"]

            if estimation_method == "ra":
                df["inner_sum_IV_num_XX"] = df["inner_sum_IV_num_XX"] * df["SI_XX"]
                df["inner_sum_IV_denom_XX"] = df["inner_sum_IV_denom_XX"] * df["SI_XX"]
                N_total = float(scalars[f"N{pl}_XX"])
                if N_total > 0:
                    scalars[f"num_delta_IV_{pairwise}{pl}_XX"] = Sum("inner_sum_IV_num_XX", df) / N_total
                    scalars[f"denom_delta_IV_{pairwise}{pl}_XX"] = Sum("inner_sum_IV_denom_XX", df) / N_total
                else:
                    scalars[f"num_delta_IV_{pairwise}{pl}_XX"] = np.nan
                    scalars[f"denom_delta_IV_{pairwise}{pl}_XX"] = np.nan

            if estimation_method == "ps":
                df["delta_Y_P_IV_XX"] = (
                    df["delta_Y_XX"]
                    * ((df.get("PS_I_Plus_1_Z_1_XX", pd.Series(0.0, index=df.index)) - df.get("PS_I_Minus_1_Z_1_XX", pd.Series(0.0, index=df.index))) / df["PS_IV_0_Z_1_XX"].replace(0, np.nan))
                    * scalars[f"PS_IV_0{pl}_XX"])
                mean_delta_Y_P_IV = Mean("delta_Y_P_IV_XX", df[df["SI_bis_XX"] == 0])
                mean_prod_sgn_Z_delta_Y = Mean("prod_sgn_delta_Z_delta_Y_XX", df)
                scalars[f"num_delta_IV_{pairwise}{pl}_XX"] = mean_prod_sgn_Z_delta_Y - mean_delta_Y_P_IV

                df["delta_D_P_IV_XX"] = (
                    df["delta_D_XX"]
                    * ((df.get("PS_I_Plus_1_Z_1_XX", pd.Series(0.0, index=df.index)) - df.get("PS_I_Minus_1_Z_1_XX", pd.Series(0.0, index=df.index))) / df["PS_IV_0_Z_1_XX"].replace(0, np.nan))
                    * scalars[f"PS_IV_0{pl}_XX"])
                mean_delta_D_P_IV = Mean("delta_D_P_IV_XX", df[df["SI_bis_XX"] == 0])
                mean_prod_sgn_Z_delta_D = Mean("prod_sgn_delta_Z_delta_D_XX", df)
                scalars[f"denom_delta_IV_{pairwise}{pl}_XX"] = mean_prod_sgn_Z_delta_D - mean_delta_D_P_IV

            if estimation_method == "dr":
                ps_iv_plus = df.get("PS_I_Plus_1_Z_1_XX", pd.Series(0.0, index=df.index))
                ps_iv_minus = df.get("PS_I_Minus_1_Z_1_XX", pd.Series(0.0, index=df.index))
                ps_iv_0_safe = df["PS_IV_0_Z_1_XX"].replace(0, np.nan)
                dr_score = df["SI_XX"] - (ps_iv_plus - ps_iv_minus) / ps_iv_0_safe * (1.0 - df["SI_bis_XX"])
                df["dr_IV_delta_Y_XX"] = dr_score * df["inner_sum_IV_num_XX"]
                scalars[f"num_delta_IV_{pairwise}{pl}_XX"] = Sum("dr_IV_delta_Y_XX", df)
                df["dr_IV_delta_D_XX"] = dr_score * df["inner_sum_IV_denom_XX"]
                scalars[f"denom_delta_IV_{pairwise}{pl}_XX"] = Sum("dr_IV_delta_D_XX", df)

            denom_iv = scalars.get(f"denom_delta_IV_{pairwise}{pl}_XX", 0.0)
            scalars[f"delta_3_{pairwise}{pl}_XX"] = (
                scalars[f"num_delta_IV_{pairwise}{pl}_XX"] / denom_iv if denom_iv != 0 else np.nan)

            scalars[f"denom_delta_IV_sum{pl}_XX"] = scalars.get(f"denom_delta_IV_sum{pl}_XX", 0.0) + denom_iv

            scalars[f"delta_Y{pl}_XX"] = Mean("inner_sum_IV_num_XX", df)
            scalars[f"delta_D{pl}_XX"] = Mean("inner_sum_IV_denom_XX", df)

            df["resid_Y_IV_XX"] = df["delta_Y_XX"] - df["mean_delta_Y_pred_IV_XX"]
            df["resid_D_IV_XX"] = df["delta_D_XX"] - df["mean_delta_D_pred_IV_XX"]
            E_abs = scalars.get(f"E_abs_delta_Z{pl}_XX", np.nan)

            if not exact_match:
                denom_ps = df["PS_IV_0_Z_1_XX"].replace(0, np.nan)
                score = df["SI_XX"] - (df.get("PS_I_Plus_1_Z_1_XX", pd.Series(0.0, index=df.index)) - df.get("PS_I_Minus_1_Z_1_XX", pd.Series(0.0, index=df.index))) * (1.0 - df["SI_bis_XX"]) / denom_ps
            else:
                denom_es = (1.0 - df["ES_I_bis_XX_Z_1"]).replace(0, np.nan)
                score = df["SI_XX"] - df["ES_I_XX_Z_1"] * ((1.0 - df["SI_bis_XX"]) / denom_es)

            df["Phi_Y_XX"] = (score * df["resid_Y_IV_XX"] - scalars[f"delta_Y{pl}_XX"] * df["abs_delta_Z_XX"]) / E_abs if E_abs != 0 else np.nan
            df["Phi_D_XX"] = (score * df["resid_D_IV_XX"] - scalars[f"delta_D{pl}_XX"] * df["abs_delta_Z_XX"]) / E_abs if E_abs != 0 else np.nan

            delta_D_bar = scalars.get(f"delta_D{pl}_XX", np.nan)
            delta3 = scalars.get(f"delta_3_{pairwise}{pl}_XX", np.nan)

            if delta_D_bar is None or np.isnan(delta_D_bar) or delta_D_bar == 0:
                df[f"Phi_3_{pairwise}{pl}_XX"] = np.nan
                scalars[f"sd_delta_3_{pairwise}{pl}_XX"] = np.nan
            else:
                df[f"Phi_3_{pairwise}{pl}_XX"] = (df["Phi_Y_XX"] - delta3 * df["Phi_D_XX"]) / delta_D_bar

                phi_col = f"Phi_3_{pairwise}{pl}_XX"
                if cluster_col is not None:
                    df["_phi_c"] = df.groupby(cluster_col)[phi_col].transform(
                        lambda s: float(np.nansum(s.to_numpy(dtype=float))))
                    df["_first_clus"] = df.groupby(cluster_col).cumcount().eq(0)
                    df["_phi_c"] = np.where(df["_first_clus"], df["_phi_c"], np.nan) / scalars[f"N_bar_c_{pairwise}{pl}_XX"]
                    nobs_c = wSum(df[~df["_phi_c"].isna()], w="weight_c_XX")
                    scalars[f"sd_delta_3_{pairwise}{pl}_XX"] = Sd("_phi_c", df, w="weight_c_XX") / np.sqrt(nobs_c) if nobs_c > 0 else np.nan
                    df.drop(columns=["_phi_c", "_first_clus"], inplace=True)
                else:
                    scalars[f"sd_delta_3_{pairwise}{pl}_XX"] = Sd(f"Phi_3_{pairwise}{pl}_XX", df) / np.sqrt(wSum(df))

            se3 = scalars.get(f"sd_delta_3_{pairwise}{pl}_XX", np.nan)
            scalars[f"LB_3_{pairwise}{pl}_XX"] = delta3 - 1.96 * se3 if np.isfinite(se3) else np.nan
            scalars[f"UB_3_{pairwise}{pl}_XX"] = delta3 + 1.96 * se3 if np.isfinite(se3) else np.nan
            df[f"inner_sum_IV_denom_{pairwise}{pl}_XX"] = df["inner_sum_IV_denom_XX"]

        scalars[f"non_missing_{pairwise}{pl}_XX"] = 1.0

    else:
        # Not feasible — Stata sets delta to missing (.), not 0
        for i in (1, 2, 3):
            scalars[f"delta_{i}_{pairwise}{pl}_XX"] = np.nan
            scalars[f"sd_delta_{i}_{pairwise}{pl}_XX"] = np.nan
            df[f"Phi_{i}_{pairwise}{pl}_XX"] = np.nan
        if aoss == 1:
            scalars[f"P_{pairwise}{pl}_XX"] = np.nan
        if waoss == 1:
            scalars[f"E_abs_delta_D_{pairwise}{pl}_XX"] = np.nan
        if ivwaoss == 1:
            scalars[f"denom_delta_IV_{pairwise}{pl}_XX"] = np.nan
        scalars[f"non_missing_{pairwise}{pl}_XX"] = 0.0

    # --- 20) Prepare to_add ---
    df = df.sort_values("ID_XX").reset_index(drop=True)
    keep_cols = ["ID_XX",
                 f"Phi_1_{pairwise}{pl}_XX", f"Phi_2_{pairwise}{pl}_XX", f"Phi_3_{pairwise}{pl}_XX",
                 f"S_{pairwise}{pl}_XX", f"abs_delta_D_{pairwise}{pl}_XX",
                 f"used_in_{pairwise}{pl}_XX", f"inner_sum_IV_denom_{pairwise}{pl}_XX"]
    if cluster is not None and "cluster_XX" in df.columns:
        keep_cols.append("cluster_XX")
    keep_cols = [c for c in keep_cols if c in df.columns]
    out_df = df[keep_cols].copy()

    # --- 21) Final scalar bookkeeping ---
    if waoss == 1 or aoss == 1:
        scalars[f"N_Switchers_1_{pairwise}{pl}_XX"] = scalars.get(f"N_Switchers{pl}_XX", 0.0)
        scalars[f"N_Stayers_1_{pairwise}{pl}_XX"] = scalars.get(f"N_Stayers{pl}_XX", 0.0)
        scalars[f"N_Switchers_2_{pairwise}{pl}_XX"] = scalars.get(f"N_Switchers{pl}_XX", 0.0)
        scalars[f"N_Stayers_2_{pairwise}{pl}_XX"] = scalars.get(f"N_Stayers{pl}_XX", 0.0)
    if ivwaoss == 1:
        scalars[f"N_Switchers_3_{pairwise}{pl}_XX"] = scalars.get(f"N_Switchers_IV{pl}_XX", 0.0)
        scalars[f"N_Stayers_3_{pairwise}{pl}_XX"] = scalars.get(f"N_Stayers_IV{pl}_XX", 0.0)

    return {"scalars": scalars, "to_add": out_df}


# ============================================================
# MAIN AGGREGATION (with placebo N>1 support)
# ============================================================

def did_multiplegt_stat_main(
    df: pd.DataFrame,
    Y: str, ID: str, Time: str, D: str,
    Z: str | None,
    estimator: list[str],
    estimation_method: str,
    order: int,
    noextrapolation: bool,
    placebo: int,
    switchers: str | None,
    disaggregate: bool,
    aoss_vs_waoss: bool,
    exact_match: bool,
    weight: str | None,
    cluster: str | None,
    by_fd_opt: Any | None,
    other_treatments: list[str] | None,
    controls: list[str] | None = None,
    cross_fitting: int = 0,
    trimming: float = 0,
    on_placebo_sample: bool = False,
    order_reg: int | None = None,
    order_logit_bis: int | None = None,
    order_logit_Plus: int | None = None,
    order_logit_Minus: int | None = None,
    bootstrap: int = 0,
    twfe: bool = False,
    seed: int = 0,
    cross_validation_opt: dict[str, Any] | None = None,
    cf_folds_file: str | None = None,
    asinstata: bool = True,
    model_deltay=None,
    model_stayer=None,
) -> dict[str, Any]:
    """Main aggregation function."""

    aoss_XX = int("aoss" in estimator)
    waoss_XX = int("waoss" in estimator)
    ivwaoss_XX = int("ivwaoss" in estimator)

    # Select required columns
    varlist = []
    for v in [Y, ID, Time, D, Z, weight, cluster] + (other_treatments or []) + (controls or []):
        if v is not None and v not in varlist:
            varlist.append(v)
    if "partition_XX" in df.columns and "partition_XX" not in varlist:
        varlist.append("partition_XX")
    if "partition_lead_XX" in df.columns and "partition_lead_XX" not in varlist:
        varlist.append("partition_lead_XX")
    df = df[[c for c in varlist if c in df.columns]].copy()

    # Map to *_XX
    mapping = {"Y_XX": Y, "ID_XX": ID, "T_XX": Time, "D_XX": D}
    if Z is not None:
        mapping["Z_XX"] = Z
    if weight is not None:
        mapping["weight_XX"] = weight
    if cluster is not None:
        mapping["cluster_XX"] = cluster
    for new_col, old_col in mapping.items():
        if old_col is not None and old_col in df.columns:
            df[new_col] = df[old_col]

    # Check cluster
    n_clus_XX = None
    if cluster is not None:
        if cluster == ID:
            cluster = None
            df.drop(columns=["cluster_XX"], inplace=True, errors="ignore")
        else:
            g = df.groupby("ID_XX")["cluster_XX"].nunique(dropna=True)
            if (g > 1).any():
                raise ValueError("ID must be nested within cluster.")
            n_clus_XX = int(df["cluster_XX"].nunique(dropna=True))

    # Drop NAs
    df["to_drop_XX"] = df["T_XX"].isna() | df["D_XX"].isna() | df["ID_XX"].isna()
    if ivwaoss_XX == 1:
        df["to_drop_XX"] = df["to_drop_XX"] | df["Z_XX"].isna()
    df = df[~df["to_drop_XX"]].copy()

    # Balance panel
    df = _balance_panel_fill(df, id_col="ID_XX", t_col="T_XX")

    # Weights — weight_XX may have been lost during _balance_panel_fill (tsfill reindex)
    if weight is None:
        df["weight_XX"] = 1.0
        df["weight_c_XX"] = 1.0
    else:
        if "weight_XX" not in df.columns:
            # Recover from original weight column if available
            if weight in df.columns:
                df["weight_XX"] = df[weight]
            else:
                warnings.warn(f"Weight column '{weight}' not found in data; using uniform weights.", stacklevel=2)
                df["weight_XX"] = 1.0
        df["weight_XX"] = df["weight_XX"].fillna(0.0).astype(float)
        df["weight_c_XX"] = 1.0
    if cluster is not None:
        df["weight_c_XX"] = df.groupby(["cluster_XX", "T_XX"])["weight_XX"].transform("sum").astype(float)

    # on_placebo_sample: create full-panel first-difference for stayer classification
    # Stata lines 888-891: gen deltaDt_XX = D.D_XX; drop if deltaDt_XX==.
    if on_placebo_sample:
        df = df.sort_values(["ID_XX", "T_XX"])
        df["deltaDt_XX"] = df.groupby("ID_XX")["D_XX"].diff()
        df = df.dropna(subset=["deltaDt_XX"]).copy()

    # IDs dataframe
    IDs_XX = pd.DataFrame({"ID_XX": pd.Series(df["ID_XX"].unique()).sort_values().to_numpy()})
    if cluster is not None and "cluster_XX" in df.columns:
        IDs_XX = IDs_XX.merge(df.groupby("ID_XX")["cluster_XX"].first().reset_index(), on="ID_XX", how="left")

    max_T_XX = int(df["T_XX"].max())

    # Cross-validation (select orders before pairwise loop)
    if cross_validation_opt is not None and (waoss_XX == 1 or ivwaoss_XX == 1):
        cv_opts = cross_validation_opt
        cv_algorithm = cv_opts.get("algorithm", "kfolds")
        cv_tolerance = cv_opts.get("tolerance", 0.01)
        cv_max_k = cv_opts.get("max_k", 5)
        cv_seed = cv_opts.get("seed", 0)
        cv_kfolds = cv_opts.get("kfolds", 5)
        cv_same_order = cv_opts.get("same_order_all_logits", False)

        # Prepare CV data
        df_cv = df.copy()
        df_cv = df_cv.sort_values(["ID_XX", "T_XX"])
        df_cv["deltaYt_XX"] = df_cv.groupby("ID_XX")["Y_XX"].diff()
        df_cv["deltaDt_XX"] = df_cv.groupby("ID_XX")["D_XX"].diff()

        if ivwaoss_XX == 1:
            df_cv["deltaZt_XX"] = df_cv.groupby("ID_XX")["Z_XX"].diff()
            df_cv["SIbist_XX"] = (df_cv["deltaZt_XX"] != 0).astype(float)
            df_cv.loc[df_cv["deltaZt_XX"].isna(), "SIbist_XX"] = np.nan
            df_cv["SI0bist_XX"] = 1 - df_cv["SIbist_XX"]
        else:
            df_cv["Sbist_XX"] = (df_cv["deltaDt_XX"] != 0).astype(float)
            df_cv.loc[df_cv["deltaDt_XX"].isna(), "Sbist_XX"] = np.nan
            df_cv["S0bist_XX"] = 1 - df_cv["Sbist_XX"]

        is_fs = ivwaoss_XX == 1
        s0_col = "SI0bist_XX" if is_fs else "S0bist_XX"
        stayers_mask = df_cv[s0_col] == 1

        # 1) reg order
        order_reg = cross_validation_select(
            df_cv[stayers_mask], "deltaYt_XX", model_type="reg",
            algorithm=cv_algorithm, tolerance=cv_tolerance, max_k=cv_max_k,
            seed=cv_seed, kfolds=cv_kfolds, controls=controls,
            first_stage=is_fs, reduced_form=is_fs, asinstata=asinstata,
            model_deltay=model_deltay, model_stayer=model_stayer)

        # 2) logit_bis order
        order_logit_bis = cross_validation_select(
            df_cv, s0_col, model_type="logit",
            algorithm=cv_algorithm, tolerance=cv_tolerance, max_k=cv_max_k,
            seed=cv_seed, kfolds=cv_kfolds, controls=controls,
            first_stage=is_fs, reduced_form=is_fs, asinstata=asinstata,
            model_deltay=model_deltay, model_stayer=model_stayer)

        if cv_same_order:
            order_logit_Plus = order_logit_bis
            order_logit_Minus = order_logit_bis
        else:
            # 3) logit_Plus
            if ivwaoss_XX == 1:
                df_cv["StPlus_XX"] = (df_cv["deltaZt_XX"] > 0).astype(float)
                df_cv.loc[df_cv["deltaZt_XX"].isna(), "StPlus_XX"] = np.nan
            else:
                df_cv["StPlus_XX"] = (df_cv["deltaDt_XX"] > 0).astype(float)
                df_cv.loc[df_cv["deltaDt_XX"].isna(), "StPlus_XX"] = np.nan
            if df_cv["StPlus_XX"].sum() > 0 and switchers != "down":
                order_logit_Plus = cross_validation_select(
                    df_cv, "StPlus_XX", model_type="logit",
                    algorithm=cv_algorithm, tolerance=cv_tolerance, max_k=cv_max_k,
                    seed=cv_seed, kfolds=cv_kfolds, controls=controls,
                    first_stage=is_fs, reduced_form=is_fs, asinstata=asinstata,
            model_deltay=model_deltay, model_stayer=model_stayer)
            else:
                order_logit_Plus = 0

            # 4) logit_Minus
            if ivwaoss_XX == 1:
                df_cv["StMinus_XX"] = (df_cv["deltaZt_XX"] < 0).astype(float)
                df_cv.loc[df_cv["deltaZt_XX"].isna(), "StMinus_XX"] = np.nan
            else:
                df_cv["StMinus_XX"] = (df_cv["deltaDt_XX"] < 0).astype(float)
                df_cv.loc[df_cv["deltaDt_XX"].isna(), "StMinus_XX"] = np.nan
            if df_cv["StMinus_XX"].sum() > 0 and switchers != "up":
                order_logit_Minus = cross_validation_select(
                    df_cv, "StMinus_XX", model_type="logit",
                    algorithm=cv_algorithm, tolerance=cv_tolerance, max_k=cv_max_k,
                    seed=cv_seed, kfolds=cv_kfolds, controls=controls,
                    first_stage=is_fs, reduced_form=is_fs, asinstata=asinstata,
            model_deltay=model_deltay, model_stayer=model_stayer)
            else:
                order_logit_Minus = 0

        print(f"Cross-validation orders: reg={order_reg}, logit_bis={order_logit_bis}, "
              f"logit_Plus={order_logit_Plus}, logit_Minus={order_logit_Minus}")

    # Initialize scalars
    scalars: dict[str, Any] = dict(
        PS_sum_XX=0.0, delta_1_1_XX=0.0,
        E_abs_delta_D_sum_XX=0.0, delta_2_1_XX=0.0,
        denom_delta_IV_sum_XX=0.0, delta_3_1_XX=0.0,
        N_Switchers_1_1_XX=0.0, N_Stayers_1_1_XX=0.0,
        N_Switchers_2_1_XX=0.0, N_Stayers_2_1_XX=0.0,
        N_Switchers_3_1_XX=0.0, N_Stayers_3_1_XX=0.0,
        N_drop_total_XX=0.0, N_drop_total_C_XX=0.0,
        IV_req_XX=float(ivwaoss_XX),
    )

    # Pairwise args common dict
    pw_common = dict(
        Y="Y_XX", ID="ID_XX", Time="T_XX", D="D_XX",
        Z="Z_XX" if Z is not None else None,
        estimator=estimator, order=order,
        noextrapolation=noextrapolation,
        weight="weight_XX", switchers=switchers,
        aoss=aoss_XX, waoss=waoss_XX, ivwaoss=ivwaoss_XX,
        estimation_method=estimation_method,
        exact_match=exact_match, cluster=cluster,
        by_fd_opt=by_fd_opt, other_treatments=other_treatments,
        controls=controls, cross_fitting=cross_fitting,
        trimming=trimming, on_placebo_sample=on_placebo_sample,
        order_reg=order_reg, order_logit_bis=order_logit_bis,
        order_logit_Plus=order_logit_Plus, order_logit_Minus=order_logit_Minus,
        cf_folds_file=cf_folds_file,
        asinstata=asinstata,
        model_deltay=model_deltay,
        model_stayer=model_stayer,
    )

    # --- MAIN EFFECTS LOOP ---
    for p in range(2, max_T_XX + 1):
        est_out = did_multiplegt_stat_pairwise(
            df=df, **pw_common, pairwise=p, IDs=IDs_XX, scalars=scalars, placebo=0)
        to_add = est_out.get("to_add", None)
        scalars = est_out["scalars"]

        if to_add is not None and isinstance(to_add, pd.DataFrame) and len(to_add) > 0:
            if "cluster_XX" in IDs_XX.columns and "cluster_XX" in to_add.columns:
                to_add = to_add.drop(columns=["cluster_XX"])
            IDs_XX = IDs_XX.merge(to_add, on="ID_XX", how="left").sort_values("ID_XX").reset_index(drop=True)

        # Stata: aggregation is conditional on delta != . (infeasible pairs excluded)
        if aoss_XX == 1:
            _d1p = scalars.get(f"delta_1_{p}_XX", np.nan)
            if _d1p is not None and not (isinstance(_d1p, float) and np.isnan(_d1p)):
                scalars["delta_1_1_XX"] += scalars.get(f"P_{p}_XX", 0.0) * _d1p
                scalars["N_Switchers_1_1_XX"] += scalars.get(f"N_Switchers_1_{p}_XX", 0.0)
                scalars["N_Stayers_1_1_XX"] += scalars.get(f"N_Stayers_1_{p}_XX", 0.0)
        if waoss_XX == 1:
            _d2p = scalars.get(f"delta_2_{p}_XX", np.nan)
            if _d2p is not None and not (isinstance(_d2p, float) and np.isnan(_d2p)):
                scalars["delta_2_1_XX"] += scalars.get(f"E_abs_delta_D_{p}_XX", 0.0) * _d2p
                scalars["N_Switchers_2_1_XX"] += scalars.get(f"N_Switchers_2_{p}_XX", 0.0)
                scalars["N_Stayers_2_1_XX"] += scalars.get(f"N_Stayers_2_{p}_XX", 0.0)
        if ivwaoss_XX == 1:
            _d3p = scalars.get(f"delta_3_{p}_XX", np.nan)
            if _d3p is not None and not (isinstance(_d3p, float) and np.isnan(_d3p)):
                scalars["delta_3_1_XX"] += scalars.get(f"denom_delta_IV_{p}_XX", 0.0) * _d3p
                scalars["N_Switchers_3_1_XX"] += scalars.get(f"N_Switchers_3_{p}_XX", 0.0)
                scalars["N_Stayers_3_1_XX"] += scalars.get(f"N_Stayers_3_{p}_XX", 0.0)

    # --- PLACEBO LOOP (supports N>1) ---
    placebo_results = {}
    if placebo > 0:
        for placebo_index in range(1, placebo + 1):
            pl_scalars: dict[str, Any] = dict(
                PS_sum_pl_XX=0.0, delta_1_1_pl_XX=0.0,
                E_abs_delta_D_sum_pl_XX=0.0, delta_2_1_pl_XX=0.0,
                denom_delta_IV_sum_pl_XX=0.0, delta_3_1_pl_XX=0.0,
                N_Switchers_1_1_pl_XX=0.0, N_Stayers_1_1_pl_XX=0.0,
                N_Switchers_2_1_pl_XX=0.0, N_Stayers_2_1_pl_XX=0.0,
                N_Switchers_3_1_pl_XX=0.0, N_Stayers_3_1_pl_XX=0.0,
                N_drop_total_XX=0.0, N_drop_total_C_XX=0.0,
            )
            # Merge with main scalars for shared state
            for k, v in scalars.items():
                if k not in pl_scalars:
                    pl_scalars[k] = v

            IDs_pl = IDs_XX[["ID_XX"]].copy()
            if cluster is not None and "cluster_XX" in IDs_XX.columns:
                IDs_pl["cluster_XX"] = IDs_XX["cluster_XX"]

            for p in range(2 + placebo_index, max_T_XX + 1):
                est_out = did_multiplegt_stat_pairwise(
                    df=df, **pw_common, pairwise=p, IDs=IDs_pl,
                    scalars=pl_scalars, placebo=placebo_index)
                to_add = est_out.get("to_add", None)
                pl_scalars = est_out["scalars"]

                if to_add is not None and isinstance(to_add, pd.DataFrame) and len(to_add) > 0:
                    if "cluster_XX" in IDs_pl.columns and "cluster_XX" in to_add.columns:
                        to_add = to_add.drop(columns=["cluster_XX"])
                    IDs_pl = IDs_pl.merge(to_add, on="ID_XX", how="left").sort_values("ID_XX").reset_index(drop=True)

                # Stata: aggregation conditional on delta != . (infeasible pairs excluded)
                if aoss_XX == 1:
                    _d1p = pl_scalars.get(f"delta_1_{p}_pl_XX", np.nan)
                    if _d1p is not None and not (isinstance(_d1p, float) and np.isnan(_d1p)):
                        pl_scalars["delta_1_1_pl_XX"] += pl_scalars.get(f"P_{p}_pl_XX", 0.0) * _d1p
                        pl_scalars["N_Switchers_1_1_pl_XX"] += pl_scalars.get(f"N_Switchers_1_{p}_pl_XX", 0.0)
                        pl_scalars["N_Stayers_1_1_pl_XX"] += pl_scalars.get(f"N_Stayers_1_{p}_pl_XX", 0.0)
                if waoss_XX == 1:
                    _d2p = pl_scalars.get(f"delta_2_{p}_pl_XX", np.nan)
                    if _d2p is not None and not (isinstance(_d2p, float) and np.isnan(_d2p)):
                        pl_scalars["delta_2_1_pl_XX"] += pl_scalars.get(f"E_abs_delta_D_{p}_pl_XX", 0.0) * _d2p
                        pl_scalars["N_Switchers_2_1_pl_XX"] += pl_scalars.get(f"N_Switchers_2_{p}_pl_XX", 0.0)
                        pl_scalars["N_Stayers_2_1_pl_XX"] += pl_scalars.get(f"N_Stayers_2_{p}_pl_XX", 0.0)
                if ivwaoss_XX == 1:
                    _d3p = pl_scalars.get(f"delta_3_{p}_pl_XX", np.nan)
                    if _d3p is not None and not (isinstance(_d3p, float) and np.isnan(_d3p)):
                        pl_scalars["delta_3_1_pl_XX"] += pl_scalars.get(f"denom_delta_IV_{p}_pl_XX", 0.0) * _d3p
                        pl_scalars["N_Switchers_3_1_pl_XX"] += pl_scalars.get(f"N_Switchers_3_{p}_pl_XX", 0.0)
                        pl_scalars["N_Stayers_3_1_pl_XX"] += pl_scalars.get(f"N_Stayers_3_{p}_pl_XX", 0.0)

            # Divide by sums
            if aoss_XX == 1 and pl_scalars.get("PS_sum_pl_XX", 0.0) != 0:
                pl_scalars["delta_1_1_pl_XX"] /= pl_scalars["PS_sum_pl_XX"]
            if waoss_XX == 1 and pl_scalars.get("E_abs_delta_D_sum_pl_XX", 0.0) != 0:
                pl_scalars["delta_2_1_pl_XX"] /= pl_scalars["E_abs_delta_D_sum_pl_XX"]
            if ivwaoss_XX == 1 and pl_scalars.get("denom_delta_IV_sum_pl_XX", 0.0) != 0:
                pl_scalars["delta_3_1_pl_XX"] /= pl_scalars["denom_delta_IV_sum_pl_XX"]

            # Aggregate IFs for placebo (with rownonmiss like Stata)
            for i in (1, 2, 3):
                IDs_pl[f"Phi_{i}_pl_XX"] = 0.0
                IDs_pl[f"_nonmiss_{i}_pl_XX"] = 0

            for p in range(2 + placebo_index, max_T_XX + 1):
                non_missing = int(pl_scalars.get(f"non_missing_{p}_pl_XX", 0) == 1)
                if aoss_XX == 1 and non_missing == 1 and f"Phi_1_{p}_pl_XX" in IDs_pl.columns and f"S_{p}_pl_XX" in IDs_pl.columns:
                    Phi_p = pd.to_numeric(IDs_pl[f"Phi_1_{p}_pl_XX"], errors="coerce")
                    S_p = pd.to_numeric(IDs_pl[f"S_{p}_pl_XX"], errors="coerce")
                    P_p = float(pl_scalars.get(f"P_{p}_pl_XX", 0.0))
                    delta_p = float(pl_scalars.get(f"delta_1_{p}_pl_XX", np.nan))
                    denom = float(pl_scalars.get("PS_sum_pl_XX", np.nan))
                    if np.isfinite(denom) and denom != 0:
                        adj = (P_p * Phi_p + (delta_p - float(pl_scalars.get("delta_1_1_pl_XX", np.nan))) * (S_p - P_p)) / denom
                        IDs_pl["_nonmiss_1_pl_XX"] += Phi_p.notna().astype(int)
                        IDs_pl["Phi_1_pl_XX"] = IDs_pl["Phi_1_pl_XX"] + adj.fillna(0.0)

                if waoss_XX == 1 and non_missing == 1 and f"Phi_2_{p}_pl_XX" in IDs_pl.columns and f"abs_delta_D_{p}_pl_XX" in IDs_pl.columns:
                    Phi_p = pd.to_numeric(IDs_pl[f"Phi_2_{p}_pl_XX"], errors="coerce")
                    abs_p = pd.to_numeric(IDs_pl[f"abs_delta_D_{p}_pl_XX"], errors="coerce")
                    E_abs_p = float(pl_scalars.get(f"E_abs_delta_D_{p}_pl_XX", 0.0))
                    delta_p = float(pl_scalars.get(f"delta_2_{p}_pl_XX", np.nan))
                    denom = float(pl_scalars.get("E_abs_delta_D_sum_pl_XX", np.nan))
                    if np.isfinite(denom) and denom != 0:
                        adj = (E_abs_p * Phi_p + (delta_p - float(pl_scalars.get("delta_2_1_pl_XX", np.nan))) * (abs_p - E_abs_p)) / denom
                        IDs_pl["_nonmiss_2_pl_XX"] += Phi_p.notna().astype(int)
                        IDs_pl["Phi_2_pl_XX"] = IDs_pl["Phi_2_pl_XX"] + adj.fillna(0.0)

                if ivwaoss_XX == 1 and non_missing == 1 and f"Phi_3_{p}_pl_XX" in IDs_pl.columns and f"inner_sum_IV_denom_{p}_pl_XX" in IDs_pl.columns:
                    Phi_p = pd.to_numeric(IDs_pl[f"Phi_3_{p}_pl_XX"], errors="coerce")
                    inn_p = pd.to_numeric(IDs_pl[f"inner_sum_IV_denom_{p}_pl_XX"], errors="coerce")
                    denom_p = float(pl_scalars.get(f"denom_delta_IV_{p}_pl_XX", 0.0))
                    delta_p = float(pl_scalars.get(f"delta_3_{p}_pl_XX", np.nan))
                    denom = float(pl_scalars.get("denom_delta_IV_sum_pl_XX", np.nan))
                    if np.isfinite(denom) and denom != 0:
                        adj = (denom_p * Phi_p + (delta_p - float(pl_scalars.get("delta_3_1_pl_XX", np.nan))) * (inn_p - denom_p)) / denom
                        IDs_pl["_nonmiss_3_pl_XX"] += Phi_p.notna().astype(int)
                        IDs_pl["Phi_3_pl_XX"] = IDs_pl["Phi_3_pl_XX"] + adj.fillna(0.0)

            # Stata: replace Phi_pl = . if not_to_use == 0
            for i in (1, 2, 3):
                IDs_pl.loc[IDs_pl[f"_nonmiss_{i}_pl_XX"] == 0, f"Phi_{i}_pl_XX"] = np.nan
                IDs_pl.drop(columns=[f"_nonmiss_{i}_pl_XX"], inplace=True)

            # SE for placebo
            if cluster is not None:
                tmp = IDs_pl[["ID_XX", "cluster_XX"]].drop_duplicates() if "cluster_XX" in IDs_pl.columns else None
                N_bar_c_pl = float(tmp.groupby("cluster_XX")["ID_XX"].size().mean()) if tmp is not None else np.nan
            else:
                N_bar_c_pl = np.nan

            for j in (1, 2, 3):
                enabled = {1: aoss_XX, 2: waoss_XX, 3: ivwaoss_XX}[j]
                if enabled != 1:
                    continue
                phi_col = f"Phi_{j}_pl_XX"
                se = _se_from_phi(IDs_pl[phi_col]) if cluster is None else _se_cluster_from_phi(IDs_pl, "cluster_XX", phi_col, N_bar_c_pl)
                pl_scalars[f"sd_delta_{j}_1_pl_XX"] = se
                pl_scalars[f"LB_{j}_1_pl_XX"] = float(pl_scalars.get(f"delta_{j}_1_pl_XX", np.nan)) - 1.96 * se
                pl_scalars[f"UB_{j}_1_pl_XX"] = float(pl_scalars.get(f"delta_{j}_1_pl_XX", np.nan)) + 1.96 * se

            placebo_results[placebo_index] = pl_scalars

    # --- Divide main effects by sums ---
    if aoss_XX == 1 and scalars.get("PS_sum_XX", 0.0) != 0:
        scalars["delta_1_1_XX"] /= scalars["PS_sum_XX"]
    if waoss_XX == 1 and scalars.get("E_abs_delta_D_sum_XX", 0.0) != 0:
        scalars["delta_2_1_XX"] /= scalars["E_abs_delta_D_sum_XX"]
    if ivwaoss_XX == 1 and scalars.get("denom_delta_IV_sum_XX", 0.0) != 0:
        scalars["delta_3_1_XX"] /= scalars["denom_delta_IV_sum_XX"]

    # --- Aggregate IFs ---
    # Stata: rowtotal(Phi_p columns) + rownonmiss check to set . if ALL pairs missing
    for i in (1, 2, 3):
        IDs_XX[f"Phi_{i}_XX"] = 0.0
        IDs_XX[f"_nonmiss_{i}_XX"] = 0  # track rownonmiss like Stata

    for p in range(2, max_T_XX + 1):
        non_missing = int(scalars.get(f"non_missing_{p}_XX", 0) == 1)
        if aoss_XX == 1 and non_missing == 1 and f"Phi_1_{p}_XX" in IDs_XX.columns and f"S_{p}_XX" in IDs_XX.columns:
            Phi_p = pd.to_numeric(IDs_XX[f"Phi_1_{p}_XX"], errors="coerce")
            S_p = pd.to_numeric(IDs_XX[f"S_{p}_XX"], errors="coerce")
            P_p = float(scalars.get(f"P_{p}_XX", 0.0))
            delta_p = float(scalars.get(f"delta_1_{p}_XX", np.nan))
            denom = float(scalars.get("PS_sum_XX", np.nan))
            if np.isfinite(denom) and denom != 0:
                adj = (P_p * Phi_p + (delta_p - float(scalars.get("delta_1_1_XX", np.nan))) * (S_p - P_p)) / denom
                IDs_XX["_nonmiss_1_XX"] += Phi_p.notna().astype(int)
                IDs_XX["Phi_1_XX"] = IDs_XX["Phi_1_XX"] + adj.fillna(0.0)

        if waoss_XX == 1 and non_missing == 1 and f"Phi_2_{p}_XX" in IDs_XX.columns and f"abs_delta_D_{p}_XX" in IDs_XX.columns:
            Phi_p = pd.to_numeric(IDs_XX[f"Phi_2_{p}_XX"], errors="coerce")
            abs_p = pd.to_numeric(IDs_XX[f"abs_delta_D_{p}_XX"], errors="coerce")
            E_abs_p = float(scalars.get(f"E_abs_delta_D_{p}_XX", 0.0))
            delta_p = float(scalars.get(f"delta_2_{p}_XX", np.nan))
            denom = float(scalars.get("E_abs_delta_D_sum_XX", np.nan))
            if np.isfinite(denom) and denom != 0:
                adj = (E_abs_p * Phi_p + (delta_p - float(scalars.get("delta_2_1_XX", np.nan))) * (abs_p - E_abs_p)) / denom
                IDs_XX["_nonmiss_2_XX"] += Phi_p.notna().astype(int)
                IDs_XX["Phi_2_XX"] = IDs_XX["Phi_2_XX"] + adj.fillna(0.0)

        if ivwaoss_XX == 1 and non_missing == 1 and f"Phi_3_{p}_XX" in IDs_XX.columns and f"inner_sum_IV_denom_{p}_XX" in IDs_XX.columns:
            Phi_p = pd.to_numeric(IDs_XX[f"Phi_3_{p}_XX"], errors="coerce")
            inn_p = pd.to_numeric(IDs_XX[f"inner_sum_IV_denom_{p}_XX"], errors="coerce")
            denom_p = float(scalars.get(f"denom_delta_IV_{p}_XX", 0.0))
            delta_p = float(scalars.get(f"delta_3_{p}_XX", np.nan))
            denom = float(scalars.get("denom_delta_IV_sum_XX", np.nan))
            if np.isfinite(denom) and denom != 0:
                adj = (denom_p * Phi_p + (delta_p - float(scalars.get("delta_3_1_XX", np.nan))) * (inn_p - denom_p)) / denom
                IDs_XX["_nonmiss_3_XX"] += Phi_p.notna().astype(int)
                IDs_XX["Phi_3_XX"] = IDs_XX["Phi_3_XX"] + adj.fillna(0.0)

    # Stata: replace Phi_XX = . if not_to_use_XX == 0
    for i in (1, 2, 3):
        IDs_XX.loc[IDs_XX[f"_nonmiss_{i}_XX"] == 0, f"Phi_{i}_XX"] = np.nan
        IDs_XX.drop(columns=[f"_nonmiss_{i}_XX"], inplace=True)

    # --- SE / CI ---
    if cluster is not None:
        tmp = IDs_XX[["ID_XX", "cluster_XX"]].drop_duplicates() if "cluster_XX" in IDs_XX.columns else None
        N_bar_c_XX = float(tmp.groupby("cluster_XX")["ID_XX"].size().mean()) if tmp is not None else np.nan
    else:
        N_bar_c_XX = np.nan

    for j in (1, 2, 3):
        enabled = {1: aoss_XX, 2: waoss_XX, 3: ivwaoss_XX}[j]
        if enabled != 1:
            continue
        phi_col = f"Phi_{j}_XX"
        se = _se_from_phi(IDs_XX[phi_col]) if cluster is None else _se_cluster_from_phi(IDs_XX, "cluster_XX", phi_col, N_bar_c_XX)
        scalars[f"sd_delta_{j}_1_XX"] = se
        scalars[f"LB_{j}_1_XX"] = float(scalars.get(f"delta_{j}_1_XX", np.nan)) - 1.96 * se
        scalars[f"UB_{j}_1_XX"] = float(scalars.get(f"delta_{j}_1_XX", np.nan)) + 1.96 * se

    # --- Build output table ---
    estims = ["aoss", "waoss", "ivwaoss"]
    # Use Stata-style row names: AS, WAS, IV-WAS (not AOSS, WAOSS, IVWAOSS)
    estim_display = {"aoss": "AS", "waoss": "WAS", "ivwaoss": "IV-WAS"}
    colnames = ["Estimate", "SE", "LB CI", "UB CI", "Switchers", "Stayers"]
    ret = np.full((3 * max_T_XX, 6), np.nan, dtype=float)
    rown = []
    for j, est in enumerate(estims, start=1):
        enabled = {"aoss": aoss_XX, "waoss": waoss_XX, "ivwaoss": ivwaoss_XX}[est]
        for p in range(1, max_T_XX + 1):
            # Stata uses: AS, as_2, as_3 ... WAS, was_2, was_3 ... IV-WAS, iv-was_2, iv-was_3
            if p == 1:
                rown.append(estim_display[est])
            else:
                rown.append(f"{estim_display[est].lower().replace('-', '-')}_{p}")
            if enabled != 1:
                continue
            if p == 1:
                delta = scalars.get(f"delta_{j}_1_XX", np.nan)
                se = scalars.get(f"sd_delta_{j}_1_XX", np.nan)
                lb = scalars.get(f"LB_{j}_1_XX", np.nan)
                ub = scalars.get(f"UB_{j}_1_XX", np.nan)
                ns = scalars.get(f"N_Switchers_{j}_1_XX", np.nan)
                nt = scalars.get(f"N_Stayers_{j}_1_XX", np.nan)
            else:
                delta = scalars.get(f"delta_{j}_{p}_XX", np.nan)
                ns = scalars.get(f"N_Switchers_{j}_{p}_XX", np.nan)
                nt = scalars.get(f"N_Stayers_{j}_{p}_XX", np.nan)
                se = scalars.get(f"sd_delta_{j}_{p}_XX", np.nan)
                lb = scalars.get(f"LB_{j}_{p}_XX", np.nan)
                ub = scalars.get(f"UB_{j}_{p}_XX", np.nan)
            ret[(j - 1) * max_T_XX + (p - 1), :] = [delta, se, lb, ub, ns, nt]

    out_table = pd.DataFrame(ret, index=rown, columns=colnames)

    out: dict[str, Any] = {
        "table": out_table,
        "pairs": int(max_T_XX),
        "N": int(out_table.iloc[0]["Switchers"] + out_table.iloc[0]["Stayers"]) if len(out_table) > 0 and not np.isnan(out_table.iloc[0]["Switchers"]) else int(df["ID_XX"].nunique()),
        "WAOSS Method": {"ra": "Regression Adjustment", "ps": "Propensity Score", "dr": "Doubly Robust"}.get(estimation_method, estimation_method),
        "Polynomial Order": int(order),
        "Common Support": "No Extrapolation" if noextrapolation else "Extrapolation",
    }
    if n_clus_XX is not None:
        out["n_clusters"] = int(n_clus_XX)

    # --- Placebo tables ---
    if placebo > 0:
        for placebo_index in range(1, placebo + 1):
            pl_s = placebo_results.get(placebo_index, {})
            ret_pl = np.full((len(estims), len(colnames)), np.nan, dtype=float)
            rown_pl = []
            for j, est in enumerate(estims, start=1):
                rown_pl.append(f"Placebo_{placebo_index}" if j == 1 else f"Placebo_{placebo_index}_{est}")
                delta = pl_s.get(f"delta_{j}_1_pl_XX", np.nan)
                se = pl_s.get(f"sd_delta_{j}_1_pl_XX", np.nan)
                lb = pl_s.get(f"LB_{j}_1_pl_XX", np.nan)
                ub = pl_s.get(f"UB_{j}_1_pl_XX", np.nan)
                ns = pl_s.get(f"N_Switchers_{j}_1_pl_XX", np.nan)
                nt = pl_s.get(f"N_Stayers_{j}_1_pl_XX", np.nan)
                ret_pl[j - 1, :] = [delta, se, lb, ub, ns, nt]
            out[f"table_placebo_{placebo_index}"] = pd.DataFrame(ret_pl, index=rown_pl, columns=colnames)

        # Backward compatibility: table_placebo = first placebo
        if "table_placebo_1" in out:
            out["table_placebo"] = out["table_placebo_1"]

    # --- AOSS vs WAOSS test ---
    if aoss_vs_waoss and aoss_XX == 1 and waoss_XX == 1:
        diff = float(scalars.get("delta_1_1_XX", np.nan)) - float(scalars.get("delta_2_1_XX", np.nan))
        diff_phi = pd.to_numeric(IDs_XX.get("Phi_1_XX", np.nan), errors="coerce") - pd.to_numeric(IDs_XX.get("Phi_2_XX", np.nan), errors="coerce")
        if cluster is not None and "cluster_XX" in IDs_XX.columns:
            tmp = pd.DataFrame({"cluster_XX": IDs_XX["cluster_XX"], "diff_phi": diff_phi})
            diff_phi_used = tmp.groupby("cluster_XX")["diff_phi"].sum()
        else:
            diff_phi_used = diff_phi
        diff_phi_used = pd.to_numeric(diff_phi_used, errors="coerce").fillna(0.0)
        n_eff = int(len(diff_phi_used))
        if n_eff > 1 and np.isfinite(diff):
            sd_diff = float(diff_phi_used.std(ddof=1))
            if np.isfinite(sd_diff) and sd_diff > 0:
                t_stat = float(diff * math.sqrt(n_eff) / sd_diff)
                Phi_abs = 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2.0)))
                pval = float(2.0 * (1.0 - Phi_abs))
                se_d = float(sd_diff / math.sqrt(n_eff))
                out["aoss_vs_waoss"] = pd.DataFrame({
                    "Estimate": [diff], "SE": [sd_diff],
                    "LB CI": [diff - 1.96 * se_d], "UB CI": [diff + 1.96 * se_d],
                    "t stat.": [t_stat], "pval.": [pval],
                }, index=["Diff."])

    return out


# ============================================================
# QUANTILE PARTITION (by_fd / by_baseline)
# ============================================================

def did_multiplegt_stat_quantiles(
    df: pd.DataFrame, ID: str, Time: str, D: str,
    Z: str | None = None, by_opt: int = 2,
    quantiles: Sequence[float] | None = None,
    by_baseline: bool = False,
) -> dict[str, Any]:
    if quantiles is None:
        quantiles = np.linspace(0, 1, by_opt + 1).tolist()

    df_bal = _balance_panel_fill(df, ID, Time)
    diff_var = Z if Z is not None else D
    _ensure_numeric(df_bal, diff_var)
    df_bal = df_bal.sort_values([ID, Time])
    df_bal["delta_pre_XX"] = df_bal.groupby(ID)[diff_var].diff().abs()

    df_bal["switchers_dummy_XX"] = ((df_bal["delta_pre_XX"].notna()) & (df_bal["delta_pre_XX"] != 0)).astype(int)
    df_bal["stayers_dummy_XX"] = ((df_bal["delta_pre_XX"].notna()) & (df_bal["delta_pre_XX"] == 0)).astype(int)
    df_bal["switchers_N_XX"] = df_bal.groupby(Time)["switchers_dummy_XX"].transform("sum")
    df_bal["stayers_N_XX"] = df_bal.groupby(Time)["stayers_dummy_XX"].transform("sum")
    df_bal["in_agg_XX"] = ((df_bal["switchers_N_XX"] > 0) & (df_bal["stayers_N_XX"] > 1)).astype(int)

    if by_baseline:
        # For by_baseline: use D_{t-1} among switchers at t+1
        df_bal["delta_fwd_XX"] = df_bal.groupby(ID)["delta_pre_XX"].shift(-1)
        df_bal["switchers_fwd_XX"] = ((df_bal["delta_fwd_XX"].notna()) & (df_bal["delta_fwd_XX"] != 0)).astype(int)
        quant_var = D  # baseline treatment D_{t-1}
        # Stata: deltaD_glob_XX is only non-missing for used_in_t_estimation_XX
        # rows (switcher at t AND stayers_t>1).  After F. shift, the filter
        # becomes: switcher at t+1 AND stayers>1 at t+1.  We replicate this
        # via a forward-shifted in_agg_XX.
        df_bal["in_agg_fwd_XX"] = df_bal.groupby(ID)["in_agg_XX"].shift(-1)
        quant_mask = (df_bal["switchers_fwd_XX"] == 1) & (df_bal["in_agg_fwd_XX"] == 1)
    else:
        quant_var = "delta_pre_XX"
        quant_mask = (df_bal["delta_pre_XX"].notna()) & (df_bal["delta_pre_XX"] != 0) & (df_bal["in_agg_XX"] == 1)

    df_switch0 = df_bal[quant_mask].copy()

    if df_switch0.empty:
        df_bal["partition_XX"] = np.nan
        return {"df": df_bal, "val_quantiles": list(quantiles), "quantiles": list(quantiles),
                "switch_df": pd.DataFrame(), "cut_off": []}

    _ensure_numeric(df_switch0, quant_var)
    vals = df_switch0[quant_var].dropna()
    np.quantile(vals, quantiles)

    df_bal["partition_XX"] = np.nan
    for _idx_row in df_bal.index:
        if quant_mask.reindex(df_bal.index, fill_value=False).iloc[0] if len(df_bal) == 0 else False:
            pass
    # Use pd.qcut for equal-frequency binning (matches Stata's xtile)
    labels = list(range(1, by_opt + 1))
    bins = np.quantile(vals, np.linspace(0, 1, by_opt + 1))
    try:
        qcut_result = pd.qcut(df_switch0[quant_var], q=by_opt, labels=labels, duplicates='drop')
        df_bal.loc[quant_mask, "partition_XX"] = qcut_result.astype(float).values
    except ValueError:
        # Fallback: pd.cut with quantile boundaries
        bins = np.quantile(vals, np.linspace(0, 1, by_opt + 1))
        bins[0] = bins[0] - 1e-10
        bins = np.unique(bins)
        if len(bins) < 2:
            df_bal["partition_XX"] = 1
        else:
            actual_labels = list(range(1, len(bins)))
            df_bal.loc[quant_mask, "partition_XX"] = pd.cut(
                df_switch0[quant_var], bins=bins, labels=actual_labels, include_lowest=True).astype(float).values

    # Assign stayers partition=0
    stayer_mask = (df_bal["delta_pre_XX"] == 0) & df_bal["delta_pre_XX"].notna()
    if by_baseline:
        # For by_baseline: the quantile is on D at the pre-period (t), so no
        # shift(-1) is needed in the pairwise function.  Store directly as
        # partition_lead_XX so the pairwise function skips the shift.
        df_bal.rename(columns={"partition_XX": "partition_lead_XX"}, inplace=True)
        # Only set stayer=0 for rows that do NOT already have a bin assignment.
        # A row can be a stayer at t (D_t==D_{t-1}) yet a forward switcher
        # (D_{t+1}!=D_t); such rows must keep their bin value.
        stayer_only = stayer_mask & df_bal["partition_lead_XX"].isna()
        df_bal.loc[stayer_only, "partition_lead_XX"] = 0
    else:
        df_bal.loc[stayer_mask, "partition_XX"] = 0

    switch_df = pd.DataFrame()
    pcol = "partition_lead_XX" if by_baseline else "partition_XX"
    part_vals = df_bal.loc[df_bal[pcol].notna() & (df_bal[pcol] > 0), pcol]
    if not part_vals.empty:
        switch_df = df_bal[df_bal[pcol] > 0].groupby(pcol).agg(
            N_switchers=(quant_var, "size"),
            delta_median=(quant_var, "median")).reset_index()

    return {"df": df_bal, "val_quantiles": list(quantiles), "quantiles": list(quantiles),
            "switch_df": switch_df, "cut_off": list(bins)}


# ============================================================
# TOP-LEVEL FUNCTION
# ============================================================

def did_multiplegt_stat(
    df: pd.DataFrame,
    Y: str, ID: str, Time: str, D: str,
    Z: str | None = None,
    estimator: str | Sequence[str] | None = None,
    estimation_method: str | None = None,
    order: int | list[int] = 1,
    noextrapolation: bool = False,
    placebo: int = 0,
    switchers: str | None = None,
    disaggregate: bool = False,
    aoss_vs_waoss: bool = False,
    exact_match: bool = False,
    by: Sequence[str] | None = None,
    by_fd: int | None = None,
    by_baseline: int | None = None,
    other_treatments: Sequence[str] | None = None,
    cluster: str | None = None,
    weight: str | None = None,
    controls: Sequence[str] | None = None,
    cross_fitting: int = 0,
    trimming: float = 0,
    on_placebo_sample: bool = False,
    bootstrap: int = 0,
    twfe: bool | dict[str, Any] = False,
    seed: int = 0,
    cross_validation: dict[str, Any] | None = None,
    iv_method: str = "manual",
    cf_folds_file: str | None = None,
    asinstata: bool = False,
    model_deltay=None,
    model_stayer=None,
) -> dict[str, Any]:
    """
    Python interface for did_multiplegt_stat.

    Parameters
    ----------
    df : DataFrame - Panel data in long format.
    Y, ID, Time, D : str - Column names for outcome, unit ID, time, treatment.
    Z : str, optional - Instrument variable for IV-WAS.
    estimator : str or list - 'aoss', 'waoss', 'ivwaoss'.
    estimation_method : str - 'ra', 'ps', 'dr' (default: 'dr' without exact_match).
    order : int or list of 1/4/8 ints - Polynomial order(s). 8 ints for IV: first 4=first-stage, last 4=reduced-form.
    placebo : int - Number of placebos (0 = none).
    iv_method : str - IV regression package: 'manual' (default, two OLS), 'linearmodels', or 'econtools'.
    controls : list of str - Control variables.
    cross_fitting : int - Number of cross-fitting folds (0 = none).
    trimming : float - Propensity score trimming threshold (0 = none).
    on_placebo_sample : bool - Estimate only on stayer sample.
    bootstrap : int - Number of bootstrap replications (0 = none).
    twfe : bool or dict - Compare with TWFE regression. Dict keys: same_sample, percentile.
    cross_validation : dict - CV options (algorithm, tolerance, max_k, seed, kfolds).
    by_baseline : int - Number of quantile bins for baseline treatment.
    cf_folds_file : str, optional - Path to CSV with cross-fitting fold IDs exported by Stata.
        Columns: pairwise, placebo_index, estimator_type, ID_XX, cf_sample_id.
        When provided, fold assignments are read from this file instead of generated internally.
    asinstata : bool, default False - If True, use Stata-faithful regressions
        (custom Newton-Raphson logit + statsmodels OLS).  If False (default),
        use scikit-learn LinearRegression / LogisticRegression for all OLS and
        logit estimations.  Note: changing this flag changes numerical results;
        Stata parity tests require asinstata=True.
    model_deltay : object, optional - Custom regression model for E[DeltaY|D1,S=0].
        Must implement .fit(X, y) and .predict(X) (sklearn-style). When provided,
        overrides the default OLS model (regardless of asinstata). Example:
        RandomForestRegressor(n_estimators=100).
    model_stayer : object, optional - Custom classification model for P(stayer|D1).
        Must implement .fit(X, y) and .predict_proba(X) (sklearn-style). When
        provided, overrides the default logit model (regardless of asinstata).
        Example: RandomForestClassifier(n_estimators=100).
    """
    if switchers is not None and switchers not in ("up", "down"):
        raise ValueError("Switchers must be None, 'up' or 'down'.")

    # Default estimator
    if estimator is None and Z is None:
        estimator_list = ["aoss", "waoss"]
    elif estimator is None and Z is not None:
        estimator_list = ["ivwaoss"]
    elif isinstance(estimator, str):
        estimator_list = [estimator]
    else:
        estimator_list = list(estimator)

    # Parse multi-order
    order_reg = order_logit_bis = order_logit_Plus = order_logit_Minus = None
    order_reg_fs = order_logit_bis_fs = order_logit_Plus_fs = order_logit_Minus_fs = None
    fs_orders = None  # first-stage orders (list of 4) when order has 8 ints
    if isinstance(order, (list, tuple)):
        if len(order) == 8:
            order_reg_fs, order_logit_bis_fs, order_logit_Plus_fs, order_logit_Minus_fs = order[:4]
            order_reg, order_logit_bis, order_logit_Plus, order_logit_Minus = order[4:]
            order_scalar = order[4]
            fs_orders = list(order[:4])
        elif len(order) == 4:
            order_reg, order_logit_bis, order_logit_Plus, order_logit_Minus = order
            order_scalar = order[0]
        elif len(order) == 1:
            order_scalar = order[0]
        else:
            raise ValueError("order must be an integer, or a list of 1, 4, or 8 integers.")
    else:
        order_scalar = int(order)

    controls_list = list(controls) if controls is not None else None
    other_treatments_list = list(other_treatments) if other_treatments is not None else None

    # Parse twfe options
    if isinstance(twfe, dict):
        twfe_active = True
        twfe_same_sample = twfe.get("same_sample", False)
        twfe_percentile = twfe.get("percentile", False)
    else:
        twfe_active = bool(twfe)
        twfe_same_sample = False
        twfe_percentile = False

    # Convert trimming from percentage (Stata convention) to decimal
    if trimming > 1:
        trimming = trimming / 100.0

    # Estimation method override
    if exact_match:
        estimation_method = "ra"
        if noextrapolation:
            noextrapolation = False
        order_scalar = 1
        order_reg = order_logit_bis = order_logit_Plus = order_logit_Minus = None
    elif estimation_method is None:
        estimation_method = "dr"

    # Validation
    if "ivwaoss" in estimator_list and any(e in ("aoss", "waoss") for e in estimator_list):
        raise ValueError("Cannot combine AOSS/WAOSS with IV-WAOSS.")
    if "ivwaoss" in estimator_list and Z is None:
        raise ValueError("IV variable Z required for ivwaoss.")
    if by is not None and by_fd is not None:
        raise ValueError("Cannot specify both by and by_fd.")
    if by is not None and by_baseline is not None:
        raise ValueError("Cannot specify both by and by_baseline.")
    if by_fd is not None and by_baseline is not None:
        raise ValueError("Cannot specify both by_fd and by_baseline.")
    if on_placebo_sample and "ivwaoss" in estimator_list:
        raise ValueError("on_placebo_sample not allowed with iv-was.")
    if on_placebo_sample and placebo > 0:
        raise ValueError("on_placebo_sample not allowed with placebo().")
    if bootstrap > 0 and "ivwaoss" not in estimator_list and not twfe_active:
        raise ValueError("Bootstrap is only available for iv-was or combined with twfe.")
    if twfe_active and len(estimator_list) > 1:
        raise ValueError("Only one estimator allowed with twfe.")
    if twfe_active and "aoss" in estimator_list:
        raise ValueError("twfe is only compatible with was and iv-was.")
    if cross_validation is not None and order_scalar > 1:
        print("Warning: order() ignored when cross_validation is specified.")
        order_scalar = 1

    # Store original order for display (could be list or int)
    # For 8-order case: first 4 are first-stage (WAS), last 4 are reduced-form (IV-WAS)
    order_display = order  # Keep original format for display
    if isinstance(order, (list, tuple)) and len(order) == 8:
        # For IV-WAS with 8 orders: store separate orders for display
        order_fs_display = list(order[:4])   # First-stage orders (1,2,3,4)
        order_rf_display = list(order[4:])   # Reduced-form/IV-WAS orders (5,6,7,8)
    else:
        order_fs_display = order_display
        order_rf_display = order_display

    out: dict[str, Any] = {
        "args": {
            "Y": Y, "ID": ID, "Time": Time, "D": D, "Z": Z,
            "estimator": estimator_list, "estimation_method": estimation_method,
            "order": order_scalar, "order_original": order_display,
            "order_fs": order_fs_display, "order_rf": order_rf_display,
            "noextrapolation": noextrapolation,
            "placebo": placebo, "switchers": switchers,
            "disaggregate": disaggregate, "aoss_vs_waoss": aoss_vs_waoss,
            "exact_match": exact_match, "by": list(by) if by else None,
            "by_fd": by_fd, "by_baseline": by_baseline,
            "other_treatments": other_treatments_list,
            "cluster": cluster, "weight": weight,
            "controls": controls_list,
            "cross_fitting": cross_fitting, "trimming": trimming,
            "on_placebo_sample": on_placebo_sample,
            "bootstrap": bootstrap, "twfe": twfe,
            "iv_method": iv_method,
            "asinstata": asinstata,
            "model_deltay": model_deltay,
            "model_stayer": model_stayer,
        }
    }

    df_work = df.copy()
    mode = "_no_by"
    by_levels = ["_no_by"]

    # --- by() ---
    if by is not None:
        by_list = list(by)
        for v in by_list:
            if not by_check(df_work, ID, v):
                raise ValueError(f"by variable {v} must be constant within ID.")
        comp = df_work[by_list].astype(str)
        by_total = comp[by_list[0]]
        for v in by_list[1:]:
            by_total = by_total + "," + comp[v]
        df_work["by_total"] = by_total
        by_levels = sorted(df_work["by_total"].dropna().unique().tolist())
        out["by_levels"] = by_levels
        mode = "by"

    # --- by_fd() ---
    if by_fd is not None:
        q_levels = np.linspace(0, 1, by_fd + 1).tolist()
        by_set = did_multiplegt_stat_quantiles(df=df_work, ID=ID, Time=Time, D=D, Z=Z,
                                               by_opt=by_fd, quantiles=q_levels, by_baseline=False)
        df_work = by_set["df"]
        out["val_quantiles"] = by_set.get("val_quantiles")
        out["switch_df"] = by_set.get("switch_df")
        part = df_work.loc[df_work["partition_XX"].notna() & (df_work["partition_XX"] != 0), "partition_XX"]
        by_levels = sorted(part.astype(int).unique().tolist())
        out["by_levels"] = by_levels
        mode = "by_fd"

    # --- by_baseline() ---
    if by_baseline is not None:
        q_levels = np.linspace(0, 1, by_baseline + 1).tolist()
        by_set = did_multiplegt_stat_quantiles(df=df_work, ID=ID, Time=Time, D=D, Z=Z,
                                               by_opt=by_baseline, quantiles=q_levels, by_baseline=True)
        df_work = by_set["df"]
        out["val_quantiles"] = by_set.get("val_quantiles")
        out["switch_df"] = by_set.get("switch_df")
        # by_baseline uses partition_lead_XX (no shift needed)
        pcol = "partition_lead_XX"
        part = df_work.loc[df_work[pcol].notna() & (df_work[pcol] != 0), pcol]
        by_levels = sorted(part.astype(int).unique().tolist())
        out["by_levels"] = by_levels
        mode = "by_baseline"

    # --- First-stage for IV-WAOSS ---
    if "ivwaoss" in estimator_list:
        fs_order_arg = fs_orders if fs_orders is not None else order
        print("=" * 80)
        print(" " * 30 + "First stage estimation")
        print("=" * 80)
        fs_result = did_multiplegt_stat(
            df, Y=D, ID=ID, Time=Time, D=Z, Z=None,
            estimator="waoss",
            estimation_method=estimation_method,
            order=fs_order_arg,
            noextrapolation=noextrapolation,
            placebo=placebo, switchers=switchers,
            exact_match=exact_match,
            other_treatments=other_treatments,
            cluster=cluster, weight=weight,
            controls=controls,
            cross_fitting=cross_fitting,
            trimming=trimming,
            on_placebo_sample=on_placebo_sample,
            cross_validation=cross_validation,
            cf_folds_file=cf_folds_file,
            asinstata=asinstata,
            model_deltay=model_deltay,
            model_stayer=model_stayer,
        )
        out["first_stage"] = fs_result
        print("=" * 80)
        print(" " * 30 + "Reduced form estimation")
        print("=" * 80)

    def _call_main(df_in, by_fd_opt=None):
        return did_multiplegt_stat_main(
            df=df_in, Y=Y, ID=ID, Time=Time, D=D, Z=Z,
            estimator=estimator_list, estimation_method=estimation_method,
            order=order_scalar, noextrapolation=noextrapolation,
            placebo=placebo, switchers=switchers,
            disaggregate=disaggregate, aoss_vs_waoss=aoss_vs_waoss,
            exact_match=exact_match, weight=weight, cluster=cluster,
            by_fd_opt=by_fd_opt, other_treatments=other_treatments_list,
            controls=controls_list, cross_fitting=cross_fitting,
            trimming=trimming, on_placebo_sample=on_placebo_sample,
            order_reg=order_reg, order_logit_bis=order_logit_bis,
            order_logit_Plus=order_logit_Plus, order_logit_Minus=order_logit_Minus,
            bootstrap=bootstrap, twfe=twfe_active, seed=seed,
            cross_validation_opt=cross_validation,
            cf_folds_file=cf_folds_file,
            asinstata=asinstata,
            model_deltay=model_deltay,
            model_stayer=model_stayer,
        )

    if mode == "_no_by":
        out["results"] = _call_main(df_work)
    elif mode == "by":
        for j, lev in enumerate(by_levels, start=1):
            df_sub = df_work[df_work["by_total"] == lev].copy()
            print(f"Running did_multiplegt_stat with by = {lev}")
            out[f"results_by_{j}"] = _call_main(df_sub)
    elif mode in ("by_fd", "by_baseline"):
        for j, lev in enumerate(by_levels, start=1):
            print(f"Running did_multiplegt_stat for bin {lev}")
            out[f"results_by_{j}"] = _call_main(df_work, by_fd_opt=int(lev))

    # --- Bootstrap ---
    if bootstrap > 0:
        _run_bootstrap(out, df_work, Y, ID, Time, D, Z, estimator_list,
                       estimation_method, order_scalar, noextrapolation, placebo,
                       switchers, exact_match, weight, cluster,
                       other_treatments_list, controls_list,
                       cross_fitting, trimming, on_placebo_sample,
                       order_reg, order_logit_bis, order_logit_Plus, order_logit_Minus,
                       bootstrap, twfe_active, seed, cross_validation,
                       twfe_same_sample=twfe_same_sample,
                       twfe_percentile=twfe_percentile,
                       iv_method=iv_method,
                       asinstata=asinstata,
                       model_deltay=model_deltay,
                       model_stayer=model_stayer)

    out["_class"] = "did_multiplegt_stat"
    return out


# ============================================================
# BOOTSTRAP
# ============================================================

def _run_bootstrap(out, df_work, Y, ID, Time, D, Z, estimator_list,
                   estimation_method, order, noextrapolation, placebo,
                   switchers, exact_match, weight, cluster,
                   other_treatments, controls, cross_fitting, trimming,
                   on_placebo_sample, order_reg, order_logit_bis,
                   order_logit_Plus, order_logit_Minus,
                   n_bootstrap, twfe, seed, cross_validation,
                   twfe_same_sample=False, twfe_percentile=False,
                   iv_method="manual",
                   asinstata=True,
                   model_deltay=None, model_stayer=None):
    """Run bootstrap for IV-WAS SEs and/or TWFE comparison."""

    results_key = "results"
    main_results = out.get(results_key, {})
    if not main_results:
        return

    ivwaoss_XX = int("ivwaoss" in estimator_list)
    int("waoss" in estimator_list)
    int("aoss" in estimator_list)

    max_T = main_results.get("pairs", 2)
    bt_effects = np.full((n_bootstrap, max_T), np.nan)
    # Bootstrap storage for placebo estimates: bt_placebo[pl_idx][i, j] = estimate for bootstrap i, estimator j
    bt_placebo = {}
    len(estimator_list)
    if placebo > 0:
        for pl_idx in range(1, placebo + 1):
            bt_placebo[pl_idx] = np.full((n_bootstrap, 3), np.nan)  # 3 estimators max
    bt_twfe_vals = np.full(n_bootstrap, np.nan) if twfe else None

    cluster_col = cluster if cluster is not None else ID

    print(f"Bootstrap ({n_bootstrap} replications)...")

    for i in range(n_bootstrap):
        if seed != 0:
            rng = np.random.RandomState(seed + i + 1)
        else:
            rng = np.random.RandomState(i + 42)

        # Resample clusters
        clusters = df_work[cluster_col].unique()
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        frames = []
        new_id = 0
        for orig_c in sampled:
            sub = df_work[df_work[cluster_col] == orig_c].copy()
            # Assign new IDs to avoid duplicate panel IDs
            orig_ids = sub[ID].unique()
            id_map = {oid: new_id + k for k, oid in enumerate(orig_ids)}
            new_id += len(orig_ids)
            sub[ID] = sub[ID].map(id_map)
            frames.append(sub)
        df_bt = pd.concat(frames, ignore_index=True)

        try:
            res_bt = did_multiplegt_stat_main(
                df=df_bt, Y=Y, ID=ID, Time=Time, D=D, Z=Z,
                estimator=estimator_list, estimation_method=estimation_method,
                order=order, noextrapolation=noextrapolation,
                placebo=placebo, switchers=switchers, disaggregate=False,
                aoss_vs_waoss=False, exact_match=exact_match,
                weight=weight, cluster=None,
                by_fd_opt=None, other_treatments=other_treatments,
                controls=controls, cross_fitting=cross_fitting,
                trimming=trimming, on_placebo_sample=on_placebo_sample,
                order_reg=order_reg, order_logit_bis=order_logit_bis,
                order_logit_Plus=order_logit_Plus, order_logit_Minus=order_logit_Minus,
                cross_validation_opt=cross_validation,
                asinstata=asinstata,
                model_deltay=model_deltay,
                model_stayer=model_stayer,
            )
            tbl = res_bt.get("table", None)
            if isinstance(tbl, pd.DataFrame):
                j_map = {"aoss": 0, "waoss": 1, "ivwaoss": 2}
                for est in estimator_list:
                    j = j_map[est]
                    for p in range(max_T):
                        row_idx = j * max_T + p
                        if row_idx < len(tbl):
                            bt_effects[i, p] = tbl.iloc[row_idx, 0]

            # Extract placebo estimates from bootstrap
            for pl_idx in range(1, placebo + 1):
                pl_tbl = res_bt.get(f"table_placebo_{pl_idx}", None)
                if isinstance(pl_tbl, pd.DataFrame):
                    for _j_est, est in enumerate(estimator_list):
                        j = {"aoss": 0, "waoss": 1, "ivwaoss": 2}[est]
                        if j < len(pl_tbl):
                            bt_placebo[pl_idx][i, j] = pl_tbl.iloc[j, 0]

            # TWFE within bootstrap
            if twfe:
                df_bt_bal = _balance_panel_fill(df_bt, ID, Time)
                max_T_bt = int(df_bt_bal[Time].max())

                # same_sample filter: restrict to periods with switchers>0 and stayers>=2
                if twfe_same_sample:
                    treat_var = Z if ivwaoss_XX == 1 else D
                    df_bt_bal = df_bt_bal.sort_values([ID, Time])
                    df_bt_bal["_dD_XX"] = df_bt_bal.groupby(ID)[treat_var].diff()
                    df_bt_bal["_stayer_XX"] = (df_bt_bal["_dD_XX"] == 0).astype(float)
                    df_bt_bal.loc[df_bt_bal["_dD_XX"].isna(), "_stayer_XX"] = np.nan
                    types = df_bt_bal.groupby(Time)["_stayer_XX"].std()
                    nb_stay = df_bt_bal.groupby(Time)["_stayer_XX"].sum()
                    yeart_used = (types != 0) & (nb_stay >= 2)
                    # Period 1 inherits from period 2
                    min_t = df_bt_bal[Time].min()
                    if min_t in yeart_used.index:
                        next_t = min_t + 1
                        if next_t in yeart_used.index:
                            yeart_used[min_t] = yeart_used[next_t]
                    good_periods = set(yeart_used[yeart_used].index)
                    df_bt_bal = df_bt_bal[df_bt_bal[Time].isin(good_periods)].copy()
                    df_bt_bal.drop(columns=["_dD_XX", "_stayer_XX"], errors="ignore", inplace=True)
                    max_T_bt = int(df_bt_bal[Time].max()) if len(df_bt_bal) > 0 else 0

                for t_val in range(1, max_T_bt + 1):
                    df_bt_bal[f"T_FE_{t_val}"] = (df_bt_bal[Time] == t_val).astype(float)
                for id_val in df_bt_bal[ID].unique():
                    df_bt_bal[f"ID_FE_{id_val}"] = (df_bt_bal[ID] == id_val).astype(float)
                t_fe_cols = [c for c in df_bt_bal.columns if c.startswith("T_FE_")]
                id_fe_cols = [c for c in df_bt_bal.columns if c.startswith("ID_FE_")]
                fe_formula = f"{Y} ~ {D} + " + " + ".join(t_fe_cols + id_fe_cols)
                if controls:
                    fe_formula += " + " + " + ".join(controls)
                try:
                    if ivwaoss_XX == 1:
                        # IV regression using selected method (manual, linearmodels, or econtools)
                        df_iv = df_bt_bal.dropna(subset=[Y, D, Z]).copy()
                        exog_cols = t_fe_cols + id_fe_cols
                        bt_twfe_vals[i] = run_iv_regression(
                            df=df_iv, y_col=Y, d_col=D, z_col=Z,
                            exog_cols=exog_cols, controls=list(controls) if controls else None,
                            method=iv_method
                        )
                    else:
                        fe_model = smf.ols(fe_formula, data=df_bt_bal.dropna(subset=[Y, D])).fit()
                        bt_twfe_vals[i] = fe_model.params.get(D, np.nan)
                except Exception:
                    pass

        except Exception:
            continue

    # Percentile CIs from bootstrap
    for p in range(max_T):
        vals = bt_effects[:, p]
        vals = vals[~np.isnan(vals)]
        if len(vals) > 5:
            tbl = main_results.get("table", None)
            if isinstance(tbl, pd.DataFrame) and p < len(tbl):
                j_idx = 0
                for est in estimator_list:
                    j_map = {"aoss": 0, "waoss": 1, "ivwaoss": 2}
                    j_idx = j_map[est]
                row = j_idx * max_T + p
                if row < len(tbl):
                    tbl.iloc[row, 1] = np.std(vals, ddof=1)  # SE
                    tbl.iloc[row, 2] = np.percentile(vals, 2.5)  # LB
                    tbl.iloc[row, 3] = np.percentile(vals, 97.5)  # UB

    # Bootstrap SEs for placebo estimates
    for pl_idx in range(1, placebo + 1):
        pl_tbl_key = f"table_placebo_{pl_idx}"
        pl_tbl = out.get(pl_tbl_key, None)
        if not isinstance(pl_tbl, pd.DataFrame):
            continue
        bt_pl = bt_placebo.get(pl_idx, None)
        if bt_pl is None:
            continue
        for _j_est, est in enumerate(estimator_list):
            j = {"aoss": 0, "waoss": 1, "ivwaoss": 2}[est]
            if j >= len(pl_tbl):
                continue
            vals = bt_pl[:, j]
            vals = vals[~np.isnan(vals)]
            if len(vals) > 5:
                bt_se = np.std(vals, ddof=1)
                bt_lb = np.percentile(vals, 2.5)
                bt_ub = np.percentile(vals, 97.5)
                pl_tbl.iloc[j, 1] = bt_se   # SE
                pl_tbl.iloc[j, 2] = bt_lb   # LB
                pl_tbl.iloc[j, 3] = bt_ub   # UB

    # TWFE comparison
    if twfe and bt_twfe_vals is not None:
        twfe_vals = bt_twfe_vals[~np.isnan(bt_twfe_vals)]
        if len(twfe_vals) > 5:
            twfe_mean = np.mean(twfe_vals)
            twfe_sd = np.std(twfe_vals, ddof=1)
            float(main_results["table"].iloc[0, 0]) if isinstance(main_results.get("table"), pd.DataFrame) else np.nan
            diff_vals = twfe_vals - bt_effects[:len(twfe_vals), 0]
            diff_vals = diff_vals[~np.isnan(diff_vals)]
            diff_mean = np.mean(diff_vals) if len(diff_vals) > 0 else np.nan
            diff_sd = np.std(diff_vals, ddof=1) if len(diff_vals) > 1 else np.nan
            t_stat_diff = diff_mean / diff_sd if diff_sd > 0 else np.nan
            pval_diff = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat_diff) / math.sqrt(2)))) if np.isfinite(t_stat_diff) else np.nan

            est_label = estimator_list[0].upper()
            if twfe_percentile:
                twfe_lb = np.percentile(twfe_vals, 2.5)
                twfe_ub = np.percentile(twfe_vals, 97.5)
                diff_lb = np.percentile(diff_vals, 2.5) if len(diff_vals) > 5 else np.nan
                diff_ub = np.percentile(diff_vals, 97.5) if len(diff_vals) > 5 else np.nan
            else:
                twfe_lb = twfe_mean - 1.96 * twfe_sd
                twfe_ub = twfe_mean + 1.96 * twfe_sd
                diff_lb = diff_mean - 1.96 * diff_sd if np.isfinite(diff_sd) else np.nan
                diff_ub = diff_mean + 1.96 * diff_sd if np.isfinite(diff_sd) else np.nan
            out["twfe_comparison"] = pd.DataFrame({
                "Estimate": [twfe_mean, diff_mean],
                "SE": [twfe_sd, diff_sd],
                "LB CI": [twfe_lb, diff_lb],
                "UB CI": [twfe_ub, diff_ub],
                "pval.": [np.nan, pval_diff],
                "t": [np.nan, t_stat_diff],
            }, index=["TWFE", f"TWFE-{est_label}"])


# ============================================================
# PRINT / SUMMARY
# ============================================================

def _fmt_float7(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{float(x):.7f}"


def _fmt_int0(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{float(x):,.0f}"


def _fmt_float5(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{float(x):.5f}"


def mat_print(mat, name=None):
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


def tab_print(mat):
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


def strdisplay(label, value, width=46):
    """Display a label=value line matching Stata's format with right-aligned value."""
    # Stata format: " {it: label}" + spaces + "=" + spaces + "value"
    # Total width is controlled by 'width' parameter
    v = f"{float(value):.0f}" if not isinstance(value, str) else value
    # Calculate spacing: label on left, value on right, = in middle
    eq_pos = width - len(str(v)) - 1  # Position of = sign
    label_space = eq_pos - len(label) - 1
    if label_space < 1:
        label_space = 1
    print(f" {label}{' ' * label_space}={' ' * 1}{v:>{len(str(v))}}")


def summary_did_multiplegt_stat(obj: dict[str, Any]):
    args = obj.get("args", {})
    estim_list = args.get("estimator", ["aoss", "waoss"])
    by_var = args.get("by")
    by_fd = args.get("by_fd")
    by_baseline = args.get("by_baseline")

    if by_var is None and by_fd is None and by_baseline is None:
        by_levs = ["_no_by"]
        by_obj = ["results"]
    else:
        by_levs = list(obj.get("by_levels", []))
        by_obj = [f"results_by_{j + 1}" for j in range(len(by_levs))]

    estims_map = {"aoss": 0, "waoss": 1, "ivwaoss": 2}
    # Display names matching Stata
    estim_titles = {
        "aoss": "Average Slope (AS)",
        "waoss": "Weighted Average Slope (WAS)",
        "ivwaoss": "IV-Weighted Average Slope (IV-WAS)"
    }
    placebo_titles = {
        "aoss": "Placebo(s) AS",
        "waoss": "Placebo(s) WAS",
        "ivwaoss": "Placebo(s) IV-WAS"
    }

    for idx, key in enumerate(by_obj):
        print_obj = obj.get(key, None)
        if print_obj is None:
            continue

        if by_levs[idx] != "_no_by":
            print(f"\n{'#' * 70}")
            print(f" By level: {by_levs[idx]}")

        # Stata-style header with 46-char dashes
        print(f"{' ' * 34}{'-' * 46}")
        table = print_obj.get("table", None)
        pairs = int(print_obj.get("pairs", 1))

        N = print_obj.get("N", np.nan)
        print(f"{' ' * 35}Number of observations{' ' * 5}={' ' * (17 - len(str(int(N))))}{int(N)}")

        methods = {"ra": "reg. adjustment", "dr": "doubly-robust", "ps": "propensity-score"}
        method = args.get("estimation_method", "dr")

        # Show estimation method - Stata shows different label for IV-WAS
        if "ivwaoss" in estim_list:
            print(f"{' ' * 35}IV-WAS Estimation method{' ' * 5}={' ' * 4}{methods.get(method, method)}")
            # Show IV regression package used
            iv_method = args.get("iv_method", "manual")
            iv_pkg_names = {"manual": "manual 2SLS", "linearmodels": "linearmodels.IV2SLS", "econtools": "econtools.ivreg"}
            print(f"{' ' * 35}IV regression package{' ' * 8}={' ' * 4}{iv_pkg_names.get(iv_method, iv_method)}")
        elif "waoss" in estim_list or "aoss" in estim_list:
            print(f"{' ' * 35}Estimation method{' ' * 10}={' ' * 4}{methods.get(method, method)}")

        # Polynomial order with parentheses like Stata
        # For IV-WAS: use order_rf (reduced-form orders, e.g., 5,6,7,8)
        # For WAS/AS: use order_fs or order_original
        if not args.get("exact_match") and args.get("order") is not None:
            if "ivwaoss" in estim_list:
                # Use reduced-form orders for IV-WAS
                order_val = args.get("order_rf", args.get("order_original", args.get("order")))
            else:
                # Use first-stage orders for WAS/AS
                order_val = args.get("order_fs", args.get("order_original", args.get("order")))
            if isinstance(order_val, (list, tuple)):
                order_str = "(" + " ".join(str(x) for x in order_val) + ")"
            else:
                order_str = f"({order_val})"
            print(f"{' ' * 34}Polynomial order{' ' * 11}={' ' * (17 - len(order_str))}{order_str}")

        if args.get("exact_match"):
            print(f"{' ' * 34}Common support{' ' * 12}={' ' * 2}exact matching")
        if args.get("noextrapolation"):
            print(f"{' ' * 34}Common support{' ' * 13}= no extrapolation")
        if args.get("switchers"):
            sw = args.get("switchers")
            print(f"{' ' * 34}Switchers{' ' * 17}={' ' * (17 - len(sw))}{sw}")

        print(f"{' ' * 34}{'-' * 46}")

        n_clusters = print_obj.get("n_clusters", None)
        if n_clusters is not None:
            cluster_name = args.get('cluster')
            print(f"(Std. err. adjusted for {n_clusters} clusters in {cluster_name})")

        for t in ("aoss", "waoss", "ivwaoss"):
            if t not in estim_list:
                continue
            # Stata-style section title
            print(f"{'-' * 80}")
            title = estim_titles[t]
            padding = (80 - len(title)) // 2
            print(f"{' ' * padding}{title}")
            print(f"{'-' * 80}")

            if isinstance(table, pd.DataFrame):
                l_bound = estims_map[t] * pairs
                u_bound = l_bound + (pairs if args.get("disaggregate") else 1)
                mat_sel = table.iloc[l_bound:u_bound]
                mat_print(mat_sel)

            # Placebo tables — grouped by estimator (matching Stata)
            placebo_n = args.get("placebo", 0)
            if placebo_n > 0:
                estim_idx = estims_map[t]  # 0=aoss, 1=waoss, 2=ivwaoss
                pl_rows = []
                for pl_idx in range(1, placebo_n + 1):
                    table_p = print_obj.get(f"table_placebo_{pl_idx}", print_obj.get("table_placebo", None))
                    if isinstance(table_p, pd.DataFrame) and estim_idx < len(table_p):
                        row = table_p.iloc[[estim_idx]].copy()
                        # Rename to Placebo_N - Stata format
                        row.index = [f"Placebo_{pl_idx}"]
                        # Skip if not computed (NaN estimate and 0 switchers)
                        if not (np.isnan(row.iloc[0]["Estimate"]) and row.iloc[0]["Switchers"] == 0):
                            pl_rows.append(row)
                if pl_rows:
                    pl_combined = pd.concat(pl_rows)
                    # Stata-style placebo section title
                    print(f"{'-' * 80}")
                    pl_title = placebo_titles[t]
                    pl_padding = (80 - len(pl_title)) // 2
                    print(f"{' ' * pl_padding}{pl_title}")
                    print(f"{'-' * 80}")
                    mat_print(pl_combined)

        if args.get("aoss_vs_waoss"):
            diff_tab = print_obj.get("aoss_vs_waoss", None)
            if diff_tab is not None:
                print(" ")
                print(f"{'-' * 80}")
                # Stata: "Test of difference between AS and WAS"
                title = "Test of difference between AS and WAS"
                padding = (80 - len(title)) // 2
                print(f"{' ' * padding}{title}")
                print("H0: AS = WAS")
                print(f"{'-' * 80}")
                tab_print(diff_tab)

        # Final closing line
        print(f"{'-' * 80}")

    # First-stage results (IV-WAOSS)
    fs_obj = obj.get("first_stage", None)
    if fs_obj is not None:
        print(f"\n{'=' * 80}")
        print(f"{' ' * 25}First stage estimation")
        print(f"{'=' * 80}")
        summary_did_multiplegt_stat(fs_obj)
        print(f"{'=' * 80}")
        print(f"{' ' * 25}Reduced form estimation (above)")
        print(f"{'=' * 80}")

    # TWFE comparison - Stata format
    twfe_tab = obj.get("twfe_comparison", None)
    if twfe_tab is not None:
        print(" ")
        # Determine estimator label for title
        if "ivwaoss" in estim_list:
            title = "Test of difference between TWFE and IV-WAS"
        elif "waoss" in estim_list:
            title = "Test of difference between TWFE and WAS"
        else:
            title = "Test of difference between TWFE and AS"
        padding = (80 - len(title)) // 2
        print(f"{' ' * padding}{title}")
        if "ivwaoss" in estim_list:
            print("H0: TWFE = IV-WAS")
        elif "waoss" in estim_list:
            print("H0: TWFE = WAS")
        else:
            print("H0: TWFE = AS")
        print(f"{'-' * 80}")
        tab_print(twfe_tab)
        print(f"{'-' * 80}")
        print("Values in Column Estimate. are means of bootstrap's point estimates.")


def print_did_multiplegt_stat(obj: dict[str, Any]):
    summary_did_multiplegt_stat(obj)

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-16

Initial public release.

### Added
- Functional API `did_multiplegt_stat(...)` mirroring the Stata ado-file syntax.
- Scikit-learn style class `DIDMultiplegtStat` with `.fit() / .summary() / .plot() /
  .to_dataframe() / .get_coefficients() / .get_confidence_intervals() /
  .get_params() / .set_params()`.
- AS (Average Slope), WAS (Weighted Average Slope), and IV-WAS estimators.
- Estimation methods: regression adjustment (`ra`), propensity score (`ps`),
  doubly robust (`dr`).
- Stata-faithful backend (`asinstata=True`) with from-scratch Newton-Raphson
  logit matching Stata's `logit, asis` defaults, plus a float32 sweep
  implementation of `_svd_wls` reproducing Stata's `reg` collinearity handling.
- Scikit-learn default backend (`asinstata=False`) for faster modern numerics.
- Pluggable custom nuisance models (`model_deltay=`, `model_stayer=`) accepting
  any sklearn-style `fit`/`predict`/`predict_proba` object.
- Three IV regression backends: manual 2SLS (default), `linearmodels`, `econtools`.
- Multi-period placebos via `placebo=N` (N > 0).
- Bootstrap standard errors and TWFE comparison with `same_sample`/`full_sample`
  and percentile / normal CIs.
- Cross-fitting (`cross_fitting=K`) with Stata-compatible MT19937-64 RNG, and
  external CSV fold-import via `cf_folds_file=` for exact ado-file parity.
- Cluster-robust standard errors via `cluster=`.
- By-group analysis: `by=[...]`, `by_fd=K`, `by_baseline=K`.
- K-fold cross-validation for polynomial order selection via
  `cross_validation={...}`.
- Trimming, no-extrapolation, exact matching, multiple control variables,
  other-treatments adjustment, on-placebo-sample option.

### Documentation
- mkdocs-material site with Stata help-file style reference, Python API
  reference, options matrix, examples, and Stata-to-Python parity guide.

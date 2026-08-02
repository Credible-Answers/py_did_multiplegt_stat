# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Harmonized the public estimator names with the paper and Stata package:
  `estimator="as"`, `"was"`, and `"iv-was"`, plus `as_vs_was=True`.
- Kept `aoss`, `waoss`, `ivwaoss`, and `aoss_vs_waoss` as deprecated input
  aliases for the transition to 0.2.0; returned metadata now uses only the
  canonical names.
- Retired public RA/PS method selection. Doubly robust estimation is automatic,
  while `exact_match=True` activates regression adjustment internally.
- Updated summaries, result metadata, accessors, plots, README examples, and all
  documentation pages to use AS, WAS, and IV-WAS consistently.

### Documentation

- Made `README.md` the source of the GitHub Pages home page so the PyPI project
  description and documentation landing page stay synchronized.
- Corrected the MkDocs site and repository links to the active
  `Credible-Answers/py_did_multiplegt_stat` project.

### Packaging

- Replaced manually duplicated version strings with tag-derived `hatch-vcs`
  versioning and runtime package metadata.
- Changed the release workflow so publishing a GitHub Release tests and builds
  the tagged commit, verifies its version, publishes it to PyPI with Trusted
  Publishing, attaches distributions to the GitHub Release, and deploys the
  tagged documentation.

## [0.1.1] - 2026-07-18

### Fixed

- Matched Stata's missing-value ordering in `noextrapolation` support checks,
  correcting AS/WAS placebo estimates and standard errors when
  `asinstata=True`, `noextrapolation=True`, and multiple placebos are used.

### Documentation

- Added a prominent random-forest example showing how to estimate both
  nuisance functions with custom scikit-learn-style models.
- Made the README examples easier to run in Jupyter notebooks by including
  commented installation commands, self-contained imports, and direct links
  to the example dataset.
- Documented two original Stata commands alongside complete class-based Python
  examples for the default scikit-learn and Stata-faithful nuisance-regression
  backends.
- Added random-forest nuisance-model versions of both documented examples.
- Replaced the functional API example with the main `DIDMultiplegtStat` model
  configured with `asinstata=True` for Stata-faithful results.
- Corrected the Stata-faithful example to match the referenced ado command's
  `lngpinc` outcome, doubly robust method, and `noextra` specification.
- Updated PyPI project metadata to point to the active GitHub repository.

### Packaging

- Added a release guard that requires the Git tag to match the package version.

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

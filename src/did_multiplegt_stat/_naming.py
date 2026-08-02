"""Public estimator terminology and compatibility helpers.

The numerical port historically used the internal names ``aoss``, ``waoss``,
and ``ivwaoss``.  The paper and the Stata package use AS, WAS, and IV-WAS.
This module keeps that translation at the package boundary so the numerical
core can remain stable while every public API uses the published terminology.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

PUBLIC_TO_INTERNAL = {
    "as": "aoss",
    "was": "waoss",
    "iv-was": "ivwaoss",
}
INTERNAL_TO_PUBLIC = {value: key for key, value in PUBLIC_TO_INTERNAL.items()}
ESTIMATOR_POSITIONS = {"aoss": 0, "waoss": 1, "ivwaoss": 2}
ESTIMATOR_LABELS = {"aoss": "AS", "waoss": "WAS", "ivwaoss": "IV-WAS"}


def _clean_estimator_name(estimator: str) -> str:
    if not isinstance(estimator, str):
        raise TypeError("Each estimator must be a string: 'as', 'was', or 'iv-was'.")
    return estimator.strip().lower()


def to_internal_estimator(estimator: str, *, warn_legacy: bool = True) -> str:
    """Return the numerical-core name for a public estimator name."""
    name = _clean_estimator_name(estimator)
    if name in PUBLIC_TO_INTERNAL:
        return PUBLIC_TO_INTERNAL[name]
    if name in INTERNAL_TO_PUBLIC:
        if warn_legacy:
            replacement = INTERNAL_TO_PUBLIC[name]
            warnings.warn(
                f"estimator={estimator!r} is deprecated; use {replacement!r} instead.",
                DeprecationWarning,
                stacklevel=3,
            )
        return name
    allowed = ", ".join(repr(name) for name in PUBLIC_TO_INTERNAL)
    raise ValueError(f"Unknown estimator {estimator!r}. Expected one of: {allowed}.")


def to_public_estimator(estimator: str, *, warn_legacy: bool = True) -> str:
    """Return the canonical public name for an estimator name."""
    internal = to_internal_estimator(estimator, warn_legacy=warn_legacy)
    return INTERNAL_TO_PUBLIC[internal]


def normalize_estimators(
    estimator: str | Sequence[str] | None,
    *,
    has_instrument: bool = False,
    warn_legacy: bool = True,
) -> tuple[list[str], list[str]]:
    """Return canonical public names and corresponding internal names."""
    if estimator is None:
        public = ["iv-was"] if has_instrument else ["as", "was"]
    elif isinstance(estimator, str):
        public = [to_public_estimator(estimator, warn_legacy=warn_legacy)]
    else:
        public = [to_public_estimator(name, warn_legacy=warn_legacy) for name in estimator]

    if not public:
        raise ValueError("estimator must contain at least one of 'as', 'was', or 'iv-was'.")
    if len(set(public)) != len(public):
        raise ValueError("estimator contains duplicate values.")
    internal = [PUBLIC_TO_INTERNAL[name] for name in public]
    return public, internal


def estimator_label(estimator: str) -> str:
    """Return the publication label (AS, WAS, or IV-WAS)."""
    return ESTIMATOR_LABELS[to_internal_estimator(estimator, warn_legacy=False)]


def normalize_color_mapping(colors: Mapping[str, str] | None) -> dict[str, str] | None:
    """Translate public estimator keys in a custom color mapping for plotting."""
    if colors is None:
        return None
    return {
        to_internal_estimator(name, warn_legacy=True): color
        for name, color in colors.items()
    }


def resolve_legacy_options(
    *,
    as_vs_was: bool,
    legacy_options: dict[str, Any],
) -> bool:
    """Resolve the short compatibility window for renamed/retired options."""
    options = dict(legacy_options)

    if "aoss_vs_waoss" in options:
        legacy_value = bool(options.pop("aoss_vs_waoss"))
        warnings.warn(
            "aoss_vs_waoss is deprecated; use as_vs_was instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        if as_vs_was and not legacy_value:
            raise ValueError("Conflicting values were provided for as_vs_was and aoss_vs_waoss.")
        as_vs_was = as_vs_was or legacy_value

    if "estimation_method" in options:
        method = options.pop("estimation_method")
        if method is not None:
            method = str(method).strip().lower()
        if method in (None, "dr"):
            warnings.warn(
                "estimation_method is retired. Doubly robust estimation is now the default; "
                "remove estimation_method='dr'.",
                DeprecationWarning,
                stacklevel=3,
            )
        elif method == "ra":
            raise ValueError(
                "estimation_method='ra' is no longer a public option. "
                "Use exact_match=True when exact matching is intended; the package then "
                "uses regression adjustment internally."
            )
        elif method == "ps":
            raise ValueError(
                "estimation_method='ps' is no longer supported. "
                "The package uses doubly robust estimation by default."
            )
        else:
            raise ValueError(
                "estimation_method is retired. Remove it to use doubly robust estimation."
            )

    if options:
        unexpected = next(iter(options))
        raise TypeError(f"Unexpected keyword argument: {unexpected!r}")
    return as_vs_was

"""Public terminology, retired options, and release configuration."""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_public_signatures_use_paper_terminology():
    from did_multiplegt_stat import DIDMultiplegtStat, did_multiplegt_stat

    class_params = inspect.signature(DIDMultiplegtStat).parameters
    function_params = inspect.signature(did_multiplegt_stat).parameters
    for params in (class_params, function_params):
        assert "as_vs_was" in params
        assert "aoss_vs_waoss" not in params
        assert "estimation_method" not in params


def test_canonical_names_are_stored_in_results(gazoline: pd.DataFrame):
    from did_multiplegt_stat import DIDMultiplegtStat

    model = DIDMultiplegtStat(
        estimator=["as", "was"],
        as_vs_was=True,
    ).fit(gazoline, Y="lngca", ID="id", Time="year", D="tau")

    assert model.estimator == ["as", "was"]
    assert model.get_params()["as_vs_was"] is True
    assert "estimation_method" not in model.get_params()
    assert model.results_["args"]["estimator"] == ["as", "was"]
    assert model.results_["args"]["as_vs_was"] is True
    assert model.results_["args"]["_estimation_method"] == "dr"
    assert "as_vs_was" in model.results_["results"]


def test_legacy_estimator_names_warn_and_normalize():
    from did_multiplegt_stat import DIDMultiplegtStat

    with pytest.warns(DeprecationWarning, match="use 'as'"):
        model = DIDMultiplegtStat(estimator="aoss")
    assert model.estimator == "as"

    with pytest.warns(DeprecationWarning, match="as_vs_was"):
        model = DIDMultiplegtStat(aoss_vs_waoss=True)
    assert model.as_vs_was is True


@pytest.mark.parametrize("method", ["ra", "ps"])
def test_retired_ra_ps_methods_are_rejected(method: str):
    from did_multiplegt_stat import DIDMultiplegtStat

    with pytest.raises(ValueError, match="no longer"):
        DIDMultiplegtStat(estimation_method=method)


def test_legacy_dr_is_a_deprecated_noop():
    from did_multiplegt_stat import DIDMultiplegtStat

    with pytest.warns(DeprecationWarning, match="Doubly robust"):
        model = DIDMultiplegtStat(estimation_method="dr")
    assert "estimation_method" not in model.get_params()


def test_functional_legacy_inputs_warn_and_return_canonical_metadata(
    gazoline: pd.DataFrame,
):
    from did_multiplegt_stat import did_multiplegt_stat

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        result = did_multiplegt_stat(
            gazoline,
            Y="lngca",
            ID="id",
            Time="year",
            D="tau",
            estimator="waoss",
            aoss_vs_waoss=False,
            estimation_method="dr",
        )
    messages = [str(item.message) for item in caught]
    assert any("use 'was'" in message for message in messages)
    assert any("as_vs_was" in message for message in messages)
    assert any("Doubly robust" in message for message in messages)
    assert result["args"]["estimator"] == ["was"]
    assert "estimation_method" not in result["args"]


def test_exact_match_activates_ra_internally(gazoline: pd.DataFrame):
    from did_multiplegt_stat import DIDMultiplegtStat

    model = DIDMultiplegtStat(estimator="was", exact_match=True).fit(
        gazoline,
        Y="lngca",
        ID="id",
        Time="year",
        D="tau",
    )
    assert model.results_["args"]["_estimation_method"] == "ra"


def test_plotting_accepts_canonical_names(gazoline: pd.DataFrame):
    from did_multiplegt_stat import DIDMultiplegtStat

    model = DIDMultiplegtStat(estimator=["as", "was"]).fit(
        gazoline, Y="lngca", ID="id", Time="year", D="tau"
    )
    fig = model.plot(estimator="was", colors={"was": "purple"})
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_tag_is_the_single_package_version_source():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    init_file = (ROOT / "src" / "did_multiplegt_stat" / "__init__.py").read_text(
        encoding="utf-8"
    )
    assert 'dynamic = ["version"]' in pyproject
    assert 'source = "vcs"' in pyproject
    assert "\nversion = " not in pyproject
    assert 'version("did-multiplegt-stat")' in init_file
    assert "__version__ = \"0.1." not in init_file


def test_release_and_documentation_workflows_are_connected():
    release = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    docs = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    assert "release:" in release
    assert "types: [published]" in release
    assert "pypa/gh-action-pypi-publish" in release
    assert "gh release upload" in release
    assert "mkdocs gh-deploy" in release
    assert docs.strip() == '--8<-- "README.md"'

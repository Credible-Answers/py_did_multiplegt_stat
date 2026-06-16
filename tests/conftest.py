"""Shared pytest fixtures and configuration."""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

# Force non-interactive backend so tests run in headless CI environments.
matplotlib.use("Agg")

# Silence noisy library warnings that aren't actionable in test output.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Make Stata reads deterministic across pandas versions.
os.environ.setdefault("PYTHONHASHSEED", "0")

DATA_DIR = Path(__file__).parent / "data"
STATA_REF_DIR = Path(__file__).parent / "stata_reference"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture(scope="session")
def stata_ref_dir() -> Path:
    return STATA_REF_DIR


@pytest.fixture(scope="session")
def gazoline(data_dir: Path) -> pd.DataFrame:
    """Excerpt of Li, Linn, Muehlegger (2014) - 48 US states, 1966-2008."""
    return pd.read_stata(data_dir / "gazoline_did_multiplegt_stat.dta")


@pytest.fixture(scope="session")
def gentzkow(data_dir: Path) -> pd.DataFrame:
    """Gentzkow et al. textbook example (county-level newspaper data)."""
    return pd.read_stata(data_dir / "gentzkowetal_didtextbook.dta")

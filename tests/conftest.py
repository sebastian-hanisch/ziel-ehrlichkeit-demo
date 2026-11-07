"""Gemeinsame Fixtures: die Zählungen über feste Zufallsversuche (einmal je Testlauf, zusammen etwa zehn Sekunden)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import zeh_constants as C  # noqa: E402
import zeh_evaluation as ev  # noqa: E402


@pytest.fixture(scope="session")
def sweeps():
    return {"finite": ev.sweep_finite(C.DEFAULT_SEED), "contest": ev.sweep_contest(C.DEFAULT_SEED), "holdout": ev.sweep_contest(C.HOLDOUT_SEED)}

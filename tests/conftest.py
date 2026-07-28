from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import hydraulic_check as hydraulic
import main_experiment_after_verification as experiment


ROUGHNESS_MM = {"UPVC": 0.018, "PE": 0.020}


@pytest.fixture(scope="session")
def hydraulic_network() -> hydraulic.PipeNetworkData:
    return hydraulic.load_network(
        ROOT / "data" / "network" / "Nodes.xlsx",
        ROOT / "data" / "network" / "Pipes.xlsx",
        ROUGHNESS_MM,
    )


@pytest.fixture(scope="session")
def experiment_network() -> experiment.Network:
    return experiment.load_network(
        ROOT / "data" / "network" / "Nodes.xlsx",
        ROOT / "data" / "network" / "Pipes.xlsx",
        ROUGHNESS_MM,
    )


@pytest.fixture(autouse=True)
def clean_experiment_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    experiment.reset_worker_caches()
    monkeypatch.setattr(experiment, "WORKER_CACHE_ENABLED", True)
    monkeypatch.setattr(experiment, "WORKER_GROUP_CACHE_MAX_ITEMS", 50_000)
    monkeypatch.setattr(experiment, "WORKER_PLAN_CACHE_MAX_ITEMS", 20_000)
    monkeypatch.setattr(experiment, "WORKER_PIPE_LOSS_CACHE_MAX_ITEMS", 20_000)

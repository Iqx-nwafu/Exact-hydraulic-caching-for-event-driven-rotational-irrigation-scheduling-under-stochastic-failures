from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import hydraulic_check as h


def test_identifier_helpers_and_emitter_parsing() -> None:
    assert h.node_key("J9") < h.node_key("J10")
    assert h.node_key("source")[0] == 10**9
    assert h.branch_position("J16") == (1, 6)
    assert h.branch_position("J104") == (10, 4)
    assert h.emitter_node("J11L") == "J11"
    assert h.parse_emitter_list(None) == ()
    assert h.parse_emitter_list(float("nan")) == ()
    assert h.parse_emitter_list(" J11L ; ; J11R ") == ("J11L", "J11R")


def test_published_network_topology_is_complete(hydraulic_network: h.PipeNetworkData) -> None:
    network = hydraulic_network
    assert len(network.nodes) == 71
    assert len(network.pipes) == 70
    assert len(network.irrigation_nodes) == 60
    assert len(network.emitters) == 120
    assert network.node_to_path["J11"] == (0, 10)
    assert network.node_to_path["J106"][-1] == 69
    assert all(path for path in network.node_to_path.values())


def _minimal_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.DataFrame(
        {
            "Node": ["J0", "J1", "J11"],
            "X": [0.0, 1.0, 2.0],
            "Y": [0.0, 0.0, 0.0],
            "Z": [0.0, 0.0, 1.0],
        }
    )
    pipes = pd.DataFrame(
        {
            "Pipe ID": ["P0", "P1"],
            "FromNode": ["J0", "J1"],
            "ToNode": ["J1", "J11"],
            "Length_m": [10.0, 20.0],
            "Diameter_m": [0.2, 0.1],
            "Material": ["UPVC", "PE"],
        }
    )
    return nodes, pipes


def _patch_excel_tables(
    monkeypatch: pytest.MonkeyPatch,
    nodes: pd.DataFrame,
    pipes: pd.DataFrame,
) -> None:
    tables = iter([nodes.copy(), pipes.copy()])
    monkeypatch.setattr(h.pd, "read_excel", lambda _path: next(tables))


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda n, p: (n.drop(columns="X"), p), "Nodes.xlsx missing columns"),
        (lambda n, p: (n, p.drop(columns="Material")), "Pipes.xlsx missing columns"),
        (
            lambda n, p: (pd.concat([n, n.iloc[[0]]], ignore_index=True), p),
            "Duplicated node IDs",
        ),
        (
            lambda n, p: (n, pd.concat([p, p.iloc[[0]]], ignore_index=True)),
            "Duplicated pipe IDs",
        ),
        (
            lambda n, p: (n, p.assign(ToNode=["J1", "missing"])),
            "references node IDs",
        ),
        (
            lambda n, p: (n, p.assign(Material=["steel", "PE"])),
            "Missing material roughness",
        ),
    ],
)
def test_load_network_rejects_invalid_tables(
    monkeypatch: pytest.MonkeyPatch,
    mutator,
    message: str,
) -> None:
    nodes, pipes = mutator(*_minimal_tables())
    _patch_excel_tables(monkeypatch, nodes, pipes)
    with pytest.raises(ValueError, match=message):
        h.load_network(Path("nodes.xlsx"), Path("pipes.xlsx"), {"UPVC": 0.018, "PE": 0.020})


def test_load_network_rejects_multiple_parents(monkeypatch: pytest.MonkeyPatch) -> None:
    nodes, pipes = _minimal_tables()
    pipes = pd.concat(
        [
            pipes,
            pd.DataFrame(
                {
                    "Pipe ID": ["P2"],
                    "FromNode": ["J0"],
                    "ToNode": ["J11"],
                    "Length_m": [5.0],
                    "Diameter_m": [0.1],
                    "Material": ["PE"],
                }
            ),
        ],
        ignore_index=True,
    )
    _patch_excel_tables(monkeypatch, nodes, pipes)
    with pytest.raises(ValueError, match="multiple upstream pipes"):
        h.load_network(Path("nodes.xlsx"), Path("pipes.xlsx"), {"UPVC": 0.018, "PE": 0.020})


def test_build_wntr_model_preserves_network_and_demands(
    hydraulic_network: h.PipeNetworkData,
) -> None:
    wn = h.build_wntr_model(hydraulic_network, {"J11": 0.026}, 25.0)
    assert len(wn.node_name_list) == len(hydraulic_network.nodes)
    assert len(wn.link_name_list) == len(hydraulic_network.pipes)
    assert wn.get_node("J0").base_head == pytest.approx(25.0)
    assert wn.get_node("J11").base_demand == pytest.approx(0.026)


@pytest.mark.parametrize(
    ("reynolds", "expected"),
    [(0.0, 0.0), (1_000.0, 0.064)],
)
def test_colebrook_known_regimes(reynolds: float, expected: float) -> None:
    assert h.colebrook_white_friction_factor(reynolds, 2e-5, 0.1) == pytest.approx(expected)


def test_colebrook_transition_and_turbulent_values_are_physical() -> None:
    f_2000 = h.colebrook_white_friction_factor(2_000.0, 2e-5, 0.1)
    f_3000 = h.colebrook_white_friction_factor(3_000.0, 2e-5, 0.1)
    f_4000 = h.colebrook_white_friction_factor(4_000.0, 2e-5, 0.1)
    f_high = h.colebrook_white_friction_factor(100_000.0, 2e-5, 0.1)
    assert 0.02 < f_3000 < 0.06
    assert h.colebrook_white_friction_factor(2_000.001, 2e-5, 0.1) == pytest.approx(f_2000, rel=1e-6)
    assert h.colebrook_white_friction_factor(3_999.999, 2e-5, 0.1) == pytest.approx(f_4000, rel=1e-6)
    assert 0.01 < f_high < 0.1


def test_darcy_factors_ignore_inactive_pipes() -> None:
    reynolds, factors = h.darcy_factors_from_roughness(
        np.array([0.0, 0.01]),
        np.array([0.0, 1.0]),
        np.array([0.1, 0.1]),
        np.array([0.02, 0.02]),
        1e-6,
    )
    np.testing.assert_array_equal(reynolds[:1], [0.0])
    np.testing.assert_array_equal(factors[:1], [0.0])
    assert reynolds[1] == pytest.approx(100_000.0)
    assert factors[1] > 0.0


def test_custom_hydraulics_conserves_flow_and_applies_engineering_loss(
    hydraulic_network: h.PipeNetworkData,
) -> None:
    result = h.compute_my_hydraulics(
        hydraulic_network,
        ("J11L", "J11R"),
        source_head_m=25.0,
        emitter_demand_m3s=0.013,
        local_loss_ratio=0.1,
    )
    assert result.node_demands_m3s == {"J11": pytest.approx(0.026)}
    path = hydraulic_network.node_to_path["J11"]
    np.testing.assert_allclose(result.pipe_flows_m3s[list(path)], 0.026)
    assert np.count_nonzero(result.pipe_flows_m3s) == len(path)
    np.testing.assert_allclose(
        result.pipe_engineering_loss_1p1_m,
        result.pipe_friction_loss_m * 1.1,
    )
    assert (
        result.node_pressure_engineering_1p1_m["J11"]
        <= result.node_pressure_friction_m["J11"]
    )


def test_group_selection_helpers(hydraulic_network: h.PipeNetworkData) -> None:
    cross = ("J11L", "J21L", "J31L", "J41L")
    same = ("J11L", "J11R", "J12L", "J12R")
    assert h.choose_cross_branch_group([same, cross]) == cross
    chosen, pressure = h.choose_lowest_pressure_group(
        [same, ("J105L", "J105R", "J106L", "J106R")],
        hydraulic_network,
        25.0,
        0.013,
        0.1,
    )
    assert chosen in {same, ("J105L", "J105R", "J106L", "J106R")}
    assert math.isfinite(pressure)
    with pytest.raises(ValueError, match="No valid groups"):
        h.choose_lowest_pressure_group([], hydraulic_network, 25.0, 0.013, 0.1)


def test_predefined_groups_are_validated(
    tmp_path: Path,
    hydraulic_network: h.PipeNetworkData,
) -> None:
    pd.DataFrame(
        [
            {
                "case_id": "C1",
                "category": "test",
                "source_head_m": 24.0,
                "emitters": "J11L;J11R",
            }
        ]
    ).to_csv(tmp_path / "typical_groups.csv", index=False)
    cases = h.read_predefined_typical_groups(hydraulic_network, tmp_path)
    assert cases[0].emitters == ("J11L", "J11R")
    assert cases[0].origin == "published_typical_groups"

    pd.DataFrame(
        [
            {
                "case_id": "C2",
                "category": "test",
                "source_head_m": 24.0,
                "emitters": "unknownL",
            }
        ]
    ).to_csv(tmp_path / "typical_groups.csv", index=False)
    with pytest.raises(ValueError, match="Unknown emitters"):
        h.read_predefined_typical_groups(hydraulic_network, tmp_path)


def test_error_metrics_handle_zero_denominators_and_empty_values() -> None:
    metrics = h.error_metrics("C1", "pressure", np.array([2.0, 1.0]), np.array([1.0, 0.0]))
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(1.0)
    assert metrics["mape_pct"] == pytest.approx(100.0)
    empty = h.error_metrics("C2", "pressure", np.array([]), np.array([]))
    assert math.isnan(empty["mae"])
    assert math.isnan(empty["mape_pct"])


def test_random_validation_cases_are_deterministic_and_bounded(
    hydraulic_network: h.PipeNetworkData,
) -> None:
    left = h.select_random_validation_cases(hydraulic_network, 3, 42, 20.0, 30.0, 4)
    right = h.select_random_validation_cases(hydraulic_network, 3, 42, 20.0, 30.0, 4)
    assert left == right
    assert [case.case_id for case in left] == ["R001", "R002", "R003"]
    assert all(len(set(case.emitters)) == 4 for case in left)
    assert all(20.0 <= case.source_head_m <= 30.0 for case in left)


def test_error_observation_and_distribution_tables() -> None:
    node_df = pd.DataFrame(
        [
            {
                "case_id": "C1",
                "node_id": "J11",
                "my_pressure_friction_only_m": 10.5,
                "epanet_pressure_m": 10.0,
            }
        ]
    )
    pipe_df = pd.DataFrame(
        [
            {
                "case_id": "C1",
                "pipe_id": "P1",
                "my_flow_m3s": 0.01,
                "epanet_flow_m3s": 0.01,
                "my_velocity_ms": 1.2,
                "epanet_velocity_ms": 1.0,
                "my_friction_headloss_m": 0.4,
                "epanet_pipe_headloss_m": 0.5,
            }
        ]
    )
    summary_df = pd.DataFrame(
        [
            {
                "case_id": "C1",
                "sum_my_active_pipe_friction_headloss_m": 0.4,
                "sum_epanet_active_pipe_headloss_m": 0.5,
                "max_my_path_friction_headloss_m": 0.4,
                "max_epanet_path_headloss_m": 0.5,
            }
        ]
    )
    observations = pd.DataFrame(h.error_observation_rows(node_df, pipe_df, summary_df))
    assert len(observations) == 6
    assert observations.loc[observations["metric"].eq("pipe_flow_m3s"), "pct_error"].iloc[0] == 0.0
    overall = h.overall_metrics_from_observations(observations)
    assert set(overall["metric"]) == set(observations["metric"])
    distribution = h.observation_error_distribution(observations)
    assert "abs_error_p95" in distribution

    stats = h.distribution_summary(pd.Series([1.0, 2.0, np.nan]))
    assert stats["mean"] == pytest.approx(1.5)
    empty = h.distribution_summary(pd.Series(dtype=float))
    assert all(math.isnan(value) for value in empty.values())


def test_evaluate_case_rejects_unknown_emitter(
    tmp_path: Path,
    hydraulic_network: h.PipeNetworkData,
) -> None:
    case = h.GroupCase("bad", "test", 25.0, ("unknownL",), "", "test")
    with pytest.raises(ValueError, match="Unknown emitter"):
        h.evaluate_case(hydraulic_network, case, 0.013, 0.1, tmp_path, export_case_inp=False)


def test_evaluate_case_against_exactly_matched_epanet_stub(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    hydraulic_network: h.PipeNetworkData,
) -> None:
    case = h.GroupCase("C1", "test", 25.0, ("J11L", "J11R"), "", "test")

    def exact_epanet(_network, _case, my_calc):
        heads: dict[str, float] = {"J0": _case.source_head_m}
        remaining = set(range(len(_network.pipe_ids)))
        while remaining:
            progressed = False
            for idx in tuple(remaining):
                start = _network.from_nodes[idx]
                if start in heads:
                    heads[_network.to_nodes[idx]] = heads[start] - float(my_calc.pipe_friction_loss_m[idx])
                    remaining.remove(idx)
                    progressed = True
            assert progressed
        pressure = {
            node: heads[node] - _network.node_z[node]
            for node in _network.nodes["node_id"]
            if node != "J0"
        }
        flows = dict(zip(_network.pipe_ids, my_calc.pipe_flows_m3s))
        return pd.Series(pressure), pd.Series(heads), pd.Series(flows)

    monkeypatch.setattr(h, "run_epanet_group", exact_epanet)
    node_rows, pipe_rows, summaries, metric_rows = h.evaluate_case(
        hydraulic_network,
        case,
        0.013,
        0.1,
        tmp_path,
        export_case_inp=False,
    )
    assert len(node_rows) == 1
    assert len(pipe_rows) == len(hydraulic_network.node_to_path["J11"])
    assert summaries[0]["n_emitters"] == 2
    assert all(row["max_abs_error"] == pytest.approx(0.0, abs=1e-12) for row in metric_rows)


def test_hydraulic_main_end_to_end_with_epanet(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = Path(h.__file__).resolve().parents[1]
    args = argparse.Namespace(
        nodes=root / "data" / "network" / "Nodes.xlsx",
        pipes=root / "data" / "network" / "Pipes.xlsx",
        run_data_dir=root / "data" / "hydraulic_verification",
        output_dir=tmp_path,
        source_head=25.0,
        source_head_min=20.0,
        source_head_max=30.0,
        emitter_demand=0.013,
        local_loss_ratio=0.1,
        roughness_upvc_mm=0.018,
        roughness_pe_mm=0.020,
        kinematic_viscosity=h.WATER_KINEMATIC_VISCOSITY_M2S,
        random_cases=1,
        random_seed=7,
        random_emitters_per_case=2,
    )
    parser = type("Parser", (), {"parse_args": lambda self: args})()
    monkeypatch.setattr(h, "build_parser", lambda: parser)
    monkeypatch.setattr(
        h,
        "select_typical_groups",
        lambda *_args, **_kwargs: [
            h.GroupCase("C01", "smoke", 25.0, ("J11L", "J11R"), "smoke", "test")
        ],
    )

    h.main()

    expected = {
        "base_network_darcy_weibach.inp",
        "node_pressure_comparison.csv",
        "pipe_hydraulics_comparison.csv",
        "error_metrics_by_group.csv",
        "random100_overall_error_metrics.csv",
        "hydraulic_verification_report.md",
        "random100_hydraulic_validation_report.md",
        "wntr_environment.json",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
    metrics = pd.read_csv(tmp_path / "error_metrics_by_group.csv")
    assert set(metrics["metric"]) >= {"active_node_pressure_m", "pipe_flow_m3s"}
    random_metrics = pd.read_csv(tmp_path / "random100_overall_error_metrics.csv")
    assert not random_metrics.empty

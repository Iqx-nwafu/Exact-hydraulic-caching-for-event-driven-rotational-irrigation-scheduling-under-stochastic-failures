"""Pipe-network input tables, tree-derived structure, and conversion to/from the platform planning JSON.

Supported inputs:
  * research Excel tables: Nodes.xlsx + Pipes.xlsx + Emitters.csv (emitter_id,node_id);
  * Hydraulic Workbench Studio planning JSON (schema ``hydraulic-irrigation-planning`` v1),
    where each ``fields`` entry is one emitter.

Everything the experiments need beyond the raw tables (source node, pipe path of every node,
laterals, emitters isolated by a node failure) is derived from the tree topology, so no
node-naming convention is assumed.

Conversion CLI:
  python src/network_topology.py to-json  --out network.json
  python src/network_topology.py to-excel --network-json network.json --out-dir converted/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PLATFORM_SCHEMAS = ("hydraulic-irrigation-planning", "hydraulic-irrigation-design")
PLATFORM_SCHEMA_VERSION = 1
PLATFORM_UNITS = {"length": "m", "pressure_head": "m", "volume": "m3", "flow": "m3/h", "duration": "h"}
NODE_COLUMNS = ["node_id", "X", "Y", "Z"]
PIPE_COLUMNS = ["Pipe ID", "FromNode", "ToNode", "Length_m", "Diameter_m", "Material"]
EMITTER_COLUMNS = ["emitter_id", "node_id"]
DEFAULT_EMITTERS_FILE = "Emitters.csv"


@dataclass(frozen=True)
class NetworkTables:
    """Standardised raw tables. ``emitters`` row order defines emitter indices."""

    nodes: pd.DataFrame  # node_id, X, Y, Z (+ any extra input columns)
    pipes: pd.DataFrame  # PIPE_COLUMNS + Roughness_mm (NaN -> resolved from Material)
    emitters: pd.DataFrame  # emitter_id, node_id
    source: str


@dataclass(frozen=True)
class TreeTopology:
    source_node: str
    parent_pipe: dict[str, int]
    children: dict[str, tuple[str, ...]]
    irrigation_nodes: list[str]
    emitters: list[str]
    emitter_to_node: list[str]
    emitter_index: dict[str, int]
    node_to_path: dict[str, tuple[int, ...]]
    laterals: tuple[tuple[str, ...], ...]
    node_lateral: dict[str, int]
    downstream_emitters: dict[str, tuple[int, ...]]


def _strip(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _standardise(nodes: pd.DataFrame, pipes: pd.DataFrame, emitters: pd.DataFrame, source: str) -> NetworkTables:
    nodes = nodes.copy()
    pipes = pipes.copy()
    emitters = emitters.copy()
    nodes.columns = [str(c).strip() for c in nodes.columns]
    pipes.columns = [str(c).strip() for c in pipes.columns]
    emitters.columns = [str(c).strip() for c in emitters.columns]
    nodes = nodes.rename(columns={nodes.columns[0]: "node_id"})
    if "Material" not in pipes.columns:
        pipes["Material"] = ""
    if "Roughness_mm" not in pipes.columns:
        pipes["Roughness_mm"] = math.nan
    for name, df, required in (
        ("nodes", nodes, NODE_COLUMNS),
        ("pipes", pipes, PIPE_COLUMNS),
        ("emitters", emitters, EMITTER_COLUMNS),
    ):
        missing = sorted(set(required) - set(df.columns))
        if missing:
            raise ValueError(f"{source}: {name} table missing columns {missing}")
    nodes["node_id"] = _strip(nodes["node_id"])
    for col in ("Pipe ID", "FromNode", "ToNode"):
        pipes[col] = _strip(pipes[col])
    pipes["Material"] = pipes["Material"].fillna("").astype(str).str.upper().str.strip()
    pipes["Roughness_mm"] = pd.to_numeric(pipes["Roughness_mm"], errors="coerce")
    emitters = emitters[EMITTER_COLUMNS].assign(emitter_id=_strip(emitters["emitter_id"]), node_id=_strip(emitters["node_id"]))
    for name, col in ((nodes, "node_id"), (pipes, "Pipe ID"), (emitters, "emitter_id")):
        dup = name[col][name[col].duplicated()].tolist()
        if dup:
            raise ValueError(f"{source}: duplicated {col}: {dup[:5]}")
    return NetworkTables(nodes=nodes, pipes=pipes, emitters=emitters.reset_index(drop=True), source=source)


def _read_table(path: Path) -> pd.DataFrame:
    if Path(path).suffix.lower() == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip")
    return pd.read_excel(path)


def read_excel_tables(nodes_path: Path, pipes_path: Path, emitters_path: Path | None = None) -> NetworkTables:
    """Nodes/Pipes/Emitters tables, each .xlsx or .csv."""
    emitters_path = Path(nodes_path).with_name(DEFAULT_EMITTERS_FILE) if emitters_path is None else Path(emitters_path)
    if not emitters_path.exists():
        raise FileNotFoundError(
            f"Emitter table not found: {emitters_path}. Provide a CSV with columns emitter_id,node_id "
            "(one row per emitter, in evaluation order) or pass --emitters."
        )
    return _standardise(
        _read_table(nodes_path), _read_table(pipes_path), _read_table(emitters_path), f"{nodes_path} + {pipes_path} + {emitters_path}"
    )


def read_platform_json(path: Path) -> NetworkTables:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if data.get("schema") not in PLATFORM_SCHEMAS:
        raise ValueError(f"{path}: unsupported schema {data.get('schema')!r}; expected one of {PLATFORM_SCHEMAS}")
    topo = data["topology"]
    materials = data.get("metadata", {}).get("research_pipe_material", {})
    nodes = pd.DataFrame(
        [{"node_id": n["node_id"], "X": n.get("x", 0.0), "Y": n.get("y", 0.0), "Z": n["elevation_m"]} for n in topo["nodes"]]
    )
    pipe_rows = []
    for p in topo["pipes"]:
        if p.get("diameter_m") is None:
            raise ValueError(f"{path}: pipe {p['pipe_id']} has no diameter_m (unfrozen design variable).")
        roughness_m = p.get("roughness_m")
        pipe_rows.append(
            {
                "Pipe ID": p["pipe_id"],
                "FromNode": p["start_node"],
                "ToNode": p["end_node"],
                "Length_m": p["length_m"],
                "Diameter_m": p["diameter_m"],
                "Material": p.get("material") or materials.get(p["pipe_id"], ""),
                "Roughness_mm": math.nan if roughness_m is None else float(roughness_m) * 1000.0,
            }
        )
    emitters = pd.DataFrame([{"emitter_id": f["field_id"], "node_id": f["node_id"]} for f in data["fields"]], columns=EMITTER_COLUMNS)
    tables = _standardise(nodes, pd.DataFrame(pipe_rows), emitters, str(path))
    declared = str(topo.get("source_node_id", "")).strip()
    derived = derive_tree(tables).source_node
    if declared and declared != derived:
        raise ValueError(f"{path}: source_node_id {declared!r} differs from the tree root {derived!r}")
    return tables


def read_network_tables(nodes_path: Path, pipes_path: Path | None = None, emitters_path: Path | None = None) -> NetworkTables:
    """``nodes_path`` may be a platform planning JSON, in which case the other paths are ignored."""
    nodes_path = Path(nodes_path)
    if nodes_path.suffix.lower() == ".json":
        return read_platform_json(nodes_path)
    if pipes_path is None:
        raise ValueError("Pipes table is required for Excel input.")
    return read_excel_tables(nodes_path, Path(pipes_path), emitters_path)


def pipe_roughness_mm(tables: NetworkTables, roughness_mm_by_material: dict[str, float]) -> list[float]:
    """Material roughness overrides take precedence; explicit Roughness_mm is used for other pipes."""
    out: list[float] = []
    missing: list[str] = []
    for pipe_id, material, explicit in tables.pipes[["Pipe ID", "Material", "Roughness_mm"]].itertuples(index=False):
        if material in roughness_mm_by_material:
            out.append(float(roughness_mm_by_material[material]))
        elif not math.isnan(explicit):
            out.append(float(explicit))
        else:
            missing.append(f"{pipe_id}({material or 'no material'})")
    if missing:
        raise ValueError(
            f"Missing Darcy-Weisbach roughness for pipes {missing[:10]}: give roughness_m (headloss_formula D-W) "
            "in the JSON / Roughness_mm in Pipes, or a Material with a --roughness-<material>-mm value."
        )
    return out


def derive_tree(tables: NetworkTables) -> TreeTopology:
    node_ids = tables.nodes["node_id"].tolist()
    node_set = set(node_ids)
    from_nodes = tables.pipes["FromNode"].tolist()
    to_nodes = tables.pipes["ToNode"].tolist()
    missing = (set(from_nodes) | set(to_nodes)) - node_set
    if missing:
        raise ValueError(f"Pipes reference missing nodes: {sorted(missing)}")

    parent_pipe: dict[str, int] = {}
    children: dict[str, list[str]] = {n: [] for n in node_ids}
    for idx, (u, v) in enumerate(zip(from_nodes, to_nodes)):
        if v in parent_pipe:
            raise ValueError(f"Node {v} has multiple upstream pipes; tree topology is assumed.")
        parent_pipe[v] = idx
        children[u].append(v)
    roots = [n for n in node_ids if n not in parent_pipe]
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one source (node without upstream pipe), found {roots}")
    source = roots[0]

    def path_to(node: str) -> tuple[int, ...]:
        path: list[int] = []
        seen: set[str] = set()
        cur = node
        while cur != source:
            if cur in seen:
                raise ValueError(f"Cycle detected while tracing {node}.")
            seen.add(cur)
            idx = parent_pipe[cur]
            path.append(idx)
            cur = from_nodes[idx]
        return tuple(reversed(path))

    emitters = tables.emitters["emitter_id"].tolist()
    emitter_to_node = tables.emitters["node_id"].tolist()
    bad = sorted({n for n in emitter_to_node if n not in node_set or n == source})
    if bad:
        raise ValueError(f"Emitters attached to unknown nodes or to the source: {bad}")
    irrigation_nodes = list(dict.fromkeys(emitter_to_node))
    irrigation_set = set(irrigation_nodes)
    node_to_path = {n: path_to(n) for n in node_ids}

    # A lateral is a maximal chain of irrigation nodes hanging from a non-irrigation node.
    lateral_of_root: dict[str, int] = {}
    lateral_members: list[list[str]] = []
    node_lateral: dict[str, int] = {}
    for node in irrigation_nodes:
        root = node
        while from_nodes[parent_pipe[root]] in irrigation_set:
            root = from_nodes[parent_pipe[root]]
        if root not in lateral_of_root:
            lateral_of_root[root] = len(lateral_members)
            lateral_members.append([])
        node_lateral[node] = lateral_of_root[root]
        lateral_members[lateral_of_root[root]].append(node)

    emitters_at: dict[str, list[int]] = {}
    for idx, node in enumerate(emitter_to_node):
        emitters_at.setdefault(node, []).append(idx)
    downstream: dict[str, tuple[int, ...]] = {}
    for node in node_ids:
        stack, found = [node], []
        while stack:
            cur = stack.pop()
            found.extend(emitters_at.get(cur, ()))
            stack.extend(children[cur])
        downstream[node] = tuple(sorted(found))

    return TreeTopology(
        source_node=source,
        parent_pipe=parent_pipe,
        children={k: tuple(v) for k, v in children.items()},
        irrigation_nodes=irrigation_nodes,
        emitters=emitters,
        emitter_to_node=emitter_to_node,
        emitter_index={e: i for i, e in enumerate(emitters)},
        node_to_path=node_to_path,
        laterals=tuple(tuple(m) for m in lateral_members),
        node_lateral=node_lateral,
        downstream_emitters=downstream,
    )


def lateral_emitter_groups(
    laterals: tuple[tuple[str, ...], ...], emitter_to_node: list[str], group_size: int
) -> list[tuple[int, ...]]:
    """Conventional rotation: walk each lateral from upstream to downstream, opening ``group_size`` emitters at a time."""
    groups: list[tuple[int, ...]] = []
    for lateral in laterals:
        members = set(lateral)
        order = [i for i, n in enumerate(emitter_to_node) if n in members]
        groups.extend(tuple(order[s : s + group_size]) for s in range(0, len(order), group_size))
    return groups


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def to_platform_json(
    tables: NetworkTables,
    roughness_mm_by_material: dict[str, float],
    *,
    source_head_m: float,
    design_flow_m3s: float,
    target_volume_m3: float,
    min_pressure_m: float,
    min_velocity_ms: float,
    max_velocity_ms: float,
    group_size: int,
    kinematic_viscosity_m2s: float,
    local_loss_ratio: float,
    name: str = "",
    source_files: list[Path] | None = None,
) -> dict[str, Any]:
    tree = derive_tree(tables)
    roughness = pipe_roughness_mm(tables, roughness_mm_by_material)
    z = tables.nodes.set_index("node_id")["Z"].astype(float)
    source_z = float(z[tree.source_node])
    nodes = [
        {"node_id": r.node_id, "elevation_m": float(r.Z), "x": float(r.X), "y": float(r.Y)}
        for r in tables.nodes[NODE_COLUMNS].itertuples(index=False)
    ]
    pipes = [
        {
            "pipe_id": r[0],
            "start_node": r[1],
            "end_node": r[2],
            "length_m": float(r[3]),
            "diameter_m": float(r[4]),
            "roughness_hw": 150.0,
            "headloss_formula": "D-W",
            "roughness_m": rough / 1000.0,
            "minor_loss_coefficient": 0.0,
            "darcy_factor": None,
            "kinematic_viscosity_m2_s": kinematic_viscosity_m2s,
        }
        for r, rough in zip(tables.pipes[PIPE_COLUMNS].itertuples(index=False), roughness)
    ]
    design_flow_m3h = design_flow_m3s * 3600.0
    fields = [
        {"field_id": e, "node_id": n, "required_volume_m3": target_volume_m3, "design_flow_m3_h": design_flow_m3h}
        for e, n in zip(tree.emitters, tree.emitter_to_node)
    ]
    return {
        "schema": PLATFORM_SCHEMAS[0],
        "schema_version": PLATFORM_SCHEMA_VERSION,
        "units": dict(PLATFORM_UNITS),
        "topology": {
            "nodes": nodes,
            "pipes": pipes,
            "source_node_id": tree.source_node,
            "name": name or f"{len(nodes)} nodes / {len(pipes)} pipes / {len(fields)} emitters",
            "source_head_m": source_head_m - source_z,
            "source_elevation_known": True,
        },
        "fields": fields,
        "settings": {
            "source_pressure_m": source_head_m - source_z,
            "max_source_flow_m3_h": design_flow_m3h * group_size,
            "min_pressure_m": min_pressure_m,
            "min_velocity_m_s": min_velocity_ms,
            "max_velocity_m_s": max_velocity_ms,
            "velocity_scope": "active_only",
            "max_group_size": group_size,
        },
        "metadata": {
            "topology_source": tables.source,
            "source_sha256": {Path(p).name: file_sha256(p) for p in (source_files or [])},
            # The platform loader rejects unknown pipe keys, so materials live here.
            "research_pipe_material": {r[0]: r[5] for r in tables.pipes[PIPE_COLUMNS].itertuples(index=False) if r[5]},
            "research_model": {
                "friction": "Darcy-Weisbach, EPANET 2.2 friction factor (Swamee-Jain, Dunlop transition)",
                "local_loss_ratio": local_loss_ratio,
                "emitter_outflow": "q = q_design * min(p / h_design, 1)",
                "demand": "demand-driven, each field is one emitter",
            },
            "assumptions": [
                "Each field entry is one emitter; field_id is the research emitter id.",
                "required_volume_m3 = design discharge x target duration per emitter.",
                "source_pressure_m is the nominal source pressure head; experiments sample an hourly source head profile.",
                "max_source_flow_m3_h = group_size x emitter design discharge.",
            ],
        },
    }


def write_excel_tables(tables: NetworkTables, out_dir: Path, fmt: str = "xlsx") -> None:
    """openpyxl stores floats with 16 significant digits; use fmt="csv" for a bit-exact copy."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pipes = tables.pipes[PIPE_COLUMNS + ["Roughness_mm"]]
    if pipes["Roughness_mm"].isna().all():
        pipes = pipes.drop(columns="Roughness_mm")
    for name, df in (("Nodes", tables.nodes[NODE_COLUMNS]), ("Pipes", pipes)):
        if fmt == "csv":
            df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")
        else:
            df.to_excel(out_dir / f"{name}.xlsx", index=False)
    tables.emitters.to_csv(out_dir / DEFAULT_EMITTERS_FILE, index=False, encoding="utf-8")


def add_network_args(parser: argparse.ArgumentParser, network_dir: Path) -> None:
    parser.add_argument("--nodes", type=Path, default=network_dir / "Nodes.xlsx")
    parser.add_argument("--pipes", type=Path, default=network_dir / "Pipes.xlsx")
    parser.add_argument("--emitters", type=Path, default=None, help=f"emitter_id,node_id table (default: {DEFAULT_EMITTERS_FILE} next to --nodes)")
    parser.add_argument("--network-json", type=Path, default=None, help="Platform planning JSON; overrides --nodes/--pipes/--emitters.")


def network_paths(args: argparse.Namespace) -> tuple[str, str, str | None]:
    """(nodes, pipes, emitters) as strings for worker initialisers; nodes is the JSON path when --network-json is set."""
    if args.network_json is not None:
        return str(args.network_json), "", None
    return str(args.nodes), str(args.pipes), None if args.emitters is None else str(args.emitters)


def network_cli_args(nodes: str, pipes: str, emitters: str | None) -> list[str]:
    """Inverse of ``network_paths`` for launching another experiment script."""
    if not pipes:
        return ["--network-json", nodes]
    return ["--nodes", nodes, "--pipes", pipes] + (["--emitters", emitters] if emitters else [])


def main() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from main_experiment_after_verification import Params

    root = Path(__file__).resolve().parents[1]
    defaults = Params()
    p = argparse.ArgumentParser(description="Convert between research Excel tables and the platform planning JSON.")
    sub = p.add_subparsers(dest="command", required=True)
    to_json = sub.add_parser("to-json", help="Nodes.xlsx + Pipes.xlsx + Emitters.csv -> planning JSON")
    add_network_args(to_json, root / "data" / "network")
    to_json.add_argument("--out", type=Path, required=True)
    to_json.add_argument("--roughness-upvc-mm", type=float, default=0.018)
    to_json.add_argument("--roughness-pe-mm", type=float, default=0.020)
    to_json.add_argument("--source-head", type=float, default=25.0)
    to_json.add_argument("--name", type=str, default="")
    to_excel = sub.add_parser("to-excel", help="planning JSON -> Nodes + Pipes (.xlsx or .csv) + Emitters.csv")
    to_excel.add_argument("--network-json", type=Path, required=True)
    to_excel.add_argument("--out-dir", type=Path, required=True)
    to_excel.add_argument("--format", choices=("xlsx", "csv"), default="xlsx")
    args = p.parse_args()

    if args.command == "to-excel":
        tables = read_platform_json(args.network_json)
        write_excel_tables(tables, args.out_dir, args.format)
        print(f"Wrote {args.out_dir}: {len(tables.nodes)} nodes, {len(tables.pipes)} pipes, {len(tables.emitters)} emitters")
        return

    nodes, pipes, emitters = network_paths(args)
    tables = read_network_tables(Path(nodes), Path(pipes) if pipes else None, None if emitters is None else Path(emitters))
    sources = [] if args.network_json is not None else [Path(nodes), Path(pipes)]
    data = to_platform_json(
        tables,
        {"UPVC": args.roughness_upvc_mm, "PE": args.roughness_pe_mm},
        source_head_m=args.source_head,
        design_flow_m3s=defaults.design_flow_m3s,
        target_volume_m3=defaults.target_per_emitter_m3,
        min_pressure_m=defaults.design_head_m,
        min_velocity_ms=defaults.min_velocity_ms,
        max_velocity_ms=defaults.max_velocity_ms,
        group_size=defaults.group_size,
        kinematic_viscosity_m2s=defaults.kinematic_viscosity_m2s,
        local_loss_ratio=defaults.local_loss_ratio,
        name=args.name,
        source_files=sources,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {args.out}: {len(data['topology']['nodes'])} nodes, {len(data['topology']['pipes'])} pipes, {len(data['fields'])} emitters")


if __name__ == "__main__":
    main()

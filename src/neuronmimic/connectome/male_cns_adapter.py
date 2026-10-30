"""Thin, provenance-preserving adapter for the local MaleCNS v1.0 release.

The adapter deliberately accepts the two compact tables already audited in the
repository.  It does not infer synapse polarity, convert voxel coordinates to
micrometres, or claim that an aggregate edge is a contact-level cable model.
All 64-bit body identifiers are serialized as strings at the adapter boundary.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class MaleCNSAdapterConfig:
    annotations: Path
    edges: Path
    swc_dir: Path | None = None
    dataset_version: str = "MaleCNS_v1.0"
    source_release: str = "male-cns-v1.0-minconf-0.5"


REQUIRED_ANNOTATION_COLUMNS = {"body", "source_class", "side", "KC_family", "cell_type"}
REQUIRED_EDGE_COLUMNS = {"pre", "post", "synapses"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _string_id(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    # IDs in the source tables are integer-valued float columns when nulls are
    # present.  Reject non-integral values instead of silently rounding them.
    number = float(value)
    if not number.is_integer():
        raise ValueError(f"non-integral MaleCNS body ID: {value!r}")
    return str(int(number))


def normalize_tables(config: MaleCNSAdapterConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read and normalize audited tables without changing source semantics."""
    annotations = pd.read_parquet(config.annotations)
    edges = pd.read_parquet(config.edges)
    missing_a = REQUIRED_ANNOTATION_COLUMNS - set(annotations.columns)
    missing_e = REQUIRED_EDGE_COLUMNS - set(edges.columns)
    if missing_a:
        raise ValueError(f"MaleCNS annotation columns missing: {sorted(missing_a)}")
    if missing_e:
        raise ValueError(f"MaleCNS edge columns missing: {sorted(missing_e)}")
    annotations = annotations.copy()
    annotations["body_id"] = annotations["body"].map(_string_id)
    annotations = annotations.rename(columns={"source_class": "coarse_class"})
    annotations["side"] = annotations["side"].fillna("unknown").astype(str)
    annotations["KC_family"] = annotations["KC_family"].fillna("unresolved").astype(str)
    edges = edges.copy()
    edges["pre_id"] = edges["pre"].map(_string_id)
    edges["post_id"] = edges["post"].map(_string_id)
    edges["synapses"] = pd.to_numeric(edges["synapses"], errors="raise").astype("int64")
    if (edges["synapses"] <= 0).any():
        raise ValueError("MaleCNS aggregate edge table contains non-positive synapse counts")
    edges["direction"] = "pre_to_post"
    return annotations, edges


def _swc_inventory(swc_dir: Path | None) -> dict[str, Any]:
    if swc_dir is None or not swc_dir.exists():
        return {"status": "BLOCKED_MISSING_DATA", "path": str(swc_dir) if swc_dir else None}
    complete = sorted(swc_dir.glob("*.swc"))
    partial = sorted(swc_dir.glob(".*.swc.part"))
    node_counts: list[int] = []
    body_ids: list[str] = []
    for path in complete:
        body_ids.append(path.stem)
        count = 0
        for line in path.open("r", encoding="utf-8", errors="replace"):
            if line.strip() and not line.lstrip().startswith("#"):
                count += 1
        node_counts.append(count)
    return {
        "status": "PARTIAL_REAL_DATA",
        "path": str(swc_dir),
        "complete_files": len(complete),
        "partial_files": len(partial),
        "body_ids_are_filename_stems": True,
        "node_count_min": min(node_counts) if node_counts else None,
        "node_count_median": float(pd.Series(node_counts).median()) if node_counts else None,
        "node_count_max": max(node_counts) if node_counts else None,
        "coordinate_units": "UNKNOWN_SOURCE_UNITS",
        "contact_to_branch_mapping": "BLOCKED_MISSING_DATA",
        "partial_file_names": [p.name for p in partial],
    }


def audit_male_cns(config: MaleCNSAdapterConfig) -> dict[str, Any]:
    annotations, edges = normalize_tables(config)
    annotation_bodies = set(annotations["body_id"].dropna())
    edge_bodies = set(edges["pre_id"]) | set(edges["post_id"])
    swc = _swc_inventory(config.swc_dir)
    swc_bodies = set()
    if config.swc_dir and config.swc_dir.exists():
        swc_bodies = {p.stem for p in config.swc_dir.glob("*.swc")}
    kc = annotations[annotations["coarse_class"].astype(str).str.fullmatch("KC", case=False, na=False)]
    return {
        "dataset_version": config.dataset_version,
        "source_release": config.source_release,
        "status": "VERIFIED_REAL_DATA_WITH_MORPHOLOGY_GAP",
        "inputs": {
            "annotations": {"path": str(config.annotations), "sha256": sha256_file(config.annotations), "rows": int(len(annotations))},
            "edges": {"path": str(config.edges), "sha256": sha256_file(config.edges), "rows": int(len(edges))},
        },
        "annotation_schema": sorted(annotations.columns.tolist()),
        "edge_schema": sorted(edges.columns.tolist()),
        "counts": {
            "annotation_rows": int(len(annotations)),
            "unique_annotated_body_ids": int(annotations["body_id"].nunique()),
            "annotation_null_body_ids": int(annotations["body_id"].isna().sum()),
            "edge_rows": int(len(edges)),
            "unique_edge_body_ids": int(len(edge_bodies)),
            "pre_ids_in_annotation": int(len(set(edges["pre_id"]) & annotation_bodies)),
            "post_ids_in_annotation": int(len(set(edges["post_id"]) & annotation_bodies)),
            "kc_annotation_rows": int(len(kc)),
            "kc_unique_body_ids": int(kc["body_id"].nunique()),
            "aggregate_synapse_count": int(edges["synapses"].sum()),
            "positive_edge_rows": int((edges["synapses"] > 0).sum()),
            "swc_ids_matching_annotation": int(len(swc_bodies & annotation_bodies)),
            "swc_ids_matching_edge": int(len(swc_bodies & edge_bodies)),
        },
        "labels": {
            "coarse_class": annotations["coarse_class"].fillna("unknown").value_counts().to_dict(),
            "kc_family": annotations["KC_family"].fillna("unresolved").value_counts().to_dict(),
            "side": annotations["side"].fillna("unknown").value_counts().to_dict(),
        },
        "morphology": swc,
        "claims": {
            "directed_aggregate_structure": "VERIFIED_REAL_DATA",
            "contact_level_synapse_placement": "BLOCKED_MISSING_DATA",
            "branch_level_cable_mapping": "BLOCKED_MISSING_DATA",
            "synaptic_polarity_or_conductance": "NOT_IDENTIFIABLE_FROM_INPUT",
            "cross_connectome_unit_reuse": "NOT_TESTED",
        },
    }


def write_adapter_outputs(config: MaleCNSAdapterConfig, output_dir: Path) -> dict[str, Path]:
    annotations, edges = normalize_tables(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_annotations = annotations[["body_id", "coarse_class", "side", "KC_family", "cell_type"]].copy()
    normalized_edges = edges[["pre_id", "post_id", "synapses", "direction"]].copy()
    paths = {
        "annotations": output_dir / "malecns_annotations_normalized.parquet",
        "edges": output_dir / "malecns_directed_edges_normalized.parquet",
        "audit": output_dir / "malecns_source_audit.json",
        "provenance": output_dir / "malecns_provenance.json",
    }
    normalized_annotations.to_parquet(paths["annotations"], index=False)
    normalized_edges.to_parquet(paths["edges"], index=False)
    audit = audit_male_cns(config)
    paths["audit"].write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    provenance = {
        "adapter": "bio_fly.male_cns_adapter",
        "adapter_version": "0.1.0",
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        "source_hashes": {"annotations": sha256_file(config.annotations), "edges": sha256_file(config.edges)},
        "output_hashes": {key: sha256_file(path) for key, path in paths.items() if key in {"annotations", "edges"}},
        "status": audit["status"],
    }
    paths["provenance"].write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return paths


__all__ = ["MaleCNSAdapterConfig", "audit_male_cns", "normalize_tables", "write_adapter_outputs"]

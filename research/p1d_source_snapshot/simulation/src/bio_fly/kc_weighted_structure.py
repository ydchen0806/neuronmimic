"""Weighted directed KC structure discovery without functional-label leakage.

The legacy KC analyses first clustered input/output fingerprints and then
attached functional names.  This module answers a narrower, prior question:
given only the weighted directed KC-to-KC graph, what partition resolutions
are reproducible and have excess within-module mass beyond a directed
strength configuration expectation?

Known subtype, lobe, downstream, odor, and lesion information is excluded from
partition discovery.  Those fields are retained only for post-hoc reference
maps or validation outside this module.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd


SIMULATION_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SIMULATION_ROOT.parent
DEFAULT_OUTPUT_ROOT = SIMULATION_ROOT / "outputs" / "kc_weighted_structure_20260825"
FLYWIRE_FEATURES = SIMULATION_ROOT / "outputs" / "kc_function_partition_20260803_codex" / "kc_neuron_functional_features.csv"
FLYWIRE_CONNECTIVITY = PROJECT_ROOT / "external" / "shiu_drosophila_brain_model" / "Connectivity_783.parquet"
LEGACY_OLD5 = SIMULATION_ROOT / "outputs" / "kc_functional_clusters_20260818" / "kc_cluster_assignments.csv"
BANC_META = SIMULATION_ROOT / "data" / "external" / "banc_v888" / "banc_888_meta.feather"
BANC_EDGES = SIMULATION_ROOT / "data" / "external" / "banc_v888" / "banc_888_edgelist_simple_v3.feather"
BANC_EDGES_V2 = PROJECT_ROOT / "data" / "raw" / "banc_v888" / "banc_888_edgelist_simple_v2.feather"
MALE_META = SIMULATION_ROOT / "data" / "external" / "male_cns_v0_9" / "body-annotations-male-cns-v0.9-minconf-0.5.feather"
MALE_EDGES = SIMULATION_ROOT / "data" / "external" / "male_cns_v0_9" / "connectome-weights-male-cns-v0.9-minconf-0.5.feather"


DATASET_REGISTRY_CONTEXT: dict[str, dict[str, str]] = {
    "flywire_v783_kc": {
        "specimen": "adult female FAFB whole brain",
        "sex": "female",
        "kc_annotation_definition": "core_fine_9 == True",
        "synapse_weight_coverage": "complete aggregate integer KC-to-KC synapse counts for retained core-KC scope",
        "alpn_to_kc_input_availability": "available: real 50-channel ALPN-to-KC glomerulus matrix used only post-freeze",
        "morphology_availability": "216 real KC skeleton panel plus soma coordinates; no full contact-level synapse placement",
        "annotation_quality": "curated FlyWire feature table with explicit core-KC flag",
        "independence_role": "reference adult female brain reconstruction",
    },
    "banc_v888_kc": {
        "specimen": "adult female brain-and-cord",
        "sex": "female",
        "kc_annotation_definition": "cell_class == kenyon_cell",
        "synapse_weight_coverage": "complete BANC simple-edgelist integer KC-to-KC counts for retained annotation scope",
        "alpn_to_kc_input_availability": "not prepared for this analysis",
        "morphology_availability": "light SWC coordinate-registration bundle plus bounded flow-centrality split-SWC audit; no full contact-to-cable mapping",
        "annotation_quality": "explicit BANC cell_class field; 98.9 percent complete soma coordinates",
        "independence_role": "supportive female lineage with partial FlyWire provenance overlap",
    },
    "banc_v888_kc_v2": {
        "specimen": "adult female brain-and-cord",
        "sex": "female",
        "kc_annotation_definition": "cell_class == kenyon_cell",
        "synapse_weight_coverage": "BANC v2 paper-version integer KC-to-KC counts; this is the version-coherent W0/W1/W2 comparator for a future v2 contact-derived W5",
        "alpn_to_kc_input_availability": "not prepared for this analysis",
        "morphology_availability": "light SWC coordinate-registration bundle plus bounded flow-centrality split-SWC audit; no full contact-to-cable mapping",
        "annotation_quality": "explicit BANC cell_class field; 98.9 percent complete soma coordinates",
        "independence_role": "within-specimen source-version comparator only; never a second biological replicate",
    },
    "male_cns_v0_9_kc": {
        "specimen": "adult male central nervous system",
        "sex": "male",
        "kc_annotation_definition": "class == Kenyon_Cell",
        "synapse_weight_coverage": "complete Male CNS integer KC-to-KC weight table for retained annotation scope",
        "alpn_to_kc_input_availability": "not prepared for this analysis",
        "morphology_availability": "soma coordinates only; no cable or contact-level weight model",
        "annotation_quality": "explicit Male CNS class field; 99.6 percent complete soma coordinates",
        "independence_role": "independent male CNS reconstruction",
    },
}


@dataclass(frozen=True)
class WeightedStructureConfig:
    """Frozen discovery contract for one or more KC-only connectomes."""

    output_dir: Path = DEFAULT_OUTPUT_ROOT
    datasets: tuple[str, ...] = ("flywire_v783_kc", "banc_v888_kc", "male_cns_v0_9_kc")
    random_seed: int = 20260825
    k_min: int = 2
    k_max: int = 14
    minimum_cluster_size: int = 80
    discovery_fraction: float = 0.50
    embedding_dimensions: int = 16
    kmeans_n_init: int = 30
    seed_repetitions: int = 8
    bootstrap_repetitions: int = 8
    bootstrap_fraction: float = 0.80
    weight_perturbation_repetitions: int = 8
    preprocessing_embedding_dimensions: tuple[int, ...] = (12, 20)
    preprocessing_edge_representations: tuple[str, ...] = (
        "binary_topology",
        "outgoing_strength_normalized_weight",
        "symmetric_in_out_strength_normalized_weight",
    )
    null_repetitions: int = 200
    degree_rewire_repetitions: int = 24
    degree_rewire_swaps_per_edge: float = 0.10
    stability_threshold: float = 0.80
    plateau_fraction_of_peak: float = 0.95
    degree_strength_bins: int = 5


@dataclass(frozen=True)
class KCWeightedConnectome:
    """A single weighted KC-to-KC graph plus scope-only node metadata."""

    dataset_id: str
    nodes: pd.DataFrame
    adjacency: sparse.csr_matrix
    metadata: dict[str, object]


@dataclass(frozen=True)
class PartitionFit:
    labels: np.ndarray
    embedding: np.ndarray
    centers: np.ndarray


def _canonical_dataset_id(value: str) -> str:
    aliases = {
        "flywire": "flywire_v783_kc",
        "flywire_v783": "flywire_v783_kc",
        "banc": "banc_v888_kc",
        "banc_v888": "banc_v888_kc",
        "banc_v2": "banc_v888_kc_v2",
        "banc_v888_v2": "banc_v888_kc_v2",
        "male_cns": "male_cns_v0_9_kc",
        "male_cns_v0_9": "male_cns_v0_9_kc",
    }
    return aliases.get(value, value)


def _parse_coordinate(value: object) -> tuple[float, float, float] | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", str(value))
    if len(numbers) < 3:
        return None
    return tuple(float(item) for item in numbers[:3])


def _load_flywire_nodes() -> pd.DataFrame:
    columns = ["root_id", "core_fine_9", "side", "soma_x_um", "soma_y_um", "soma_z_um"]
    frame = pd.read_csv(FLYWIRE_FEATURES, usecols=columns)
    frame = frame.loc[frame["core_fine_9"].astype(bool)].copy()
    frame = frame.rename(columns={"root_id": "node_id", "soma_x_um": "x", "soma_y_um": "y", "soma_z_um": "z"})
    frame["node_id"] = pd.to_numeric(frame["node_id"], errors="raise").astype(np.int64)
    frame["side"] = frame["side"].fillna("unknown").astype(str)
    return frame[["node_id", "side", "x", "y", "z"]].drop_duplicates("node_id").sort_values("node_id").reset_index(drop=True)


def _load_banc_nodes() -> pd.DataFrame:
    columns = ["root_id", "cell_class", "side", "root_position"]
    frame = pd.read_feather(BANC_META, columns=columns)
    frame = frame.loc[frame["cell_class"].fillna("").astype(str).str.lower().eq("kenyon_cell")].copy()
    coordinates = frame["root_position"].map(_parse_coordinate)
    frame["x"] = coordinates.map(lambda value: value[0] if value else np.nan)
    frame["y"] = coordinates.map(lambda value: value[1] if value else np.nan)
    frame["z"] = coordinates.map(lambda value: value[2] if value else np.nan)
    frame = frame.rename(columns={"root_id": "node_id"})
    frame["node_id"] = pd.to_numeric(frame["node_id"], errors="coerce")
    frame = frame.dropna(subset=["node_id"])
    frame["node_id"] = frame["node_id"].astype(np.int64)
    frame["side"] = frame["side"].fillna("unknown").astype(str)
    return frame[["node_id", "side", "x", "y", "z"]].drop_duplicates("node_id").sort_values("node_id").reset_index(drop=True)


def _load_male_nodes() -> pd.DataFrame:
    columns = ["bodyId", "class", "somaSide", "somaLocation"]
    frame = pd.read_feather(MALE_META, columns=columns)
    frame = frame.loc[frame["class"].fillna("").astype(str).str.lower().eq("kenyon_cell")].copy()
    coordinates = frame["somaLocation"].map(_parse_coordinate)
    frame["x"] = coordinates.map(lambda value: value[0] if value else np.nan)
    frame["y"] = coordinates.map(lambda value: value[1] if value else np.nan)
    frame["z"] = coordinates.map(lambda value: value[2] if value else np.nan)
    frame = frame.rename(columns={"bodyId": "node_id", "somaSide": "side"})
    frame["node_id"] = pd.to_numeric(frame["node_id"], errors="raise").astype(np.int64)
    frame["side"] = frame["side"].fillna("unknown").astype(str)
    return frame[["node_id", "side", "x", "y", "z"]].drop_duplicates("node_id").sort_values("node_id").reset_index(drop=True)


def _stream_parquet_kc_edges(path: Path, node_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import pyarrow.parquet as pq

    node_ids = np.asarray(node_ids, dtype=np.int64)
    source_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity"], batch_size=1_000_000
    ):
        table = batch.to_pandas()
        pre = table["Presynaptic_ID"].to_numpy(dtype=np.int64)
        post = table["Postsynaptic_ID"].to_numpy(dtype=np.int64)
        keep = np.isin(pre, node_ids, assume_unique=False) & np.isin(post, node_ids, assume_unique=False)
        if not keep.any():
            continue
        source_parts.append(pre[keep])
        target_parts.append(post[keep])
        weight_parts.append(table["Connectivity"].to_numpy(dtype=float)[keep])
    return (
        np.concatenate(source_parts) if source_parts else np.empty(0, dtype=np.int64),
        np.concatenate(target_parts) if target_parts else np.empty(0, dtype=np.int64),
        np.concatenate(weight_parts) if weight_parts else np.empty(0, dtype=float),
    )


def _stream_feather_kc_edges(
    path: Path,
    node_ids: np.ndarray,
    pre_column: str,
    post_column: str,
    weight_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter Arrow IPC/Feather record batches without materializing full CNS edges."""

    import pyarrow as pa

    node_ids = np.asarray(node_ids, dtype=np.int64)
    source_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    with pa.memory_map(str(path), "r") as mapped:
        reader = pa.ipc.open_file(mapped)
        index = {field.name: position for position, field in enumerate(reader.schema)}
        missing = {pre_column, post_column, weight_column} - set(index)
        if missing:
            raise ValueError(f"{path.name}: missing edge columns {sorted(missing)}")
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            pre = batch.column(index[pre_column]).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            post = batch.column(index[post_column]).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            weight = batch.column(index[weight_column]).to_numpy(zero_copy_only=False).astype(float, copy=False)
            keep = np.isin(pre, node_ids, assume_unique=False) & np.isin(post, node_ids, assume_unique=False)
            if keep.any():
                source_parts.append(pre[keep])
                target_parts.append(post[keep])
                weight_parts.append(weight[keep])
    return (
        np.concatenate(source_parts) if source_parts else np.empty(0, dtype=np.int64),
        np.concatenate(target_parts) if target_parts else np.empty(0, dtype=np.int64),
        np.concatenate(weight_parts) if weight_parts else np.empty(0, dtype=float),
    )


def _make_connectome(
    dataset_id: str,
    nodes: pd.DataFrame,
    source: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    source_paths: dict[str, Path],
) -> KCWeightedConnectome:
    nodes = nodes.drop_duplicates("node_id").sort_values("node_id").reset_index(drop=True)
    node_index = pd.Series(np.arange(len(nodes), dtype=np.int64), index=nodes["node_id"].to_numpy(dtype=np.int64))
    pre_index = pd.Series(source).map(node_index).to_numpy(dtype=float)
    post_index = pd.Series(target).map(node_index).to_numpy(dtype=float)
    valid = np.isfinite(pre_index) & np.isfinite(post_index) & np.isfinite(weight) & (weight > 0.0)
    pre_index = pre_index[valid].astype(np.int64)
    post_index = post_index[valid].astype(np.int64)
    values = weight[valid].astype(float)
    no_self = pre_index != post_index
    dropped_self = int((~no_self).sum())
    adjacency = sparse.coo_matrix(
        (values[no_self], (pre_index[no_self], post_index[no_self])), shape=(len(nodes), len(nodes))
    ).tocsr()
    adjacency.sum_duplicates()
    adjacency.eliminate_zeros()
    if adjacency.nnz == 0:
        raise ValueError(f"{dataset_id}: KC-to-KC graph is empty")
    metadata = {
        "dataset_id": dataset_id,
        "node_count": int(len(nodes)),
        "edge_count": int(adjacency.nnz),
        "synaptic_weight": float(adjacency.sum()),
        "dropped_self_edges": dropped_self,
        "source_paths": {name: str(path) for name, path in source_paths.items()},
    }
    return KCWeightedConnectome(dataset_id=dataset_id, nodes=nodes, adjacency=adjacency, metadata=metadata)


def load_kc_connectome(dataset_id: str) -> KCWeightedConnectome:
    """Load a KC-only weighted graph; annotations define scope, never labels."""

    dataset_id = _canonical_dataset_id(dataset_id)
    if dataset_id == "flywire_v783_kc":
        nodes = _load_flywire_nodes()
        source, target, weight = _stream_parquet_kc_edges(FLYWIRE_CONNECTIVITY, nodes["node_id"].to_numpy())
        return _make_connectome(
            dataset_id, nodes, source, target, weight, {"nodes": FLYWIRE_FEATURES, "edges": FLYWIRE_CONNECTIVITY}
        )
    if dataset_id == "banc_v888_kc":
        nodes = _load_banc_nodes()
        source, target, weight = _stream_feather_kc_edges(BANC_EDGES, nodes["node_id"].to_numpy(), "pre", "post", "count")
        return _make_connectome(dataset_id, nodes, source, target, weight, {"nodes": BANC_META, "edges": BANC_EDGES})
    if dataset_id == "banc_v888_kc_v2":
        nodes = _load_banc_nodes()
        source, target, weight = _stream_feather_kc_edges(BANC_EDGES_V2, nodes["node_id"].to_numpy(), "pre", "post", "count")
        return _make_connectome(dataset_id, nodes, source, target, weight, {"nodes": BANC_META, "edges": BANC_EDGES_V2})
    if dataset_id == "male_cns_v0_9_kc":
        nodes = _load_male_nodes()
        source, target, weight = _stream_feather_kc_edges(MALE_EDGES, nodes["node_id"].to_numpy(), "body_pre", "body_post", "weight")
        return _make_connectome(dataset_id, nodes, source, target, weight, {"nodes": MALE_META, "edges": MALE_EDGES})
    raise ValueError(f"unsupported KC connectome {dataset_id!r}")


def _split_weighted_adjacency(
    adjacency: sparse.csr_matrix, fraction: float, rng: np.random.Generator
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("discovery_fraction must lie between zero and one")
    source = adjacency.tocoo()
    counts = np.rint(source.data).astype(np.int64)
    if np.max(np.abs(source.data - counts)) > 1e-8:
        raise ValueError("KC weighted structure discovery requires integer synapse counts")
    discovery_values = rng.binomial(counts, fraction).astype(float)
    evaluation_values = counts.astype(float) - discovery_values
    discovery = sparse.coo_matrix((discovery_values, (source.row, source.col)), shape=adjacency.shape).tocsr()
    evaluation = sparse.coo_matrix((evaluation_values, (source.row, source.col)), shape=adjacency.shape).tocsr()
    discovery.eliminate_zeros()
    evaluation.eliminate_zeros()
    return discovery, evaluation


def _embedding(adjacency: sparse.csr_matrix, dimensions: int, random_seed: int) -> np.ndarray:
    n_nodes = adjacency.shape[0]
    if adjacency.nnz == 0 or n_nodes < 3:
        raise ValueError("cannot embed an empty or too-small directed graph")
    outgoing = np.asarray(adjacency.sum(axis=1)).ravel()
    incoming = np.asarray(adjacency.sum(axis=0)).ravel()
    left = sparse.diags(np.divide(1.0, np.sqrt(outgoing), out=np.zeros_like(outgoing), where=outgoing > 0.0))
    right = sparse.diags(np.divide(1.0, np.sqrt(incoming), out=np.zeros_like(incoming), where=incoming > 0.0))
    normalized = left @ adjacency @ right
    k = min(max(2, int(dimensions)), n_nodes - 1)
    try:
        vectors_left, singular, vectors_right = randomized_svd(
            normalized,
            n_components=k,
            n_iter=7,
            random_state=random_seed,
            flip_sign=True,
        )
    except Exception as error:
        raise RuntimeError(f"directed spectral embedding failed: {error}") from error
    order = np.argsort(singular)[::-1]
    singular = singular[order]
    vectors_left = vectors_left[:, order]
    vectors_right = vectors_right[order, :].T
    scale = np.sqrt(np.maximum(singular, 0.0))
    combined = np.concatenate((vectors_left * scale, vectors_right * scale), axis=1)
    return StandardScaler().fit_transform(combined)


def _canonicalize_labels(labels: np.ndarray, embedding: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    unique = np.unique(labels)
    centers = np.vstack([embedding[labels == label].mean(axis=0) for label in unique])
    sizes = np.asarray([(labels == label).sum() for label in unique], dtype=int)
    # Largest groups are C1/C2..., with an embedding coordinate tie-breaker.
    order = sorted(range(len(unique)), key=lambda index: (-int(sizes[index]), *centers[index, :2].tolist()))
    remap = {int(unique[old]): int(new) for new, old in enumerate(order)}
    canonical = np.asarray([remap[int(value)] for value in labels], dtype=np.int64)
    canonical_centers = np.vstack([embedding[canonical == label].mean(axis=0) for label in range(len(unique))])
    return canonical, canonical_centers


def _fit_partition(embedding: np.ndarray, k: int, random_seed: int, n_init: int = 10) -> PartitionFit:
    model = KMeans(n_clusters=int(k), n_init=int(n_init), random_state=random_seed)
    raw = model.fit_predict(embedding)
    labels, centers = _canonicalize_labels(raw, embedding)
    return PartitionFit(labels=labels, embedding=embedding, centers=centers)


def _directed_structure_metrics(adjacency: sparse.csr_matrix, labels: np.ndarray) -> tuple[dict[str, float], pd.DataFrame]:
    """Score a partition against a directed weighted strength configuration."""

    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) != adjacency.shape[0]:
        raise ValueError("labels do not match adjacency")
    edge = adjacency.tocoo()
    total = float(edge.data.sum())
    if total <= 0.0:
        raise ValueError("adjacency has no positive mass")
    n_clusters = int(labels.max()) + 1
    same = labels[edge.row] == labels[edge.col]
    internal_mass = float(edge.data[same].sum())
    outgoing = np.asarray(adjacency.sum(axis=1)).ravel()
    incoming = np.asarray(adjacency.sum(axis=0)).ravel()
    cluster_out = np.bincount(labels, weights=outgoing, minlength=n_clusters)
    cluster_in = np.bincount(labels, weights=incoming, minlength=n_clusters)
    expected_internal = float(np.dot(cluster_out, cluster_in) / total)
    q = (internal_mass - expected_internal) / total
    counts = np.bincount(labels, minlength=n_clusters).astype(int)
    rows: list[dict[str, object]] = []
    for cluster in range(n_clusters):
        member = labels == cluster
        outbound = float(edge.data[member[edge.row] & ~member[edge.col]].sum())
        inbound = float(edge.data[~member[edge.row] & member[edge.col]].sum())
        inner = float(edge.data[member[edge.row] & member[edge.col]].sum())
        possible_inner = max(int(counts[cluster]) * max(int(counts[cluster]) - 1, 0), 1)
        possible_external = max(int(counts[cluster]) * max(len(labels) - int(counts[cluster]), 0), 1)
        rows.append(
            {
                "module_id": f"C{cluster + 1}",
                "module_index": cluster,
                "n_neurons": int(counts[cluster]),
                "within_weight": inner,
                "outgoing_external_weight": outbound,
                "incoming_external_weight": inbound,
                "within_weight_per_possible_pair": inner / possible_inner,
                "outgoing_external_weight_per_possible_pair": outbound / possible_external,
                "incoming_external_weight_per_possible_pair": inbound / possible_external,
                "within_to_outgoing_external_ratio": inner / max(outbound, 1e-12),
                "within_to_incoming_external_ratio": inner / max(inbound, 1e-12),
                "outgoing_strength": float(outgoing[member].sum()),
                "incoming_strength": float(incoming[member].sum()),
            }
        )
    possible_within = float(np.sum(counts * np.maximum(counts - 1, 0)))
    possible_total = float(len(labels) * max(len(labels) - 1, 0))
    possible_between = max(possible_total - possible_within, 1.0)
    within_per_pair = internal_mass / max(possible_within, 1.0)
    between_per_pair = (total - internal_mass) / possible_between
    metrics = {
        "total_weight": total,
        "within_weight": internal_mass,
        "between_weight": total - internal_mass,
        "within_weight_fraction": internal_mass / total,
        "within_to_between_total_weight_ratio": internal_mass / max(total - internal_mass, 1e-12),
        "within_minus_between_total_weight_contrast": (internal_mass - (total - internal_mass)) / total,
        "expected_within_weight_strength_configuration": expected_internal,
        "directed_strength_corrected_modularity": q,
        "within_weight_enrichment_over_strength_expected": internal_mass / max(expected_internal, 1e-12),
        "within_weight_per_possible_pair": within_per_pair,
        "between_weight_per_possible_pair": between_per_pair,
        "within_between_per_pair_gap": within_per_pair - between_per_pair,
        "within_between_per_pair_ratio": within_per_pair / max(between_per_pair, 1e-12),
        "minimum_cluster_size": float(counts.min()),
        "effective_cluster_number": float(np.exp(-np.sum((counts / counts.sum()) * np.log(counts / counts.sum())))),
    }
    return metrics, pd.DataFrame.from_records(rows)


def _assignment_consistency(reference: np.ndarray, other: np.ndarray) -> tuple[float, np.ndarray]:
    reference = np.asarray(reference, dtype=int)
    other = np.asarray(other, dtype=int)
    contingency = pd.crosstab(pd.Series(reference), pd.Series(other)).to_numpy(dtype=int)
    row, column = linear_sum_assignment(-contingency)
    mapping = {int(column_value): int(row_value) for row_value, column_value in zip(row, column, strict=True)}
    mapped = np.asarray([mapping.get(int(value), -1) for value in other], dtype=int)
    return float(np.mean(mapped == reference)), mapped


def _bootstrap_embeddings(
    discovery: sparse.csr_matrix, config: WeightedStructureConfig, offset: int
) -> list[np.ndarray]:
    edge = discovery.tocoo()
    counts = np.rint(edge.data).astype(np.int64)
    embeddings: list[np.ndarray] = []
    for replicate in range(config.bootstrap_repetitions):
        rng = np.random.default_rng(config.random_seed + offset + replicate)
        values = rng.binomial(counts, config.bootstrap_fraction).astype(float)
        boot = sparse.coo_matrix((values, (edge.row, edge.col)), shape=discovery.shape).tocsr()
        boot.eliminate_zeros()
        embeddings.append(_embedding(boot, config.embedding_dimensions, config.random_seed + offset + 1000 + replicate))
    return embeddings


def _poisson_weight_perturbation_embeddings(
    discovery: sparse.csr_matrix, config: WeightedStructureConfig, offset: int
) -> list[np.ndarray]:
    """Perturb observed synapse counts without injecting functional information.

    Each retained edge count is resampled from a Poisson distribution with its
    discovery-half count as the mean.  This is deliberately distinct from the
    80-percent edge subsampling used for the primary bootstrap gate.
    """

    edge = discovery.tocoo()
    counts = np.rint(edge.data).astype(np.int64)
    embeddings: list[np.ndarray] = []
    for replicate in range(config.weight_perturbation_repetitions):
        rng = np.random.default_rng(config.random_seed + offset + 20_000 + replicate)
        values = rng.poisson(counts).astype(float)
        perturbed = sparse.coo_matrix((values, (edge.row, edge.col)), shape=discovery.shape).tocsr()
        perturbed.eliminate_zeros()
        embeddings.append(
            _embedding(
                perturbed,
                config.embedding_dimensions,
                config.random_seed + offset + 21_000 + replicate,
            )
        )
    return embeddings


def _rank_correlation(left: pd.Series, right: pd.Series) -> float:
    """Small dependency-free Spearman diagnostic for candidate-objective trends."""

    paired = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(paired) < 3 or paired["left"].nunique() < 2 or paired["right"].nunique() < 2:
        return float("nan")
    return float(paired["left"].rank(method="average").corr(paired["right"].rank(method="average")))


def _candidate_objective_diagnostics(curve: pd.DataFrame) -> pd.DataFrame:
    """Document why directed strength-corrected Q, rather than a ratio, is primary."""

    candidates = (
        (
            "directed_strength_corrected_modularity",
            "heldout_directed_strength_corrected_modularity",
            "primary; subtracts expected within mass under directed source/target strengths",
        ),
        (
            "within_to_between_total_weight_ratio",
            "heldout_within_to_between_total_weight_ratio",
            "descriptive raw-mass ratio; susceptible to module-size composition",
        ),
        (
            "within_minus_between_total_weight_contrast",
            "heldout_within_minus_between_total_weight_contrast",
            "descriptive signed raw-mass contrast; does not correct strength",
        ),
        (
            "within_enrichment_over_strength_expected",
            "heldout_within_weight_enrichment_over_strength_expected",
            "normalized enrichment; shares the directed-strength expectation",
        ),
        (
            "within_between_per_possible_pair_ratio",
            "heldout_within_between_per_pair_ratio",
            "density diagnostic; sensitive to small-module resolution",
        ),
    )
    valid = curve.loc[curve["valid_minimum_cluster_size"]].copy()
    records: list[dict[str, object]] = []
    for name, column, rationale in candidates:
        if column not in curve:
            continue
        available = valid.loc[np.isfinite(valid[column])].copy()
        selected_k = int(available.loc[available[column].idxmax(), "k"]) if not available.empty else np.nan
        records.append(
            {
                "candidate_objective": name,
                "source_column": column,
                "is_primary": bool(name == "directed_strength_corrected_modularity"),
                "selection_rationale": rationale,
                "max_valid_k_without_stability_gate": selected_k,
                "max_valid_score_without_stability_gate": float(available[column].max()) if not available.empty else np.nan,
                "score_vs_k_spearman": _rank_correlation(curve[column], curve["k"]),
                "score_vs_minimum_cluster_size_spearman": _rank_correlation(
                    curve[column], curve["heldout_minimum_cluster_size"]
                ),
                "score_vs_effective_cluster_number_spearman": _rank_correlation(
                    curve[column], curve["heldout_effective_cluster_number"]
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _row_normalized_adjacency(adjacency: sparse.csr_matrix) -> sparse.csr_matrix:
    outgoing = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.divide(1.0, outgoing, out=np.zeros_like(outgoing), where=outgoing > 0.0)
    return sparse.diags(inverse) @ adjacency


def _symmetric_strength_normalized_adjacency(adjacency: sparse.csr_matrix) -> sparse.csr_matrix:
    outgoing = np.asarray(adjacency.sum(axis=1)).ravel()
    incoming = np.asarray(adjacency.sum(axis=0)).ravel()
    source_scale = np.divide(1.0, np.sqrt(outgoing), out=np.zeros_like(outgoing), where=outgoing > 0.0)
    target_scale = np.divide(1.0, np.sqrt(incoming), out=np.zeros_like(incoming), where=incoming > 0.0)
    return sparse.diags(source_scale) @ adjacency @ sparse.diags(target_scale)


def _edge_representations(adjacency: sparse.csr_matrix) -> tuple[tuple[str, sparse.csr_matrix], ...]:
    """Return transparent alternative edge encodings for post-selection checks."""

    binary = adjacency.copy().astype(float)
    binary.data[:] = 1.0
    return (
        ("synapse_count_weight", adjacency.tocsr()),
        ("binary_topology", binary.tocsr()),
        ("outgoing_strength_normalized_weight", _row_normalized_adjacency(adjacency).tocsr()),
        ("symmetric_in_out_strength_normalized_weight", _symmetric_strength_normalized_adjacency(adjacency).tocsr()),
    )


def _weight_representation_metrics(adjacency: sparse.csr_matrix, labels: np.ndarray) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for name, matrix in _edge_representations(adjacency):
        metrics, _ = _directed_structure_metrics(matrix.tocsr(), labels)
        records.append({"edge_representation": name, **metrics})
    return pd.DataFrame.from_records(records)


def _complexity_penalty(k: int, edge_weight: float) -> float:
    """BIC-like block-parameter penalty, reported as a diagnostic not a p value."""

    return float((k * k * math.log(max(edge_weight, 2.0))) / (2.0 * max(edge_weight, 1.0)))


def _select_optimal_k(curve: pd.DataFrame, config: WeightedStructureConfig) -> tuple[int, str]:
    usable = curve.loc[
        curve["valid_minimum_cluster_size"]
        & curve["stable_by_ari_gate"]
        & np.isfinite(curve["heldout_directed_strength_corrected_modularity"])
    ].copy()
    if usable.empty:
        bootstrap_tested = (
            curve["bootstrap_stability_tested"]
            if "bootstrap_stability_tested" in curve
            else curve["stability_ari_mean"].notna()
        )
        fallback = curve.loc[curve["valid_minimum_cluster_size"] & bootstrap_tested].copy()
        if fallback.empty:
            fallback = curve.loc[curve["valid_minimum_cluster_size"]].copy()
        if fallback.empty:
            raise RuntimeError("no K meets the frozen minimum-cluster-size requirement")
        stability_column = (
            "stability_ari_mean" if fallback["stability_ari_mean"].notna().any() else "seed_stability_ari_mean"
        )
        return (
            int(
                fallback.sort_values(
                    [stability_column, "heldout_directed_strength_corrected_modularity", "k"],
                    ascending=[False, False, True],
                ).iloc[0]["k"]
            ),
            "NO_K_PASSED_BOOTSTRAP_STABILITY_GATE",
        )
    peak = float(usable["heldout_directed_strength_corrected_modularity"].max())
    threshold = peak * config.plateau_fraction_of_peak if peak > 0.0 else peak
    plateau = usable.loc[usable["heldout_directed_strength_corrected_modularity"] >= threshold]
    chosen = plateau.sort_values("k").iloc[0]
    return int(chosen["k"]), "SMALLEST_STABLE_K_WITHIN_FROZEN_FRACTION_OF_PEAK_HELDOUT_Q"


def _quantile_codes(values: np.ndarray, bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.empty(0, dtype=int)
    ranked = pd.Series(values).rank(method="first")
    n_bins = max(1, min(int(bins), len(values)))
    return pd.qcut(ranked, q=n_bins, labels=False, duplicates="drop").to_numpy(dtype=int)


def _q_from_edges(row: np.ndarray, col: np.ndarray, data: np.ndarray, labels: np.ndarray, n_nodes: int) -> float:
    matrix = sparse.coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes)).tocsr()
    return float(_directed_structure_metrics(matrix, labels)[0]["directed_strength_corrected_modularity"])


def _degree_strength_stratified_labels(
    labels: np.ndarray, adjacency: sparse.csr_matrix, rng: np.random.Generator, bins: int
) -> tuple[np.ndarray, float]:
    degrees = np.diff(adjacency.indptr) + np.diff(adjacency.tocsc().indptr)
    strengths = np.asarray(adjacency.sum(axis=1)).ravel() + np.asarray(adjacency.sum(axis=0)).ravel()
    strata = _quantile_codes(np.log1p(degrees), bins) * max(bins, 1) + _quantile_codes(np.log1p(strengths), bins)
    shuffled = labels.copy()
    mixable = np.zeros(len(labels), dtype=bool)
    for value in np.unique(strata):
        indices = np.flatnonzero(strata == value)
        if len(indices) < 2 or len(np.unique(labels[indices])) < 2:
            continue
        shuffled[indices] = rng.permutation(labels[indices])
        mixable[indices] = True
    return shuffled, float(mixable.mean())


def _degree_preserving_weighted_rewire(
    adjacency: sparse.csr_matrix,
    labels: np.ndarray,
    rng: np.random.Generator,
    swaps_per_edge: float,
) -> tuple[float, int, int]:
    """2-switch directed baseline: exact degrees, global weight multiset, no loops/duplicates."""

    edge = adjacency.tocoo()
    row = edge.row.copy()
    col = edge.col.copy()
    data = edge.data.copy()
    n_nodes = adjacency.shape[0]
    keys = set((row.astype(np.int64) * n_nodes + col.astype(np.int64)).tolist())
    attempts = max(1, int(len(row) * swaps_per_edge))
    accepted = 0
    for _ in range(attempts):
        first, second = rng.integers(0, len(row), size=2)
        if first == second or row[first] == row[second] or col[first] == col[second]:
            continue
        new_first = int(row[first]) * n_nodes + int(col[second])
        new_second = int(row[second]) * n_nodes + int(col[first])
        old_first = int(row[first]) * n_nodes + int(col[first])
        old_second = int(row[second]) * n_nodes + int(col[second])
        if row[first] == col[second] or row[second] == col[first]:
            continue
        if new_first in keys or new_second in keys:
            continue
        keys.remove(old_first)
        keys.remove(old_second)
        keys.add(new_first)
        keys.add(new_second)
        col[first], col[second] = col[second], col[first]
        accepted += 1
    shuffled_weight = rng.permutation(data)
    return _q_from_edges(row, col, shuffled_weight, labels, n_nodes), accepted, attempts


def _null_distribution(
    adjacency: sparse.csr_matrix, labels: np.ndarray, config: WeightedStructureConfig, offset: int
) -> pd.DataFrame:
    edge = adjacency.tocoo()
    n_nodes = adjacency.shape[0]
    total = float(edge.data.sum())
    observed = float(_directed_structure_metrics(adjacency, labels)[0]["directed_strength_corrected_modularity"])
    cluster_sizes = np.bincount(labels)
    rows: list[dict[str, object]] = []
    degree_mixable: list[float] = []
    for replicate in range(config.null_repetitions):
        rng = np.random.default_rng(config.random_seed + offset + replicate)
        random_labels = rng.permutation(labels)
        rows.append({"null_kind": "random_size_matched_label_partition", "replicate": replicate, "q": _q_from_edges(edge.row, edge.col, edge.data, random_labels, n_nodes), "observed_q": observed})
        degree_labels, fraction = _degree_strength_stratified_labels(labels, adjacency, rng, config.degree_strength_bins)
        degree_mixable.append(fraction)
        rows.append({"null_kind": "degree_strength_stratified_label_partition", "replicate": replicate, "q": _q_from_edges(edge.row, edge.col, edge.data, degree_labels, n_nodes), "observed_q": observed})
        rows.append({"null_kind": "topology_fixed_global_weight_shuffle", "replicate": replicate, "q": _q_from_edges(edge.row, edge.col, rng.permutation(edge.data), labels, n_nodes), "observed_q": observed})
        outgoing = np.bincount(labels[edge.row], weights=edge.data, minlength=len(cluster_sizes))
        incoming = np.bincount(labels[edge.col], weights=edge.data, minlength=len(cluster_sizes))
        probability = float(np.dot(outgoing, incoming) / max(total * total, 1e-12))
        sampled_internal = float(rng.binomial(int(round(total)), min(max(probability, 0.0), 1.0)))
        expected = float(np.dot(outgoing, incoming) / total)
        rows.append({"null_kind": "directed_strength_configuration", "replicate": replicate, "q": (sampled_internal - expected) / total, "observed_q": observed})
    for replicate in range(config.degree_rewire_repetitions):
        rng = np.random.default_rng(config.random_seed + offset + 10_000 + replicate)
        q, accepted, attempts = _degree_preserving_weighted_rewire(adjacency, labels, rng, config.degree_rewire_swaps_per_edge)
        rows.append({"null_kind": "directed_degree_preserving_rewire_global_weight_shuffle", "replicate": replicate, "q": q, "observed_q": observed, "accepted_swaps": accepted, "attempted_swaps": attempts})
    frame = pd.DataFrame.from_records(rows)
    frame["degree_strength_label_null_mixable_node_fraction"] = float(np.mean(degree_mixable)) if degree_mixable else np.nan
    return frame


def _soft_membership(
    labels: np.ndarray, embedding: np.ndarray, centers: np.ndarray, stability_counts: np.ndarray, n_replicates: int
) -> pd.DataFrame:
    squared = ((embedding[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    nearest = np.min(squared, axis=1)
    temperature = max(float(np.median(nearest[nearest > 0.0])) if np.any(nearest > 0.0) else 1.0, 1e-12)
    logits = -(squared - squared.min(axis=1, keepdims=True)) / temperature
    probability = np.exp(logits - logits.max(axis=1, keepdims=True))
    probability /= probability.sum(axis=1, keepdims=True)
    entropy = -np.sum(np.where(probability > 0.0, probability * np.log(probability), 0.0), axis=1) / math.log(probability.shape[1])
    max_probability = probability.max(axis=1)
    consistency = stability_counts / max(n_replicates, 1)
    status = np.where(
        (max_probability >= 0.75) & (consistency >= 0.80),
        "core_neuron",
        np.where((entropy >= 0.55) | (consistency < 0.80), "boundary_neuron", "intermediate_membership"),
    )
    frame = pd.DataFrame(
        {
            "structural_module": [f"C{value + 1}" for value in labels],
            "membership_max": max_probability,
            "membership_entropy_normalized": entropy,
            "bootstrap_assignment_consistency": consistency,
            "soft_structure_status": status,
        }
    )
    for index in range(probability.shape[1]):
        frame[f"membership_C{index + 1}"] = probability[:, index]
    return frame


def _spatial_partition(nodes: pd.DataFrame, k: int, random_seed: int) -> np.ndarray | None:
    """Fit a coordinate-only control, allowing documented sparse coordinate gaps."""

    columns = ["x", "y", "z"]
    if any(column not in nodes for column in columns):
        return None
    values = nodes.loc[:, columns].to_numpy(dtype=float)
    finite_row = np.isfinite(values).all(axis=1)
    # BANC and male metadata have sparse absent soma locations.  The control
    # remains coordinate-only after per-axis median imputation, but is omitted
    # when coordinate coverage is materially incomplete.
    if float(finite_row.mean()) < 0.95:
        return None
    medians = np.nanmedian(values, axis=0)
    values = np.where(np.isfinite(values), values, medians[None, :])
    return _fit_partition(StandardScaler().fit_transform(values), k, random_seed).labels


def _degree_partition(adjacency: sparse.csr_matrix, k: int, random_seed: int) -> np.ndarray:
    values = np.column_stack(
        (
            np.diff(adjacency.indptr),
            np.diff(adjacency.tocsc().indptr),
            np.asarray(adjacency.sum(axis=1)).ravel(),
            np.asarray(adjacency.sum(axis=0)).ravel(),
        )
    )
    return _fit_partition(StandardScaler().fit_transform(np.log1p(values)), k, random_seed).labels


def _legacy_old5_labels(nodes: pd.DataFrame) -> np.ndarray | None:
    if not LEGACY_OLD5.exists():
        return None
    legacy = pd.read_csv(LEGACY_OLD5, usecols=["root_id", "cluster"]).rename(columns={"root_id": "node_id", "cluster": "old5_cluster"})
    merged = nodes[["node_id"]].merge(legacy, on="node_id", how="left")
    # U0 is a historical unassigned residual, not one of the five old clusters.
    labels = merged["old5_cluster"].fillna("unmapped").astype(str)
    return labels.to_numpy(dtype=object)


def _old5_map(old_labels: np.ndarray, new_labels: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = pd.crosstab(pd.Series(old_labels, name="old5_cluster"), pd.Series([f"C{item + 1}" for item in new_labels], name="structural_module"))
    long = table.stack().rename("n_neurons").reset_index()
    long["old5_fraction"] = long["n_neurons"] / long.groupby("old5_cluster")["n_neurons"].transform("sum")
    summary_rows: list[dict[str, object]] = []
    for cluster, frame in long.groupby("old5_cluster", sort=True):
        values = frame["old5_fraction"].to_numpy(dtype=float)
        entropy = -np.sum(values[values > 0.0] * np.log(values[values > 0.0]))
        effective = float(np.exp(entropy))
        dominant = frame.loc[frame["old5_fraction"].idxmax()]
        summary_rows.append(
            {
                "old5_cluster": cluster,
                "n_neurons": int(frame["n_neurons"].sum()),
                "dominant_new_module": str(dominant["structural_module"]),
                "dominant_new_module_fraction": float(dominant["old5_fraction"]),
                "effective_number_of_new_modules": effective,
                "reinterpretation": "stable_reference_group"
                if float(dominant["old5_fraction"]) >= 0.80
                else ("heterogeneous_split" if effective >= 1.75 else "partial_overlap"),
            }
        )
    return long, pd.DataFrame.from_records(summary_rows)


def _module_to_module_connectivity(adjacency: sparse.csr_matrix, labels: np.ndarray) -> pd.DataFrame:
    """Aggregate held-out directed synapse counts between frozen modules."""

    edge = adjacency.tocoo()
    k = int(np.max(labels)) + 1
    frame = pd.DataFrame(
        {
            "presynaptic_module": [f"C{value + 1}" for value in labels[edge.row]],
            "postsynaptic_module": [f"C{value + 1}" for value in labels[edge.col]],
            "synapse_weight": edge.data.astype(float),
        }
    )
    grouped = (
        frame.groupby(["presynaptic_module", "postsynaptic_module"], as_index=False)
        .agg(synapse_weight=("synapse_weight", "sum"), n_nonzero_edges=("synapse_weight", "size"))
    )
    module_order = [f"C{index + 1}" for index in range(k)]
    full = pd.MultiIndex.from_product(
        [module_order, module_order], names=["presynaptic_module", "postsynaptic_module"]
    ).to_frame(index=False)
    full = full.merge(grouped, on=["presynaptic_module", "postsynaptic_module"], how="left").fillna(0.0)
    full["source_total_synapse_weight"] = full.groupby("presynaptic_module")["synapse_weight"].transform("sum")
    full["row_weight_fraction"] = np.divide(
        full["synapse_weight"],
        full["source_total_synapse_weight"],
        out=np.zeros(len(full), dtype=float),
        where=full["source_total_synapse_weight"].to_numpy(dtype=float) > 0.0,
    )
    full["is_within_module"] = full["presynaptic_module"].eq(full["postsynaptic_module"])
    return full


def _module_spatial_summary(nodes: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Summarize observed soma coordinates without using them in discovery."""

    columns = ["x", "y", "z"]
    records: list[dict[str, object]] = []
    values = nodes.loc[:, columns].to_numpy(dtype=float) if all(column in nodes for column in columns) else None
    for module_index in range(int(np.max(labels)) + 1):
        member = labels == module_index
        if values is None:
            records.append(
                {
                    "module_id": f"C{module_index + 1}",
                    "spatial_coordinate_coverage": 0.0,
                    "spatial_coordinate_n": 0,
                    "soma_centroid_x": np.nan,
                    "soma_centroid_y": np.nan,
                    "soma_centroid_z": np.nan,
                    "soma_rms_distance_to_centroid": np.nan,
                }
            )
            continue
        current = values[member]
        finite = np.isfinite(current).all(axis=1)
        observed = current[finite]
        if len(observed):
            centroid = observed.mean(axis=0)
            rms = float(np.sqrt(np.mean(np.sum((observed - centroid[None, :]) ** 2, axis=1))))
        else:
            centroid = np.full(3, np.nan)
            rms = np.nan
        records.append(
            {
                "module_id": f"C{module_index + 1}",
                "spatial_coordinate_coverage": float(finite.mean()),
                "spatial_coordinate_n": int(finite.sum()),
                "soma_centroid_x": float(centroid[0]),
                "soma_centroid_y": float(centroid[1]),
                "soma_centroid_z": float(centroid[2]),
                "soma_rms_distance_to_centroid": rms,
            }
        )
    return pd.DataFrame.from_records(records)


def _soft_structure_summary(assignment: pd.DataFrame) -> pd.DataFrame:
    """Summarize post-bootstrap soft membership without treating it as probability."""

    return (
        assignment.groupby(["structural_module", "soft_structure_status"], dropna=False, as_index=False)
        .agg(
            n_neurons=("node_id", "size"),
            membership_max_mean=("membership_max", "mean"),
            membership_entropy_mean=("membership_entropy_normalized", "mean"),
            bootstrap_consistency_mean=("bootstrap_assignment_consistency", "mean"),
            connector_score_mean=("connector_score", "mean"),
        )
        .sort_values(["structural_module", "soft_structure_status"])
        .reset_index(drop=True)
    )


def _write_figures(
    output_dir: Path,
    dataset_id: str,
    curve: pd.DataFrame,
    adjacency: sparse.csr_matrix,
    labels: np.ndarray,
    nodes: pd.DataFrame,
    assignment: pd.DataFrame,
    module_summary: pd.DataFrame,
    module_to_module: pd.DataFrame,
    old_map: pd.DataFrame | None,
    old_summary: pd.DataFrame | None,
) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    paths: dict[str, Path] = {}
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    curve_path = figures / "KC_K_SELECTION_CURVE.png"
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.2), sharex=True, constrained_layout=True)
    axes[0].plot(curve["k"], curve["heldout_directed_strength_corrected_modularity"], marker="o", label="held-out directed strength-corrected Q")
    axes[0].plot(curve["k"], curve["complexity_adjusted_structure_score"], marker="s", label="complexity-adjusted Q")
    invalid = curve.loc[~curve["valid_minimum_cluster_size"]]
    axes[0].scatter(
        invalid["k"],
        invalid["heldout_directed_strength_corrected_modularity"],
        marker="x",
        color="#7f7f7f",
        s=42,
        label="fails minimum-size gate",
        zorder=4,
    )
    selected = curve.loc[curve["selected_primary_k"]]
    axes[0].scatter(
        selected["k"],
        selected["heldout_directed_strength_corrected_modularity"],
        marker="*",
        color="#111111",
        s=130,
        label="selected K",
        zorder=5,
    )
    axes[0].set_ylabel("structure score")
    axes[0].legend(frameon=False, fontsize=8)
    selected_text = ", ".join(str(int(value)) for value in selected["k"].to_numpy()) or "none"
    axes[0].set_title(f"{dataset_id}: weighted directed KC structure selection (selected K={selected_text})")
    axes[1].plot(
        curve["k"],
        curve["heldout_within_between_per_pair_ratio"],
        marker="o",
        color="#177e89",
        label="within/between mass per pair",
    )
    axes[1].set_ylabel("within / between")
    second = axes[1].twinx()
    second.plot(
        curve["k"],
        curve["seed_stability_ari_mean"],
        marker="s",
        color="#4c78a8",
        label="KMeans-seed ARI",
    )
    bootstrap = curve.loc[curve["bootstrap_stability_tested"]]
    second.plot(
        bootstrap["k"],
        bootstrap["stability_ari_mean"],
        marker="D",
        linestyle="none",
        color="#d95d39",
        label="synapse-bootstrap ARI",
    )
    second.axhline(0.80, color="#d95d39", linestyle="--", linewidth=1)
    second.scatter(
        selected["k"],
        selected["stability_ari_mean"],
        marker="*",
        color="#111111",
        s=110,
        zorder=5,
    )
    second.set_ylabel("stability ARI")
    second.legend(frameon=False, fontsize=8, loc="best")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    axes[1].set_xlabel("number of modules K")
    fig.savefig(curve_path, dpi=240)
    plt.close(fig)
    paths["k_selection_curve"] = curve_path

    order = np.argsort(labels, kind="stable")
    rank = np.empty(len(order), dtype=int)
    rank[order] = np.arange(len(order))
    edge = adjacency.tocoo()
    bins = min(600, len(labels))
    row_bin = np.floor(rank[edge.row] * bins / len(labels)).astype(int)
    col_bin = np.floor(rank[edge.col] * bins / len(labels)).astype(int)
    image = np.zeros((bins, bins), dtype=np.float32)
    np.add.at(image, (row_bin, col_bin), edge.data.astype(np.float32))
    adjacency_path = figures / "KC_WEIGHTED_ADJACENCY_BY_MODULE.png"
    fig, ax = plt.subplots(figsize=(8.4, 7.3), constrained_layout=True)
    ax.imshow(np.log1p(image), cmap="magma", origin="lower", interpolation="nearest")
    running = 0
    for size in module_summary["n_neurons"].to_numpy(dtype=int)[:-1]:
        running += size
        position = running * bins / len(labels)
        ax.axhline(position, color="white", linewidth=0.6, alpha=0.75)
        ax.axvline(position, color="white", linewidth=0.6, alpha=0.75)
    ax.set_title(f"{dataset_id}: KC-to-KC weighted adjacency ordered by structural module")
    ax.set_xlabel("postsynaptic KC rank")
    ax.set_ylabel("presynaptic KC rank")
    fig.savefig(adjacency_path, dpi=240)
    plt.close(fig)
    paths["weighted_adjacency"] = adjacency_path

    module_order = [f"C{index + 1}" for index in range(int(np.max(labels)) + 1)]
    mixing = module_to_module.pivot(
        index="presynaptic_module", columns="postsynaptic_module", values="synapse_weight"
    ).reindex(index=module_order, columns=module_order, fill_value=0.0)
    row_fraction = module_to_module.pivot(
        index="presynaptic_module", columns="postsynaptic_module", values="row_weight_fraction"
    ).reindex(index=module_order, columns=module_order, fill_value=0.0)
    mixing_path = figures / "KC_MODULE_TO_MODULE_WEIGHT_MATRIX.png"
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), constrained_layout=True)
    sns.heatmap(np.log1p(mixing), cmap="magma", square=True, ax=axes[0], cbar_kws={"label": "log(1 + weight)"})
    axes[0].set_title("held-out KC-to-KC weight")
    axes[0].set_xlabel("postsynaptic module")
    axes[0].set_ylabel("presynaptic module")
    sns.heatmap(row_fraction, cmap="crest", vmin=0, vmax=1, annot=True, fmt=".2f", square=True, ax=axes[1], cbar_kws={"label": "outgoing fraction"})
    axes[1].set_title("row-normalized directed weight")
    axes[1].set_xlabel("postsynaptic module")
    axes[1].set_ylabel("presynaptic module")
    fig.savefig(mixing_path, dpi=240)
    plt.close(fig)
    paths["module_to_module_weight"] = mixing_path

    coordinate_columns = ["x", "y", "z"]
    coordinate_values = nodes.loc[:, coordinate_columns].to_numpy(dtype=float)
    finite = np.isfinite(coordinate_values).all(axis=1)
    if finite.any():
        spatial_path = figures / "KC_MODULE_SPATIAL_DISTRIBUTION.png"
        fig = plt.figure(figsize=(13.1, 4.2), constrained_layout=True)
        axes = (
            fig.add_subplot(1, 3, 1, projection="3d"),
            fig.add_subplot(1, 3, 2),
            fig.add_subplot(1, 3, 3),
        )
        palette = sns.color_palette("colorblind", n_colors=len(module_order))
        for module_index, color in enumerate(palette):
            keep = finite & (labels == module_index)
            axes[0].scatter(
                coordinate_values[keep, 0],
                coordinate_values[keep, 1],
                coordinate_values[keep, 2],
                s=4,
                alpha=0.25,
                color=color,
                linewidths=0,
                label=f"C{module_index + 1}",
            )
            axes[1].scatter(
                coordinate_values[keep, 0],
                coordinate_values[keep, 1],
                s=4,
                alpha=0.28,
                color=color,
                linewidths=0,
            )
            axes[2].scatter(
                coordinate_values[keep, 0],
                coordinate_values[keep, 2],
                s=4,
                alpha=0.28,
                color=color,
                linewidths=0,
            )
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].set_zlabel("z")
        axes[0].view_init(elev=22, azim=-55)
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("z")
        axes[0].legend(frameon=False, markerscale=2.5, fontsize=8)
        fig.suptitle(
            f"{dataset_id}: observed soma coordinates by frozen structural module "
            f"({int(finite.sum())}/{len(finite)} complete; source coordinate units)",
            fontsize=10,
        )
        fig.savefig(spatial_path, dpi=240)
        plt.close(fig)
        paths["module_spatial_distribution"] = spatial_path

    soft_path = figures / "KC_SOFT_STRUCTURE_BOUNDARY.png"
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1), constrained_layout=True)
    colors = {
        "core_neuron": "#4c78a8",
        "intermediate_membership": "#f28e2b",
        "boundary_neuron": "#e15759",
    }
    for status, frame in assignment.groupby("soft_structure_status", sort=True):
        axes[0].scatter(
            frame["membership_max"],
            frame["membership_entropy_normalized"],
            s=6,
            alpha=0.32,
            color=colors.get(str(status), "#777777"),
            linewidths=0,
            label=str(status).replace("_", " "),
        )
    axes[0].axvline(0.75, color="black", linewidth=0.8, linestyle="--")
    axes[0].axhline(0.55, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_xlabel("maximum normalized affinity")
    axes[0].set_ylabel("normalized affinity entropy")
    axes[0].set_title("core and boundary assignments")
    axes[0].legend(frameon=False, fontsize=7)
    ordered = assignment.sort_values("connector_score", ascending=False).head(min(300, len(assignment)))
    point_colors = [colors.get(str(value), "#777777") for value in ordered["soft_structure_status"]]
    axes[1].scatter(
        np.arange(len(ordered)),
        ordered["connector_score"],
        s=8,
        alpha=0.70,
        color=point_colors,
        linewidths=0,
    )
    axes[1].set_xlabel("top connector-ranked neurons")
    axes[1].set_ylabel("entropy x (1 - max affinity)")
    axes[1].set_title("boundary/connector score")
    fig.savefig(soft_path, dpi=240)
    plt.close(fig)
    paths["soft_structure_boundary"] = soft_path

    if old_map is not None and old_summary is not None:
        old_path = figures / "OLD5_TO_NEW_PARTITION_MAP.png"
        heat = old_map.pivot(index="old5_cluster", columns="structural_module", values="old5_fraction").fillna(0.0)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [1.25, 1]}, constrained_layout=True)
        sns.heatmap(heat, annot=True, fmt=".2f", cmap="crest", vmin=0, vmax=1, ax=axes[0])
        axes[0].set_title("Legacy five-group overlap with new modules")
        axes[0].set_xlabel("new structure module")
        axes[0].set_ylabel("legacy group")
        plot = old_summary.sort_values("effective_number_of_new_modules")
        axes[1].barh(plot["old5_cluster"], plot["effective_number_of_new_modules"], color="#d95d39")
        axes[1].axvline(1.75, color="black", linestyle="--", linewidth=1)
        axes[1].set_xlabel("effective number of new modules")
        axes[1].set_title("Legacy-group split diagnostic")
        fig.savefig(old_path, dpi=240)
        plt.close(fig)
        paths["old5_to_new"] = old_path
    return paths


def _evaluate_partition_baselines(
    graph: KCWeightedConnectome, evaluation: sparse.csr_matrix, labels: np.ndarray, config: WeightedStructureConfig
) -> pd.DataFrame:
    k = int(labels.max()) + 1
    rows: list[dict[str, object]] = []
    for name, baseline in (("optimal_weighted_directed", labels), ("degree_strength_partition", _degree_partition(evaluation, k, config.random_seed + 7101))):
        metrics, _ = _directed_structure_metrics(evaluation, baseline)
        rows.append({"partition": name, **metrics})
    spatial = _spatial_partition(graph.nodes, k, config.random_seed + 7201)
    if spatial is not None:
        metrics, _ = _directed_structure_metrics(evaluation, spatial)
        rows.append({"partition": "spatial_partition", **metrics})
    rng = np.random.default_rng(config.random_seed + 7301)
    random_values = []
    for _ in range(config.null_repetitions):
        metrics, _ = _directed_structure_metrics(evaluation, rng.permutation(labels))
        random_values.append(metrics["directed_strength_corrected_modularity"])
    rows.append(
        {
            "partition": "random_size_matched_partition_null_mean",
            "directed_strength_corrected_modularity": float(np.mean(random_values)),
            "random_partition_q_sd": float(np.std(random_values, ddof=1)),
            "random_partition_empirical_p_at_least_optimal": float((1 + np.sum(np.asarray(random_values) >= rows[0]["directed_strength_corrected_modularity"])) / (len(random_values) + 1)),
        }
    )
    return pd.DataFrame.from_records(rows)


def analyze_weighted_structure(graph: KCWeightedConnectome, config: WeightedStructureConfig) -> dict[str, Path]:
    """Run held-out K selection, stability, nulls, soft membership, and maps."""

    output_dir = config.output_dir / graph.dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)
    split_rng = np.random.default_rng(config.random_seed + sum(map(ord, graph.dataset_id)))
    discovery, evaluation = _split_weighted_adjacency(graph.adjacency, config.discovery_fraction, split_rng)
    embedding = _embedding(discovery, config.embedding_dimensions, config.random_seed + 101)
    boot_embeddings = _bootstrap_embeddings(discovery, config, sum(map(ord, graph.dataset_id)) * 10)
    weight_perturbation_embeddings = _poisson_weight_perturbation_embeddings(
        discovery, config, sum(map(ord, graph.dataset_id)) * 10
    )
    fits: dict[int, PartitionFit] = {}
    stability_records: list[dict[str, object]] = []
    curve_records: list[dict[str, object]] = []
    for k in range(config.k_min, config.k_max + 1):
        if k >= len(graph.nodes):
            continue
        fit = _fit_partition(embedding, k, config.random_seed + k, n_init=config.kmeans_n_init)
        fits[k] = fit
        metrics, _ = _directed_structure_metrics(evaluation, fit.labels)
        seed_ari: list[float] = []
        seed_nmi: list[float] = []
        for replicate in range(config.seed_repetitions):
            other = _fit_partition(
                embedding,
                k,
                config.random_seed + 1000 + 37 * k + replicate,
                n_init=config.kmeans_n_init,
            )
            agreement, _ = _assignment_consistency(fit.labels, other.labels)
            ari = float(adjusted_rand_score(fit.labels, other.labels))
            nmi = float(normalized_mutual_info_score(fit.labels, other.labels))
            seed_ari.append(ari)
            seed_nmi.append(nmi)
            stability_records.append({"k": k, "stability_kind": "kmeans_seed", "replicate": replicate, "ari": ari, "nmi": nmi, "assignment_agreement": agreement})
        penalty = _complexity_penalty(k, metrics["total_weight"])
        curve_records.append(
            {
                "dataset_id": graph.dataset_id,
                "k": k,
                **{f"heldout_{name}": value for name, value in metrics.items()},
                "complexity_penalty_bic_like": penalty,
                "complexity_adjusted_structure_score": metrics["directed_strength_corrected_modularity"] - penalty,
                "seed_stability_ari_mean": float(np.mean(seed_ari)),
                "seed_stability_nmi_mean": float(np.mean(seed_nmi)),
                "stability_ari_mean": np.nan,
                "stability_ari_min": np.nan,
                "stability_nmi_mean": np.nan,
                "valid_minimum_cluster_size": bool(metrics["minimum_cluster_size"] >= config.minimum_cluster_size),
                "stable_by_ari_gate": False,
                "bootstrap_stability_tested": False,
            }
        )
    curve = pd.DataFrame.from_records(curve_records).sort_values("k").reset_index(drop=True)
    seed_eligible = curve.loc[
        curve["valid_minimum_cluster_size"] & curve["seed_stability_ari_mean"].ge(config.stability_threshold)
    ].copy()
    if seed_eligible.empty:
        bootstrap_candidates = curve.loc[curve["valid_minimum_cluster_size"], "k"].tolist()
    else:
        peak = float(seed_eligible["heldout_directed_strength_corrected_modularity"].max())
        threshold = peak * config.plateau_fraction_of_peak if peak > 0.0 else peak
        bootstrap_candidates = seed_eligible.loc[
            seed_eligible["heldout_directed_strength_corrected_modularity"].ge(threshold), "k"
        ].tolist()
    for k in bootstrap_candidates:
        fit = fits[int(k)]
        boot_ari: list[float] = []
        boot_nmi: list[float] = []
        for replicate, boot_embedding in enumerate(boot_embeddings):
            other = _fit_partition(
                boot_embedding,
                int(k),
                config.random_seed + 2000 + 41 * int(k) + replicate,
                n_init=config.kmeans_n_init,
            )
            agreement, _ = _assignment_consistency(fit.labels, other.labels)
            ari = float(adjusted_rand_score(fit.labels, other.labels))
            nmi = float(normalized_mutual_info_score(fit.labels, other.labels))
            boot_ari.append(ari)
            boot_nmi.append(nmi)
            stability_records.append({"k": int(k), "stability_kind": "synapse_bootstrap", "replicate": replicate, "ari": ari, "nmi": nmi, "assignment_agreement": agreement})
        update = curve["k"].eq(int(k))
        curve.loc[update, "stability_ari_mean"] = float(np.mean(boot_ari))
        curve.loc[update, "stability_ari_min"] = float(np.min(boot_ari))
        curve.loc[update, "stability_nmi_mean"] = float(np.mean(boot_nmi))
        curve.loc[update, "stable_by_ari_gate"] = bool(np.mean(boot_ari) >= config.stability_threshold)
        curve.loc[update, "bootstrap_stability_tested"] = True
    selected_k, selection_status = _select_optimal_k(curve, config)
    curve["selected_primary_k"] = curve["k"].eq(selected_k)
    curve["selection_status"] = selection_status
    candidate_diagnostics = _candidate_objective_diagnostics(curve)
    selected = fits[selected_k]
    selected_metrics, module_summary = _directed_structure_metrics(evaluation, selected.labels)
    module_summary = module_summary.merge(_module_spatial_summary(graph.nodes, selected.labels), on="module_id", how="left")
    module_to_module = _module_to_module_connectivity(evaluation, selected.labels)
    consensus = np.zeros((len(selected.labels), len(selected.labels)), dtype=np.uint8)
    stable_counts = np.zeros(len(selected.labels), dtype=np.int16)
    consensus_labels: list[np.ndarray] = [selected.labels]
    for replicate, boot_embedding in enumerate(boot_embeddings):
        other = _fit_partition(
            boot_embedding,
            selected_k,
            config.random_seed + 9000 + replicate,
            n_init=config.kmeans_n_init,
        )
        _, mapped = _assignment_consistency(selected.labels, other.labels)
        stable_counts += (mapped == selected.labels).astype(np.int16)
        consensus_labels.append(mapped)
    for replicate, perturbed_embedding in enumerate(weight_perturbation_embeddings):
        other = _fit_partition(
            perturbed_embedding,
            selected_k,
            config.random_seed + 30_000 + replicate,
            n_init=config.kmeans_n_init,
        )
        agreement, _ = _assignment_consistency(selected.labels, other.labels)
        stability_records.append(
            {
                "k": selected_k,
                "stability_kind": "poisson_weight_perturbation",
                "replicate": replicate,
                "ari": float(adjusted_rand_score(selected.labels, other.labels)),
                "nmi": float(normalized_mutual_info_score(selected.labels, other.labels)),
                "assignment_agreement": agreement,
                "stability_condition": "Poisson(synapse_count)",
            }
        )
    for dimensions in config.preprocessing_embedding_dimensions:
        if int(dimensions) == int(config.embedding_dimensions):
            continue
        varied_embedding = _embedding(
            discovery,
            int(dimensions),
            config.random_seed + 31_000 + int(dimensions),
        )
        other = _fit_partition(
            varied_embedding,
            selected_k,
            config.random_seed + 32_000 + int(dimensions),
            n_init=config.kmeans_n_init,
        )
        agreement, _ = _assignment_consistency(selected.labels, other.labels)
        stability_records.append(
            {
                "k": selected_k,
                "stability_kind": "preprocessing_embedding_dimensions",
                "replicate": int(dimensions),
                "ari": float(adjusted_rand_score(selected.labels, other.labels)),
                "nmi": float(normalized_mutual_info_score(selected.labels, other.labels)),
                "assignment_agreement": agreement,
                "stability_condition": f"directed_spectral_dimensions={int(dimensions)}",
            }
        )
    available_representations = dict(_edge_representations(discovery))
    for representation_name in config.preprocessing_edge_representations:
        if representation_name not in available_representations:
            raise ValueError(f"unknown preprocessing edge representation {representation_name!r}")
        varied_embedding = _embedding(
            available_representations[representation_name],
            config.embedding_dimensions,
            config.random_seed + 33_000 + sum(map(ord, representation_name)),
        )
        other = _fit_partition(
            varied_embedding,
            selected_k,
            config.random_seed + 34_000 + sum(map(ord, representation_name)),
            n_init=config.kmeans_n_init,
        )
        agreement, _ = _assignment_consistency(selected.labels, other.labels)
        stability_records.append(
            {
                "k": selected_k,
                "stability_kind": "preprocessing_edge_representation",
                "replicate": representation_name,
                "ari": float(adjusted_rand_score(selected.labels, other.labels)),
                "nmi": float(normalized_mutual_info_score(selected.labels, other.labels)),
                "assignment_agreement": agreement,
                "stability_condition": representation_name,
            }
        )
    for labels in consensus_labels:
        consensus += (labels[:, None] == labels[None, :]).astype(np.uint8)
    membership = _soft_membership(selected.labels, selected.embedding, selected.centers, stable_counts, len(boot_embeddings))
    assignment = pd.concat((graph.nodes.reset_index(drop=True), membership), axis=1)
    assignment["connector_score"] = assignment["membership_entropy_normalized"] * (1.0 - assignment["membership_max"])
    soft_summary = _soft_structure_summary(assignment)
    old_map = old_summary = None
    old_labels = _legacy_old5_labels(graph.nodes) if graph.dataset_id == "flywire_v783_kc" else None
    if old_labels is not None:
        old_map, old_summary = _old5_map(old_labels, selected.labels)
    nulls = _null_distribution(evaluation, selected.labels, config, sum(map(ord, graph.dataset_id)) * 100)
    null_summary = (
        nulls.groupby("null_kind", as_index=False)
        .agg(null_mean_q=("q", "mean"), null_sd_q=("q", "std"), repetitions=("q", "size"), observed_q=("observed_q", "first"))
        .sort_values("null_kind")
    )
    p_values = {
        str(kind): float(
            (1 + np.sum(frame["q"].to_numpy(dtype=float) >= float(frame["observed_q"].iloc[0]))) / (len(frame) + 1)
        )
        for kind, frame in nulls.groupby("null_kind", sort=False)
    }
    null_summary["empirical_p_null_at_least_observed"] = null_summary["null_kind"].map(p_values)
    baselines = _evaluate_partition_baselines(graph, evaluation, selected.labels, config)
    representations = _weight_representation_metrics(evaluation, selected.labels)
    stability_frame = pd.DataFrame.from_records(stability_records)
    stability_summary = (
        stability_frame.groupby(["k", "stability_kind"], dropna=False, as_index=False)
        .agg(
            repetitions=("replicate", "size"),
            ari_mean=("ari", "mean"),
            ari_min=("ari", "min"),
            nmi_mean=("nmi", "mean"),
            assignment_agreement_mean=("assignment_agreement", "mean"),
        )
        .sort_values(["k", "stability_kind"])
    )
    paths = {
        "k_selection": output_dir / "kc_partition_score_by_k.csv",
        "candidate_objectives": output_dir / "kc_candidate_objective_diagnostics.csv",
        "stability": output_dir / "kc_partition_stability_by_k.csv",
        "stability_summary": output_dir / "kc_partition_stability_summary.csv",
        "assignments": output_dir / "kc_weighted_structural_assignments.csv",
        "module_summary": output_dir / "kc_weighted_module_connectivity_summary.csv",
        "module_to_module_connectivity": output_dir / "kc_module_to_module_weighted_connectivity.csv",
        "soft_structure_summary": output_dir / "kc_soft_structure_summary.csv",
        "null_draws": output_dir / "kc_weighted_structure_null_draws.csv",
        "null_summary": output_dir / "kc_weighted_structure_null_summary.csv",
        "partition_baselines": output_dir / "kc_optimal_vs_baseline_partitions.csv",
        "weight_representations": output_dir / "kc_selected_partition_weight_representation_metrics.csv",
        "consensus": output_dir / "kc_selected_partition_consensus.npz",
        "metadata": output_dir / "KC_WEIGHTED_STRUCTURE_METADATA.json",
    }
    curve.to_csv(paths["k_selection"], index=False)
    candidate_diagnostics.to_csv(paths["candidate_objectives"], index=False)
    stability_frame.to_csv(paths["stability"], index=False)
    stability_summary.to_csv(paths["stability_summary"], index=False)
    assignment.to_csv(paths["assignments"], index=False)
    module_summary.to_csv(paths["module_summary"], index=False)
    module_to_module.to_csv(paths["module_to_module_connectivity"], index=False)
    soft_summary.to_csv(paths["soft_structure_summary"], index=False)
    nulls.to_csv(paths["null_draws"], index=False)
    null_summary.to_csv(paths["null_summary"], index=False)
    baselines.to_csv(paths["partition_baselines"], index=False)
    representations.to_csv(paths["weight_representations"], index=False)
    np.savez_compressed(paths["consensus"], coassignment_count=consensus, n_partitions=len(consensus_labels), root_id=graph.nodes["node_id"].to_numpy(dtype=np.int64))
    if old_map is not None and old_summary is not None:
        paths["old5_map"] = output_dir / "old5_to_new_partition_overlap.csv"
        paths["old5_summary"] = output_dir / "old5_reinterpretation.csv"
        old_map.to_csv(paths["old5_map"], index=False)
        old_summary.to_csv(paths["old5_summary"], index=False)
    paths.update(
        _write_figures(
            output_dir,
            graph.dataset_id,
            curve,
            evaluation,
            selected.labels,
            graph.nodes,
            assignment,
            module_summary,
            module_to_module,
            old_map,
            old_summary,
        )
    )
    metadata = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "dataset": graph.metadata,
        "discovery_contract": "weighted directed KC-to-KC synapse-count graph only; labels, downstream readouts, odor data, morphology and old clusters excluded from partition discovery",
        "primary_objective": "held-out directed strength-corrected modularity Q = (observed within weight - directed strength-configuration expected within weight) / total held-out weight",
        "optimization": f"directed spectral embedding followed by KMeans with {config.kmeans_n_init} restarts per fitted partition",
        "k_selection_rule": "smallest K that passes minimum-size and bootstrap-ARI gates and reaches the frozen fraction of the maximum eligible held-out Q",
        "selected_k": selected_k,
        "selection_status": selection_status,
        "selected_metrics": selected_metrics,
        "candidate_objective_audit": "The K curve records raw-mass ratios, contrast, density, and strength-normalized enrichment; Q remains primary because it explicitly corrects directed source/target strengths and is assessed on held-out counts.",
        "weight_perturbation": "Poisson perturbation of discovery-half synapse counts for the selected partition; descriptive stability check, not part of K selection.",
        "preprocessing_variation": "Selected partition refit at directed spectral dimensions "
        f"{list(config.preprocessing_embedding_dimensions)} and edge representations "
        f"{list(config.preprocessing_edge_representations)}; descriptive stability checks, not part of K selection.",
        "soft_membership_boundary": "soft values are normalized spectral-geometry affinities, not calibrated biological probabilities",
        "rewire_boundary": "the degree-preserving rewire keeps directed unweighted in/out degrees and the global weight multiset, but not each node's weighted strength",
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return paths


def _cross_dataset_figure(output_dir: Path, curve: pd.DataFrame) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = output_dir / "Fig_cross_mb_k_selection.png"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), constrained_layout=True)
    for dataset_id, frame in curve.groupby("dataset_id", sort=True):
        axes[0].plot(frame["k"], frame["heldout_directed_strength_corrected_modularity"], marker="o", label=dataset_id)
        axes[1].plot(frame["k"], frame["seed_stability_ari_mean"], marker="o", label=dataset_id)
        boot = frame.loc[frame["bootstrap_stability_tested"]]
        axes[1].scatter(boot["k"], boot["stability_ari_mean"], marker="D", s=30, zorder=4)
        selected = frame.loc[frame["selected_primary_k"]]
        axes[0].scatter(
            selected["k"],
            selected["heldout_directed_strength_corrected_modularity"],
            marker="*",
            color="#111111",
            s=95,
            zorder=5,
        )
        axes[1].scatter(selected["k"], selected["stability_ari_mean"], marker="*", color="#111111", s=95, zorder=5)
        invalid = frame.loc[~frame["valid_minimum_cluster_size"]]
        axes[0].scatter(
            invalid["k"],
            invalid["heldout_directed_strength_corrected_modularity"],
            marker="x",
            color="#7f7f7f",
            s=24,
            zorder=4,
        )
    axes[0].set_title("Held-out directed structure objective")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("directed strength-corrected Q")
    axes[1].set_title("Seed stability; diamonds=edge bootstrap; stars=selected")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("mean ARI")
    axes[1].axhline(0.80, color="black", linestyle="--", linewidth=1)
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def _module_connectivity_profile(summary: pd.DataFrame, mixing: pd.DataFrame) -> pd.DataFrame:
    """Make unit-free post-discovery module features for cross-brain matching."""

    profile = summary.copy()
    profile["module_size_fraction"] = profile["n_neurons"] / max(float(profile["n_neurons"].sum()), 1.0)
    self_weight = mixing.loc[mixing["is_within_module"], ["presynaptic_module", "row_weight_fraction"]].rename(
        columns={"presynaptic_module": "module_id", "row_weight_fraction": "within_outgoing_weight_fraction"}
    )
    profile = profile.merge(self_weight, on="module_id", how="left")
    profile["within_outgoing_weight_fraction"] = profile["within_outgoing_weight_fraction"].fillna(0.0)
    features = (
        "module_size_fraction",
        "within_weight_per_possible_pair",
        "outgoing_external_weight_per_possible_pair",
        "incoming_external_weight_per_possible_pair",
        "within_to_outgoing_external_ratio",
        "within_to_incoming_external_ratio",
        "within_outgoing_weight_fraction",
    )
    for column in features:
        profile[f"log1p_{column}"] = np.log1p(profile[column].to_numpy(dtype=float))
    return profile


def _cross_dataset_module_phenotype_matching(
    output_dir: Path, registry: pd.DataFrame
) -> dict[str, Path]:
    """Match module-level connectivity phenotypes across independent stable brains.

    This has no shared-neuron or registration information, so a Hungarian
    match is deliberately reported as a descriptive phenotype correspondence,
    never a homologous-cell-type claim.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    stable = registry.loc[
        registry["selection_status"].eq("SMALLEST_STABLE_K_WITHIN_FROZEN_FRACTION_OF_PEAK_HELDOUT_Q")
    ].copy()
    required = {"flywire_v783_kc", "male_cns_v0_9_kc"}
    available = set(stable["dataset_id"].astype(str))
    exclusions: list[dict[str, object]] = []
    for row in registry.itertuples(index=False):
        dataset = str(row.dataset_id)
        if dataset not in required:
            exclusions.append(
                {
                    "dataset_id": dataset,
                    "included_in_independent_module_matching": False,
                    "reason": "not an independent adult-brain replication relative to FlyWire" if dataset == "banc_v888_kc" else "outside preregistered comparison",
                }
            )
    exclusion_path = output_dir / "CROSS_MB_MODULE_PHENOTYPE_MATCHING_EXCLUSIONS.csv"
    pd.DataFrame.from_records(exclusions).to_csv(exclusion_path, index=False)
    if not required.issubset(available):
        empty_path = output_dir / "CROSS_MB_MODULE_PHENOTYPE_MATCHING.csv"
        pd.DataFrame(
            columns=["reference_dataset", "reference_module", "query_dataset", "query_module", "cosine_similarity", "hungarian_match"]
        ).to_csv(empty_path, index=False)
        return {"cross_module_matching": empty_path, "cross_module_matching_exclusions": exclusion_path}

    profiles: dict[str, pd.DataFrame] = {}
    for dataset in sorted(required):
        dataset_dir = output_dir / dataset
        profiles[dataset] = _module_connectivity_profile(
            pd.read_csv(dataset_dir / "kc_weighted_module_connectivity_summary.csv"),
            pd.read_csv(dataset_dir / "kc_module_to_module_weighted_connectivity.csv"),
        ).sort_values("module_id").reset_index(drop=True)
    feature_columns = [column for column in profiles["flywire_v783_kc"].columns if column.startswith("log1p_")]
    combined = pd.concat(
        [profiles[dataset].assign(dataset_id=dataset) for dataset in sorted(required)], ignore_index=True
    )
    center = combined[feature_columns].mean(axis=0)
    scale = combined[feature_columns].std(axis=0, ddof=0).replace(0.0, 1.0)
    standard = {
        dataset: (profiles[dataset][feature_columns] - center) / scale for dataset in sorted(required)
    }
    reference_dataset = "flywire_v783_kc"
    query_dataset = "male_cns_v0_9_kc"
    reference_values = standard[reference_dataset].to_numpy(dtype=float)
    query_values = standard[query_dataset].to_numpy(dtype=float)
    reference_norm = np.linalg.norm(reference_values, axis=1, keepdims=True)
    query_norm = np.linalg.norm(query_values, axis=1, keepdims=True)
    similarity = (reference_values @ query_values.T) / np.maximum(reference_norm * query_norm.T, 1e-12)
    row, column = linear_sum_assignment(-similarity)
    matches = {(int(left), int(right)) for left, right in zip(row, column, strict=True)}
    records: list[dict[str, object]] = []
    for left in range(similarity.shape[0]):
        for right in range(similarity.shape[1]):
            records.append(
                {
                    "reference_dataset": reference_dataset,
                    "reference_module": str(profiles[reference_dataset].iloc[left]["module_id"]),
                    "query_dataset": query_dataset,
                    "query_module": str(profiles[query_dataset].iloc[right]["module_id"]),
                    "cosine_similarity": float(similarity[left, right]),
                    "hungarian_match": (left, right) in matches,
                    "matching_method": "Hungarian maximum cosine similarity over standardized module connectivity profiles",
                    "interpretation_boundary": "descriptive cross-brain phenotype correspondence; not a cell-type or anatomical homology claim",
                }
            )
    matching = pd.DataFrame.from_records(records)
    matching_path = output_dir / "CROSS_MB_MODULE_PHENOTYPE_MATCHING.csv"
    matching.to_csv(matching_path, index=False)
    profile_path = output_dir / "CROSS_MB_MODULE_CONNECTIVITY_PROFILES.csv"
    pd.concat([profiles[dataset].assign(dataset_id=dataset) for dataset in sorted(required)], ignore_index=True).to_csv(
        profile_path, index=False
    )
    figure_path = output_dir / "Fig_cross_mb_module_phenotype_matching.png"
    fig, ax = plt.subplots(figsize=(5.4, 4.5), constrained_layout=True)
    sns.heatmap(
        similarity,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        vmin=-1,
        vmax=1,
        square=True,
        xticklabels=profiles[query_dataset]["module_id"].tolist(),
        yticklabels=profiles[reference_dataset]["module_id"].tolist(),
        cbar_kws={"label": "cosine similarity"},
        ax=ax,
    )
    for left, right in matches:
        ax.add_patch(plt.Rectangle((right, left), 1, 1, fill=False, edgecolor="black", linewidth=2.0))
    ax.set_xlabel("male CNS module")
    ax.set_ylabel("FlyWire module")
    ax.set_title("connectivity-phenotype correspondence")
    fig.savefig(figure_path, dpi=240)
    plt.close(fig)
    return {
        "cross_module_matching": matching_path,
        "cross_module_profiles": profile_path,
        "cross_module_matching_exclusions": exclusion_path,
        "cross_module_matching_figure": figure_path,
    }


def run_kc_weighted_structure_discovery(config: WeightedStructureConfig | None = None) -> dict[str, Path]:
    """Run independent KC-only weighted structure discovery across datasets."""

    config = config or WeightedStructureConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    registry_rows: list[dict[str, object]] = []
    curves: list[pd.DataFrame] = []
    outputs: dict[str, Path] = {}
    for requested in config.datasets:
        dataset_id = _canonical_dataset_id(requested)
        graph = load_kc_connectome(dataset_id)
        paths = analyze_weighted_structure(graph, config)
        curve = pd.read_csv(paths["k_selection"])
        curves.append(curve)
        selected = curve.loc[curve["selected_primary_k"]].iloc[0]
        registry_context = DATASET_REGISTRY_CONTEXT[dataset_id]
        registry_rows.append(
            {
                "dataset_id": dataset_id,
                "n_kc": graph.metadata["node_count"],
                "n_kc_to_kc_edges": graph.metadata["edge_count"],
                "kc_to_kc_synaptic_weight": graph.metadata["synaptic_weight"],
                "selected_k": int(selected["k"]),
                "selection_status": str(selected["selection_status"]),
                "heldout_directed_strength_corrected_modularity": float(selected["heldout_directed_strength_corrected_modularity"]),
                "bootstrap_ari": float(selected["stability_ari_mean"]),
                "spatial_coordinate_complete_fraction": float(graph.nodes[["x", "y", "z"]].notna().all(axis=1).mean()),
                "spatial_coordinates_available": bool(graph.nodes[["x", "y", "z"]].notna().all(axis=1).mean() >= 0.95),
                "scope": "KC-only mushroom-body subgraph; independent partition discovery",
                **registry_context,
                "independence_boundary": registry_context["independence_role"],
            }
        )
        outputs[f"{dataset_id}_output_dir"] = config.output_dir / dataset_id
    all_curve = pd.concat(curves, ignore_index=True)
    registry = pd.DataFrame.from_records(registry_rows)
    registry_path = config.output_dir / "MB_CONNECTOME_REGISTRY.csv"
    curve_path = config.output_dir / "kc_partition_score_by_k_all_datasets.csv"
    registry.to_csv(registry_path, index=False)
    all_curve.to_csv(curve_path, index=False)
    cross_path = _cross_dataset_figure(config.output_dir, all_curve)
    cross_module_paths = _cross_dataset_module_phenotype_matching(config.output_dir, registry)
    summary_path = config.output_dir / "KC_WEIGHTED_STRUCTURE_CROSS_DATASET_SUMMARY.md"
    lines = [
        "# KC Weighted Structure Discovery",
        "",
        "The primary discovery input is each dataset's weighted directed KC-to-KC graph. No subtype, lobe, odor, downstream, lesion, or old-cluster feature enters partition fitting.",
        "",
        "| dataset | KCs | KC-to-KC edges | selected K | held-out Q | bootstrap ARI | independence boundary |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in registry.itertuples(index=False):
        lines.append(f"| {row.dataset_id} | {int(row.n_kc)} | {int(row.n_kc_to_kc_edges)} | {int(row.selected_k)} | {row.heldout_directed_strength_corrected_modularity:.5f} | {row.bootstrap_ari:.3f} | {row.independence_boundary} |")
    lines.extend(
        [
            "",
            "Selected K is the smallest resolution that meets the minimum-size and bootstrap-stability gates while reaching the predeclared fraction of the maximum eligible held-out directed strength-corrected modularity. It is not a biological ground truth, and cross-dataset numerical K agreement is evaluated descriptively.",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    outputs.update(
        {
            "registry": registry_path,
            "all_k_selection": curve_path,
            "cross_k_figure": cross_path,
            "summary": summary_path,
            **cross_module_paths,
        }
    )
    return outputs


__all__ = [
    "KCWeightedConnectome",
    "PartitionFit",
    "WeightedStructureConfig",
    "_directed_structure_metrics",
    "_select_optimal_k",
    "_soft_membership",
    "analyze_weighted_structure",
    "load_kc_connectome",
    "run_kc_weighted_structure_discovery",
]

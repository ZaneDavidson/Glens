"""
Embedding view builders for GPCR sequence representations.

These functions consume reconstructed residue-level embeddings and produce
fixed-size arrays suitable for ridge, elastic-net, or future region-aware models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from glens.models.embed_model import ResidueEmbeddingResult
from glens.models.regions import RegionMasks

GLOBAL_VIEW_NAMES = ["X_global_mean", "X_global_std", "X_global_mean_std"]

REGION_VIEW_NAMES = [
    "X_7tm_core_mean",
    "X_tm_mean",
    "X_intracellular_mean",
    "X_cytoplasmic_tm_ends_mean",
    "X_tm3_cyt_mean",
    "X_tm5_cyt_mean",
    "X_tm6_cyt_mean",
    "X_tm7_cyt_mean",
    "X_icl_mean",
    "X_icl2_mean",
    "X_icl3_mean",
    "X_h8_mean",
    "X_coupling_face_mean",

    # Concat views.
    "X_tm_cyt_concat",
    "X_loop_concat",
    "X_coupling_face_concat",
    "X_tm3_tm5_icl2_concat",
    "X_tm3_tm5_tm6_icl2_concat",
    "X_tm3_tm5_icl2_h8_concat",
    "X_tm3_tm5_icl2_loop_concat",

    "X_region_concat",
]

CONCAT_VIEW_COMPONENTS: dict[str, list[str]] = {
    # Broad concat views.
    "X_tm_cyt_concat": [
        "X_tm3_cyt_mean",
        "X_tm5_cyt_mean",
        "X_tm6_cyt_mean",
        "X_tm7_cyt_mean",
    ],
    "X_loop_concat": [
        "X_icl2_mean",
        "X_icl3_mean",
        "X_h8_mean",
    ],
    "X_coupling_face_concat": [
        "X_tm3_cyt_mean",
        "X_tm5_cyt_mean",
        "X_tm6_cyt_mean",
        "X_tm7_cyt_mean",
        "X_icl2_mean",
        "X_icl3_mean",
        "X_h8_mean",
    ],

    # Targeted subset views.
    "X_tm3_tm5_icl2_concat": [
        "X_tm3_cyt_mean",
        "X_tm5_cyt_mean",
        "X_icl2_mean",
    ],
    "X_tm3_tm5_tm6_icl2_concat": [
        "X_tm3_cyt_mean",
        "X_tm5_cyt_mean",
        "X_tm6_cyt_mean",
        "X_icl2_mean",
    ],
    "X_tm3_tm5_icl2_h8_concat": [
        "X_tm3_cyt_mean",
        "X_tm5_cyt_mean",
        "X_icl2_mean",
        "X_h8_mean",
    ],
    "X_tm3_tm5_icl2_loop_concat": [
        "X_tm3_cyt_mean",
        "X_tm5_cyt_mean",
        "X_icl2_mean",
        "X_icl3_mean",
        "X_h8_mean",
    ],
}

REGION_CONCAT_COMPONENTS = [
    "X_global_mean",
    "X_7tm_core_mean",
    "X_intracellular_mean",
    "X_icl2_mean",
    "X_icl3_mean",
    "X_h8_mean",
    "X_cytoplasmic_tm_ends_mean",
    "X_coupling_face_mean",
]

REGION_VIEW_TO_MASK = {
    "X_7tm_core_mean": "7tm_core",
    "X_tm_mean": "tm_all",
    "X_intracellular_mean": "intracellular",
    "X_cytoplasmic_tm_ends_mean": "cytoplasmic_tm_ends",

    # individual TM cytoplasmic end views.
    "X_tm3_cyt_mean": "tm3_cyt",
    "X_tm5_cyt_mean": "tm5_cyt",
    "X_tm6_cyt_mean": "tm6_cyt",
    "X_tm7_cyt_mean": "tm7_cyt",

    "X_icl_mean": "icl_all",
    "X_icl2_mean": "ICL2",
    "X_icl3_mean": "ICL3",
    "X_h8_mean": "H8",
    "X_coupling_face_mean": "coupling_face",
}


@dataclass(frozen=True)
class EmbeddingViews:
    arrays: dict[str, np.ndarray]
    metadata: dict[str, object]


def mean_pool(residue_embeddings: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is None:
        return residue_embeddings.mean(axis=0).astype(np.float32, copy=False)

    keep = mask.astype(bool)
    if not np.any(keep):
        raise ValueError("Cannot mean-pool an empty mask.")

    return residue_embeddings[keep].mean(axis=0).astype(np.float32, copy=False)


def std_pool(residue_embeddings: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is None:
        return residue_embeddings.std(axis=0).astype(np.float32, copy=False)

    keep = mask.astype(bool)
    if not np.any(keep):
        raise ValueError("Cannot std-pool an empty mask.")

    return residue_embeddings[keep].std(axis=0).astype(np.float32, copy=False)


def mean_pool_or_zero(
    residue_embeddings: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, bool, int]:
    keep = mask.astype(bool)
    count = int(np.sum(keep))
    if count == 0:
        return (
            np.zeros(residue_embeddings.shape[1], dtype=np.float32),
            True,
            0,
        )
    return mean_pool(residue_embeddings, keep), False, count


def _concat_arrays(arrays: dict[str, np.ndarray], components: list[str]) -> np.ndarray:
    return np.concatenate([arrays[name] for name in components]).astype(
        np.float32,
        copy=False,
    )


def weighted_pool(
    residue_embeddings: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    if residue_embeddings.shape[0] != weights.shape[0]:
        raise ValueError(
            "Residue embedding and weight lengths differ: "
            f"{residue_embeddings.shape[0]} != {weights.shape[0]}"
        )

    weights = weights.astype(np.float32, copy=False)
    total = float(weights.sum())

    if total <= 0:
        raise ValueError("Cannot weighted-pool with non-positive total weight.")

    return (
        (residue_embeddings * weights[:, None]).sum(axis=0) / total
    ).astype(np.float32, copy=False)


def build_global_views(result: ResidueEmbeddingResult) -> EmbeddingViews:
    x_mean = mean_pool(result.residue_embeddings)
    x_std = std_pool(result.residue_embeddings)

    arrays = {
        "X_global_mean": x_mean,
        "X_global_std": x_std,
        "X_global_mean_std": np.concatenate([x_mean, x_std]).astype(
            np.float32,
            copy=False,
        ),
    }

    metadata: dict[str, object] = {
        "view_builder": "global",
        "view_names": list(arrays.keys()),
        "sequence_length": int(result.residue_embeddings.shape[0]),
        "embedding_dim": int(result.residue_embeddings.shape[1]),
        "coverage_min": float(result.coverage.min()),
        "coverage_max": float(result.coverage.max()),
        "coverage_mean": float(result.coverage.mean()),
    }

    return EmbeddingViews(arrays=arrays, metadata=metadata)


def build_region_views(
    result: ResidueEmbeddingResult,
    masks: RegionMasks,
) -> EmbeddingViews:
    """Build fixed-size region-aware views from residue embeddings and masks.

    Empty/missing regions are represented by zero vectors and recorded in
    metadata. This keeps NPZ shapes rectangular while making missingness auditable.
    """
    if masks.sequence_length != result.residue_embeddings.shape[0]:
        raise ValueError(
            f"Region mask length mismatch for {masks.gpcrdb_entry_name}: "
            f"mask length {masks.sequence_length}, residue embeddings "
            f"length {result.residue_embeddings.shape[0]}."
        )

    arrays: dict[str, np.ndarray] = {}
    missing_view_names: list[str] = []
    missing_region_names: list[str] = []
    residue_counts: dict[str, int] = {}

    for view_name, mask_name in REGION_VIEW_TO_MASK.items():
        pooled, missing, count = mean_pool_or_zero(
            result.residue_embeddings,
            masks.masks[mask_name],
        )
        arrays[view_name] = pooled
        residue_counts[mask_name] = count
        if missing:
            missing_view_names.append(view_name)
            missing_region_names.append(mask_name)

    for view_name, components in CONCAT_VIEW_COMPONENTS.items():
        arrays[view_name] = _concat_arrays(arrays, components)

    region_concat_parts: list[np.ndarray] = []
    for component in REGION_CONCAT_COMPONENTS:
        if component == "X_global_mean":
            region_concat_parts.append(mean_pool(result.residue_embeddings))
        else:
            region_concat_parts.append(arrays[component])

    arrays["X_region_concat"] = np.concatenate(region_concat_parts).astype(
        np.float32,
        copy=False,
    )

    metadata: dict[str, object] = {
        "view_builder": "regions",
        "view_names": list(arrays.keys()),
        "sequence_length": int(result.residue_embeddings.shape[0]),
        "embedding_dim": int(result.residue_embeddings.shape[1]),
        "region_view_names": list(REGION_VIEW_NAMES),
        "concat_view_components": {
            name: list(components)
            for name, components in CONCAT_VIEW_COMPONENTS.items()
        },
        "region_concat_components": list(REGION_CONCAT_COMPONENTS),
        "missing_region_names": missing_region_names,
        "missing_view_names": missing_view_names,
        "region_residue_counts": residue_counts,
        "region_source": masks.source,
    }

    return EmbeddingViews(arrays=arrays, metadata=metadata)


def merge_views(*views: EmbeddingViews) -> EmbeddingViews:
    arrays: dict[str, np.ndarray] = {}

    view_builders: list[str] = []
    view_names: list[str] = []

    metadata: dict[str, object] = {
        "view_builder": "merged",
    }

    for view in views:
        overlap = set(arrays).intersection(view.arrays)
        if overlap:
            raise ValueError(f"Duplicate view names while merging: {sorted(overlap)}")

        arrays.update(view.arrays)

        view_builders.append(str(view.metadata.get("view_builder", "unknown")))
        view_names.extend(view.arrays.keys())

        for key in (
            "sequence_length",
            "embedding_dim",
            "coverage_min",
            "coverage_max",
            "coverage_mean",
            "missing_region_names",
            "missing_view_names",
            "region_residue_counts",
            "concat_view_components",
            "region_concat_components",
            "region_source",
        ):
            if key in view.metadata:
                metadata[key] = view.metadata[key]

    metadata["view_builders"] = view_builders
    metadata["view_names"] = view_names

    return EmbeddingViews(arrays=arrays, metadata=metadata)


def stack_view_rows(
    per_receptor_views: list[EmbeddingViews],
    view_names: list[str],
) -> dict[str, np.ndarray]:
    stacked: dict[str, np.ndarray] = {}

    for view_name in view_names:
        stacked[view_name] = np.vstack([
            views.arrays[view_name]
            for views in per_receptor_views
        ]).astype(np.float32, copy=False)

    return stacked


def _metadata_list(
    per_receptor_views: list[EmbeddingViews],
    key: str,
    default: object,
) -> list[object]:
    return [views.metadata.get(key, default) for views in per_receptor_views]


def merge_view_metadata(
    per_receptor_views: list[EmbeddingViews],
) -> dict[str, object]:
    if not per_receptor_views:
        return {}

    view_builders: list[str] = []
    view_names_seen: set[str] = set()
    view_names: list[str] = []

    sequence_lengths: list[int] = []
    coverage_means: list[float] = []

    region_residue_counts: dict[str, list[int]] = {}
    region_missing_counts: dict[str, int] = {}

    component_metadata_keys = (
        "concat_view_components",
        "region_concat_components",
    )

    component_metadata: dict[str, object] = {}

    for view in per_receptor_views:
        builder = view.metadata.get("view_builder", "unknown")
        view_builders.append(str(builder))

        for name in view.arrays:
            if name not in view_names_seen:
                view_names_seen.add(name)
                view_names.append(name)

        sequence_length = view.metadata.get("sequence_length")
        if isinstance(sequence_length, int):
            sequence_lengths.append(sequence_length)

        coverage_mean = view.metadata.get("coverage_mean")
        if isinstance(coverage_mean, int | float):
            coverage_means.append(float(coverage_mean))

        counts_obj = view.metadata.get("region_residue_counts")
        if isinstance(counts_obj, dict):
            for region_name, count in counts_obj.items():
                region = str(region_name)
                region_residue_counts.setdefault(region, []).append(int(count))

                if int(count) == 0:
                    region_missing_counts[region] = region_missing_counts.get(region, 0) + 1

        for key in component_metadata_keys:
            value = view.metadata.get(key)

            if isinstance(value, list) and key not in component_metadata:
                component_metadata[key] = [str(item) for item in value]

            elif isinstance(value, dict) and key not in component_metadata:
                component_metadata[key] = {
                    str(name): [str(item) for item in components]
                    for name, components in value.items()
                }

    metadata: dict[str, object] = {
        "view_builders": sorted(set(view_builders)),
        "view_names": view_names,
    }

    if sequence_lengths:
        metadata.update(
            {
                "sequence_length_min": int(np.min(sequence_lengths)),
                "sequence_length_max": int(np.max(sequence_lengths)),
                "sequence_length_mean": float(np.mean(sequence_lengths)),
            }
        )

    if coverage_means:
        metadata.update(
            {
                "coverage_mean_min": float(np.min(coverage_means)),
                "coverage_mean_max": float(np.max(coverage_means)),
                "coverage_mean_mean": float(np.mean(coverage_means)),
            }
        )

    if region_residue_counts:
        metadata["region_residue_count_summary"] = {
            region: {
                "min": int(np.min(counts)),
                "mean": float(np.mean(counts)),
                "max": int(np.max(counts)),
                "missing": int(region_missing_counts.get(region, 0)),
            }
            for region, counts in sorted(region_residue_counts.items())
        }

    metadata.update(component_metadata)

    return metadata
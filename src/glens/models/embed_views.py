"""
Embedding view builders for GPCR sequence representations.

These functions intentionally stay model-agnostic: they consume reconstructed
residue-level embeddings and produce fixed-size arrays suitable for ridge,
elastic-net, or future region-aware models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from glens.models.embed_model import ResidueEmbeddingResult


@dataclass(frozen=True)
class EmbeddingViews:
    arrays: dict[str, np.ndarray]
    metadata: dict[str, str | int | float | list[str]]


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

    metadata: dict[str, str | int | float | list[str]] = {
        "view_builder": "global",
        "view_names": list(arrays.keys()),
        "sequence_length": int(result.residue_embeddings.shape[0]),
        "embedding_dim": int(result.residue_embeddings.shape[1]),
        "coverage_min": float(result.coverage.min()),
        "coverage_max": float(result.coverage.max()),
        "coverage_mean": float(result.coverage.mean()),
    }

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


def merge_view_metadata(
    per_receptor_views: list[EmbeddingViews],
) -> dict[str, str | int | float | list[str]]:
    if not per_receptor_views:
        return {}

    first = per_receptor_views[0].metadata
    # these casts are bad code, but md should be well-formed. Fix with helpers later if needed.
    lengths = [
        float(cast(int, views.metadata["sequence_length"]))
        for views in per_receptor_views
    ]
    coverage_means = [
        float(cast(float, views.metadata["coverage_mean"]))
        for views in per_receptor_views
    ]

    return {
        "view_builder": str(first["view_builder"]),
        "view_names": list(first["view_names"]),  # type: ignore[arg-type]
        "sequence_length_min": int(np.min(lengths)),
        "sequence_length_max": int(np.max(lengths)),
        "sequence_length_mean": float(np.mean(lengths)),
        "coverage_mean_min": float(np.min(coverage_means)),
        "coverage_mean_max": float(np.max(coverage_means)),
        "coverage_mean_mean": float(np.mean(coverage_means)),
    }
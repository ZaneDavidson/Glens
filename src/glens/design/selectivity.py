"""Selectivity-delta scoring primitives.

These utilities operate on predicted family score vectors in the
canonical family order:

Gs, Gi/o, Gq/11, G12/13

Utilities are blind to prediction sources.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_FAMILY_NAMES: tuple[str, ...] = ("Gs", "Gi/o", "Gq/11", "G12/13")


@dataclass(frozen=True)
class DesignObjective:
    """How to score a mutant relative to WT.

    Parameters
    ----------
    target_family:
        Family whose selectivity should be increased.
    avoid_families:
        Families that should not increase. Positive deltas for these families
        are penalized.
    preserve_families:
        Families whose score should remain close to WT. Absolute deltas are
        penalized.
    target_delta_weight:
        Reward for increasing the target family's raw score.
    margin_weight:
        Reward for increasing target-vs-next-best selectivity margin.
    avoid_increase_weight:
        Penalty for positive deltas in explicitly avoided families.
    off_target_increase_weight:
        Penalty for increasing the strongest non-target family. This catches
        off-target increases even when ``avoid_families`` is empty.
    preserve_change_weight:
        Penalty for changing families that should be preserved.
    uncertainty_weight:
        Optional penalty applied by ensemble summaries to the SD of the design
        score across models.
    """

    target_family: str
    avoid_families: tuple[str, ...] = ()
    preserve_families: tuple[str, ...] = ()
    target_delta_weight: float = 0.5
    margin_weight: float = 1.0
    avoid_increase_weight: float = 1.0
    off_target_increase_weight: float = 0.5
    preserve_change_weight: float = 0.25
    uncertainty_weight: float = 0.5


@dataclass(frozen=True)
class SelectivityDelta:
    """Single-model WT vs. mutant selectivity comparison."""

    family_names: tuple[str, ...]
    target_family: str
    wt_scores: tuple[float, ...]
    mutant_scores: tuple[float, ...]
    delta_scores: tuple[float, ...]
    wt_top_family: str
    mutant_top_family: str
    wt_target_margin: float
    mutant_target_margin: float
    delta_target_margin: float
    delta_target_score: float
    avoid_increase_penalty: float
    off_target_increase_penalty: float
    preserve_change_penalty: float
    design_score: float

    def as_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "target_family": self.target_family,
            "wt_top_family": self.wt_top_family,
            "mutant_top_family": self.mutant_top_family,
            "wt_target_margin": self.wt_target_margin,
            "mutant_target_margin": self.mutant_target_margin,
            "delta_target_margin": self.delta_target_margin,
            "delta_target_score": self.delta_target_score,
            "avoid_increase_penalty": self.avoid_increase_penalty,
            "off_target_increase_penalty": self.off_target_increase_penalty,
            "preserve_change_penalty": self.preserve_change_penalty,
            "design_score": self.design_score,
        }
        for family, wt, mut, delta in zip(
            self.family_names,
            self.wt_scores,
            self.mutant_scores,
            self.delta_scores,
            strict=True,
        ):
            safe = _safe_family_name(family)
            row[f"wt_{safe}"] = wt
            row[f"mutant_{safe}"] = mut
            row[f"delta_{safe}"] = delta
        return row


@dataclass(frozen=True)
class EnsembleDeltaSummary:
    """Consensus summary across multiple WT vs. mutant model predictions."""

    family_names: tuple[str, ...]
    target_family: str
    n_models: int
    mean_design_score: float
    sd_design_score: float
    uncertainty_adjusted_score: float
    mean_delta_target_score: float
    sd_delta_target_score: float
    mean_delta_target_margin: float
    sd_delta_target_margin: float
    target_score_agreement: float
    target_margin_agreement: float
    top_family_flip_agreement: float
    mean_delta_scores: tuple[float, ...]
    sd_delta_scores: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "target_family": self.target_family,
            "n_models": self.n_models,
            "mean_design_score": self.mean_design_score,
            "sd_design_score": self.sd_design_score,
            "uncertainty_adjusted_score": self.uncertainty_adjusted_score,
            "mean_delta_target_score": self.mean_delta_target_score,
            "sd_delta_target_score": self.sd_delta_target_score,
            "mean_delta_target_margin": self.mean_delta_target_margin,
            "sd_delta_target_margin": self.sd_delta_target_margin,
            "target_score_agreement": self.target_score_agreement,
            "target_margin_agreement": self.target_margin_agreement,
            "top_family_flip_agreement": self.top_family_flip_agreement,
        }
        for family, mean_delta, sd_delta in zip(
            self.family_names,
            self.mean_delta_scores,
            self.sd_delta_scores,
            strict=True,
        ):
            safe = _safe_family_name(family)
            row[f"mean_delta_{safe}"] = mean_delta
            row[f"sd_delta_{safe}"] = sd_delta
        return row


def family_index(
    family: str,
    family_names: Sequence[str] = DEFAULT_FAMILY_NAMES,
) -> int:
    """Return idx of a family name in the active family order."""
    try:
        return tuple(family_names).index(family)
    except ValueError as exc:
        valid = ", ".join(family_names)
        raise ValueError(f"Unknown family {family!r}. Expected one of: {valid}") from exc


def top_family(
    scores: Sequence[float],
    family_names: Sequence[str] = DEFAULT_FAMILY_NAMES,
) -> str:
    """Return the family with the largest score."""
    arr = _as_score_vector(scores, family_names=family_names)
    return tuple(family_names)[int(np.argmax(arr))]


def selectivity_margin(
    scores: Sequence[float],
    target_family: str,
    family_names: Sequence[str] = DEFAULT_FAMILY_NAMES,
) -> float:
    """Return target score - argmax(non-target scores)."""
    arr = _as_score_vector(scores, family_names=family_names)
    target_idx = family_index(target_family, family_names)
    non_target = np.delete(arr, target_idx)
    return float(arr[target_idx] - np.max(non_target))


def score_selectivity_delta(
    wt_scores: Sequence[float],
    mutant_scores: Sequence[float],
    objective: DesignObjective,
    family_names: Sequence[str] = DEFAULT_FAMILY_NAMES,
) -> SelectivityDelta:
    """Score delta of mutant's selectivity score vs. WT"""
    names = tuple(family_names)
    wt = _as_score_vector(wt_scores, family_names=names)
    mutant = _as_score_vector(mutant_scores, family_names=names)
    delta = mutant - wt

    target_idx = family_index(objective.target_family, names)
    avoid_indices = _family_indices(objective.avoid_families, names)
    preserve_indices = _family_indices(objective.preserve_families, names)
    non_target_indices = [idx for idx in range(len(names)) if idx != target_idx]

    wt_margin = selectivity_margin(wt.tolist(), objective.target_family, names)
    mutant_margin = selectivity_margin(mutant.tolist(), objective.target_family, names)
    delta_margin = mutant_margin - wt_margin
    delta_target = float(delta[target_idx])

    avoid_penalty = _positive_sum(delta, avoid_indices)
    strongest_off_target_increase = max(
        [0.0, *[float(delta[idx]) for idx in non_target_indices]]
    )
    preserve_penalty = float(np.sum(np.abs(delta[preserve_indices]))) if preserve_indices else 0.0

    design_score = (
        objective.target_delta_weight * delta_target
        + objective.margin_weight * delta_margin
        - objective.avoid_increase_weight * avoid_penalty
        - objective.off_target_increase_weight * strongest_off_target_increase
        - objective.preserve_change_weight * preserve_penalty
    )

    return SelectivityDelta(
        family_names=names,
        target_family=objective.target_family,
        wt_scores=tuple(float(value) for value in wt),
        mutant_scores=tuple(float(value) for value in mutant),
        delta_scores=tuple(float(value) for value in delta),
        wt_top_family=top_family(wt.tolist(), names),
        mutant_top_family=top_family(mutant.tolist(), names),
        wt_target_margin=wt_margin,
        mutant_target_margin=mutant_margin,
        delta_target_margin=float(delta_margin),
        delta_target_score=delta_target,
        avoid_increase_penalty=float(avoid_penalty),
        off_target_increase_penalty=float(strongest_off_target_increase),
        preserve_change_penalty=float(preserve_penalty),
        design_score=float(design_score),
    )


def summarize_ensemble_delta(
    wt_scores_by_model: Sequence[Sequence[float]],
    mutant_scores_by_model: Sequence[Sequence[float]],
    objective: DesignObjective,
    family_names: Sequence[str] = DEFAULT_FAMILY_NAMES,
) -> EnsembleDeltaSummary:
    """Summarize WT vs. mutant selectivity deltas across models."""
    names = tuple(family_names)
    wt = _as_score_matrix(wt_scores_by_model, family_names=names)
    mutant = _as_score_matrix(mutant_scores_by_model, family_names=names)
    if wt.shape != mutant.shape:
        raise ValueError(f"WT and mutant score arrays differ: {wt.shape} != {mutant.shape}")

    single_model = [
        score_selectivity_delta(wt[row_idx], mutant[row_idx], objective, names)
        for row_idx in range(wt.shape[0])
    ]

    design_scores = np.array([item.design_score for item in single_model], dtype=np.float64)
    delta_target_scores = np.array(
        [item.delta_target_score for item in single_model],
        dtype=np.float64,
    )
    delta_target_margins = np.array(
        [item.delta_target_margin for item in single_model],
        dtype=np.float64,
    )
    deltas = mutant - wt

    target_idx = family_index(objective.target_family, names)
    target_wins = mutant[:, target_idx] == np.max(mutant, axis=1)

    mean_design = float(np.mean(design_scores))
    sd_design = _sample_sd(design_scores)

    return EnsembleDeltaSummary(
        family_names=names,
        target_family=objective.target_family,
        n_models=int(wt.shape[0]),
        mean_design_score=mean_design,
        sd_design_score=sd_design,
        uncertainty_adjusted_score=float(
            mean_design - objective.uncertainty_weight * sd_design
        ),
        mean_delta_target_score=float(np.mean(delta_target_scores)),
        sd_delta_target_score=_sample_sd(delta_target_scores),
        mean_delta_target_margin=float(np.mean(delta_target_margins)),
        sd_delta_target_margin=_sample_sd(delta_target_margins),
        target_score_agreement=float(np.mean(delta_target_scores > 0.0)),
        target_margin_agreement=float(np.mean(delta_target_margins > 0.0)),
        top_family_flip_agreement=float(np.mean(target_wins)),
        mean_delta_scores=tuple(float(value) for value in np.mean(deltas, axis=0)),
        sd_delta_scores=tuple(float(value) for value in _sample_sd_by_column(deltas)),
    )


def _family_indices(
    families: Sequence[str],
    family_names: Sequence[str],
) -> list[int]:
    return [family_index(family, family_names) for family in families]


def _positive_sum(values: np.ndarray, indices: Sequence[int]) -> float:
    if not indices:
        return 0.0
    selected = values[list(indices)]
    return float(np.sum(np.maximum(selected, 0.0)))


def _as_score_vector(
    scores: Sequence[float] | np.ndarray,
    *,
    family_names: Sequence[str],
) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D score vector, got shape {arr.shape}.")
    if arr.shape[0] != len(family_names):
        raise ValueError(
            f"Expected {len(family_names)} family scores, got {arr.shape[0]}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("Family scores must be finite.")
    return arr


def _as_score_matrix(
    scores: Sequence[Sequence[float]] | np.ndarray,
    *,
    family_names: Sequence[str],
) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D score matrix, got shape {arr.shape}.")
    if arr.shape[1] != len(family_names):
        raise ValueError(
            f"Expected {len(family_names)} family-score columns, got {arr.shape[1]}."
        )
    if arr.shape[0] == 0:
        raise ValueError("At least one model prediction is required.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Family scores must be finite.")
    return arr


def _sample_sd(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def _sample_sd_by_column(values: np.ndarray) -> np.ndarray:
    if values.shape[0] <= 1:
        return np.zeros(values.shape[1], dtype=np.float64)
    return np.std(values, axis=0, ddof=1)


def _safe_family_name(family: str) -> str:
    return family.replace("/", "_").replace(" ", "_").replace("-", "_")

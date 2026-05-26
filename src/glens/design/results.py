"""Mutation design-result construction.

This module converts mutation candidates plus model-ensemble predictions into
ranked, CSV-ready design records. It intentionally does not load models or
embed sequences. Model-side prediction wiring belongs under ``glens.models``;
this module only consumes prediction-like objects.

The common batch layout is:

    row 0: wild type
    row 1..n: candidate mutants in the same order as the candidates list

but functions also expose explicit row indices for tests and advanced workflows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

from glens.design.candidates import MutationCandidate
from glens.design.selectivity import (
    DesignObjective,
    EnsembleDeltaSummary,
    SelectivityDelta,
    score_selectivity_delta,
    summarize_ensemble_delta,
)

FloatArray: TypeAlias = NDArray[np.float64]


class FamilyEnsemblePredictionLike(Protocol):
    """Minimal prediction interface consumed by design-result code."""

    @property
    def model_names(self) -> tuple[str, ...]:
        """Names of models in ensemble order."""
        ...

    @property
    def family_names(self) -> tuple[str, ...]:
        """Family names in prediction-column order."""
        ...

    @property
    def weights(self) -> tuple[float, ...]:
        """Ensemble weights in model order."""
        ...

    @property
    def weighted_mean(self) -> FloatArray:
        """Weighted mean predictions, shape ``(n_rows, n_families)``."""
        ...

    def scores_by_model_for_row(self, row_index: int) -> FloatArray:
        """Return model-by-family predictions for one row."""
        ...


@dataclass(frozen=True)
class MutationDesignResult:
    """Auditable design result for one mutation candidate."""

    candidate: MutationCandidate
    objective: DesignObjective
    family_names: tuple[str, ...]
    model_names: tuple[str, ...]
    wt_row_index: int
    mutant_row_index: int
    wt_weighted_scores: tuple[float, ...]
    mutant_weighted_scores: tuple[float, ...]
    delta_weighted_scores: tuple[float, ...]
    weighted_delta: SelectivityDelta
    ensemble_summary: EnsembleDeltaSummary
    per_model_deltas: tuple[SelectivityDelta, ...]

    @property
    def rank_score(self) -> float:
        """Primary ranking score for design candidates."""
        return self.ensemble_summary.uncertainty_adjusted_score

    def as_dict(self) -> dict[str, Any]:
        """Return a flat, CSV-friendly result row."""
        row = self.candidate.as_dict()
        row.update(
            {
                "target_family": self.objective.target_family,
                "rank_score": self.rank_score,
                "weighted_design_score": self.weighted_delta.design_score,
                "weighted_wt_top_family": self.weighted_delta.wt_top_family,
                "weighted_mutant_top_family": self.weighted_delta.mutant_top_family,
                "weighted_wt_target_margin": self.weighted_delta.wt_target_margin,
                "weighted_mutant_target_margin": self.weighted_delta.mutant_target_margin,
                "weighted_delta_target_margin": self.weighted_delta.delta_target_margin,
                "weighted_delta_target_score": self.weighted_delta.delta_target_score,
                "weighted_avoid_increase_penalty": (
                    self.weighted_delta.avoid_increase_penalty
                ),
                "weighted_off_target_increase_penalty": (
                    self.weighted_delta.off_target_increase_penalty
                ),
                "weighted_preserve_change_penalty": (
                    self.weighted_delta.preserve_change_penalty
                ),
                "wt_row_index": self.wt_row_index,
                "mutant_row_index": self.mutant_row_index,
            }
        )

        for family, wt, mutant, delta in zip(
            self.family_names,
            self.wt_weighted_scores,
            self.mutant_weighted_scores,
            self.delta_weighted_scores,
            strict=True,
        ):
            safe = _safe_family_name(family)
            row[f"weighted_wt_{safe}"] = wt
            row[f"weighted_mutant_{safe}"] = mutant
            row[f"weighted_delta_{safe}"] = delta

        row.update(_prefixed_dict("ensemble", self.ensemble_summary.as_dict()))
        return row


def build_mutation_design_result(
    *,
    candidate: MutationCandidate,
    wt_prediction: FamilyEnsemblePredictionLike,
    mutant_prediction: FamilyEnsemblePredictionLike,
    objective: DesignObjective,
    mutant_row_index: int,
    wt_row_index: int = 0,
) -> MutationDesignResult:
    """Build one mutation design result from WT and mutant prediction rows.

    ``wt_prediction`` and ``mutant_prediction`` may be the same object when WT
    and mutants are predicted together in one batch.
    """
    _validate_prediction_compatibility(wt_prediction, mutant_prediction)

    family_names = tuple(wt_prediction.family_names)
    wt_weighted = _score_row(wt_prediction.weighted_mean, wt_row_index)
    mutant_weighted = _score_row(mutant_prediction.weighted_mean, mutant_row_index)
    delta_weighted = mutant_weighted - wt_weighted

    wt_by_model = wt_prediction.scores_by_model_for_row(wt_row_index)
    mutant_by_model = mutant_prediction.scores_by_model_for_row(mutant_row_index)
    _validate_model_score_matrices(wt_by_model, mutant_by_model, family_names)

    weighted_delta = score_selectivity_delta(
        _score_vector_to_list(wt_weighted),
        _score_vector_to_list(mutant_weighted),
        objective,
        family_names,
    )
    ensemble_summary = summarize_ensemble_delta(
        _score_matrix_to_nested_list(wt_by_model),
        _score_matrix_to_nested_list(mutant_by_model),
        objective,
        family_names,
    )

    per_model_deltas = tuple(
        score_selectivity_delta(
            _score_vector_to_list(wt_by_model[model_idx]),
            _score_vector_to_list(mutant_by_model[model_idx]),
            objective,
            family_names,
        )
        for model_idx in range(wt_by_model.shape[0])
    )

    return MutationDesignResult(
        candidate=candidate,
        objective=objective,
        family_names=family_names,
        model_names=tuple(wt_prediction.model_names),
        wt_row_index=wt_row_index,
        mutant_row_index=mutant_row_index,
        wt_weighted_scores=_float_tuple(wt_weighted),
        mutant_weighted_scores=_float_tuple(mutant_weighted),
        delta_weighted_scores=_float_tuple(delta_weighted),
        weighted_delta=weighted_delta,
        ensemble_summary=ensemble_summary,
        per_model_deltas=per_model_deltas,
    )


def build_mutation_design_results(
    *,
    candidates: Sequence[MutationCandidate],
    prediction: FamilyEnsemblePredictionLike,
    objective: DesignObjective,
    wt_row_index: int = 0,
    mutant_start_row_index: int = 1,
    sort: bool = True,
) -> tuple[MutationDesignResult, ...]:
    """Build design results for the common WT-plus-mutants batch layout."""
    results = tuple(
        build_mutation_design_result(
            candidate=candidate,
            wt_prediction=prediction,
            mutant_prediction=prediction,
            objective=objective,
            wt_row_index=wt_row_index,
            mutant_row_index=mutant_start_row_index + idx,
        )
        for idx, candidate in enumerate(candidates)
    )

    if sort:
        return rank_mutation_design_results(results)
    return results


def rank_mutation_design_results(
    results: Sequence[MutationDesignResult],
    *,
    descending: bool = True,
) -> tuple[MutationDesignResult, ...]:
    """Rank mutation results by uncertainty-adjusted design score.

    Ties are broken by agreement and then by mean target-margin improvement.
    """
    return tuple(
        sorted(
            results,
            key=lambda result: (
                result.rank_score,
                result.ensemble_summary.target_margin_agreement,
                result.ensemble_summary.mean_delta_target_margin,
            ),
            reverse=descending,
        )
    )


def mutation_results_to_rows(
    results: Sequence[MutationDesignResult],
) -> list[dict[str, Any]]:
    """Convert mutation design results to flat table rows."""
    return [result.as_dict() for result in results]


def per_model_delta_rows(
    results: Sequence[MutationDesignResult],
) -> list[dict[str, Any]]:
    """Return a long-form per-model audit table for mutation deltas."""
    rows: list[dict[str, Any]] = []

    for result in results:
        candidate_row = result.candidate.as_dict()
        for model_name, delta in zip(
            result.model_names,
            result.per_model_deltas,
            strict=True,
        ):
            row = dict(candidate_row)
            row.update(
                {
                    "model_name": model_name,
                    "target_family": result.objective.target_family,
                    "wt_top_family": delta.wt_top_family,
                    "mutant_top_family": delta.mutant_top_family,
                    "wt_target_margin": delta.wt_target_margin,
                    "mutant_target_margin": delta.mutant_target_margin,
                    "delta_target_margin": delta.delta_target_margin,
                    "delta_target_score": delta.delta_target_score,
                    "design_score": delta.design_score,
                    "avoid_increase_penalty": delta.avoid_increase_penalty,
                    "off_target_increase_penalty": delta.off_target_increase_penalty,
                    "preserve_change_penalty": delta.preserve_change_penalty,
                }
            )

            for family, wt, mutant, family_delta in zip(
                delta.family_names,
                delta.wt_scores,
                delta.mutant_scores,
                delta.delta_scores,
                strict=True,
            ):
                safe = _safe_family_name(family)
                row[f"wt_{safe}"] = wt
                row[f"mutant_{safe}"] = mutant
                row[f"delta_{safe}"] = family_delta

            rows.append(row)

    return rows


def _validate_prediction_compatibility(
    wt_prediction: FamilyEnsemblePredictionLike,
    mutant_prediction: FamilyEnsemblePredictionLike,
) -> None:
    if wt_prediction.family_names != mutant_prediction.family_names:
        raise ValueError(
            "WT and mutant predictions use different family_names: "
            f"{wt_prediction.family_names} != {mutant_prediction.family_names}."
        )
    if wt_prediction.model_names != mutant_prediction.model_names:
        raise ValueError(
            "WT and mutant predictions use different model_names: "
            f"{wt_prediction.model_names} != {mutant_prediction.model_names}."
        )
    if len(wt_prediction.weights) != len(wt_prediction.model_names):
        raise ValueError("WT prediction weights do not match model_names.")
    if mutant_prediction.weights != wt_prediction.weights:
        raise ValueError(
            "WT and mutant predictions use different ensemble weights: "
            f"{wt_prediction.weights} != {mutant_prediction.weights}."
        )


def _validate_model_score_matrices(
    wt_by_model: FloatArray,
    mutant_by_model: FloatArray,
    family_names: Sequence[str],
) -> None:
    if wt_by_model.shape != mutant_by_model.shape:
        raise ValueError(
            f"WT and mutant per-model scores differ: "
            f"{wt_by_model.shape} != {mutant_by_model.shape}."
        )
    if wt_by_model.ndim != 2:
        raise ValueError(
            f"Expected per-model score matrices to be 2D, got {wt_by_model.shape}."
        )
    if wt_by_model.shape[1] != len(family_names):
        raise ValueError(
            f"Expected {len(family_names)} family columns, got {wt_by_model.shape[1]}."
        )


def _score_row(scores: FloatArray, row_index: int) -> FloatArray:
    if row_index < 0 or row_index >= scores.shape[0]:
        raise IndexError(f"row_index {row_index} outside n_rows={scores.shape[0]}.")
    row = np.asarray(scores[row_index], dtype=np.float64)
    if row.ndim != 1:
        raise ValueError(f"Expected 1D score row, got shape {row.shape}.")
    if not np.all(np.isfinite(row)):
        raise ValueError("Score row contains non-finite values.")
    return row


def _score_vector_to_list(values: FloatArray) -> list[float]:
    return [float(value) for value in values]


def _score_matrix_to_nested_list(values: FloatArray) -> list[list[float]]:
    return [
        [float(value) for value in row]
        for row in values
    ]


def _float_tuple(values: Sequence[float] | FloatArray) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _prefixed_dict(prefix: str, row: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in row.items()}


def _safe_family_name(family: str) -> str:
    return family.replace("/", "_").replace(" ", "_").replace("-", "_")

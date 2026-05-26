from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from glens.design.candidates import MutationCandidate
from glens.design.mutations import PointMutation
from glens.design.results import (
    FamilyEnsemblePredictionLike,
    build_mutation_design_result,
    build_mutation_design_results,
    mutation_results_to_rows,
    per_model_delta_rows,
    rank_mutation_design_results,
)
from glens.design.selectivity import DesignObjective


@dataclass(frozen=True)
class DummyPrediction:
    model_names: tuple[str, ...]
    family_names: tuple[str, ...]
    weights: tuple[float, ...]
    predictions: np.ndarray

    @property
    def weighted_mean(self) -> np.ndarray:
        weights = np.asarray(self.weights, dtype=np.float64)
        weights = weights / float(np.sum(weights))
        return np.average(self.predictions, axis=0, weights=weights)

    def scores_by_model_for_row(self, row_index: int) -> np.ndarray:
        return self.predictions[:, row_index, :]


def _candidate() -> MutationCandidate:
    return MutationCandidate(
        mutation=PointMutation(sequence_index=1, wt_aa="R", mutant_aa="A"),
        source="unit_test",
        note="",
    )


def _prediction() -> DummyPrediction:
    # rows: WT, good mutant, bad mutant
    predictions = np.array(
        [
            [
                [0.40, 0.35, 0.10, 0.05],
                [0.55, 0.30, 0.10, 0.05],
                [0.42, 0.50, 0.10, 0.05],
            ],
            [
                [0.42, 0.33, 0.10, 0.05],
                [0.50, 0.31, 0.10, 0.05],
                [0.41, 0.49, 0.10, 0.05],
            ],
        ],
        dtype=np.float64,
    )
    return DummyPrediction(
        model_names=("m1", "m2"),
        family_names=("Gs", "Gi/o", "Gq/11", "G12/13"),
        weights=(1.0, 1.0),
        predictions=predictions,
    )


def test_dummy_prediction_satisfies_protocol() -> None:
    prediction: FamilyEnsemblePredictionLike = _prediction()

    assert prediction.model_names == ("m1", "m2")
    assert prediction.weighted_mean.shape == (3, 4)


def test_build_mutation_design_result_exposes_weighted_and_ensemble_scores() -> None:
    result = build_mutation_design_result(
        candidate=_candidate(),
        wt_prediction=_prediction(),
        mutant_prediction=_prediction(),
        objective=DesignObjective(target_family="Gs", avoid_families=("Gi/o",)),
        wt_row_index=0,
        mutant_row_index=1,
    )

    assert result.candidate.label == "R2A"
    assert result.rank_score > 0.0
    assert result.weighted_delta.delta_target_margin > 0.0
    assert result.ensemble_summary.n_models == 2
    assert result.ensemble_summary.target_margin_agreement == pytest.approx(1.0)
    assert len(result.per_model_deltas) == 2

    row = result.as_dict()
    assert row["mutation"] == "R2A"
    assert row["target_family"] == "Gs"
    assert row["weighted_delta_Gs"] > 0.0
    assert row["ensemble_target_margin_agreement"] == pytest.approx(1.0)


def test_build_mutation_design_results_ranks_good_candidate_first() -> None:
    candidates = (
        _candidate(),
        MutationCandidate(
            mutation=PointMutation(sequence_index=2, wt_aa="R", mutant_aa="E"),
            source="unit_test",
        ),
    )

    results = build_mutation_design_results(
        candidates=candidates,
        prediction=_prediction(),
        objective=DesignObjective(target_family="Gs", avoid_families=("Gi/o",)),
    )

    assert [result.candidate.label for result in results] == ["R2A", "R3E"]
    assert results[0].rank_score > results[1].rank_score


def test_rank_mutation_design_results_can_sort_ascending() -> None:
    candidates = (
        _candidate(),
        MutationCandidate(
            mutation=PointMutation(sequence_index=2, wt_aa="R", mutant_aa="E"),
            source="unit_test",
        ),
    )
    results = build_mutation_design_results(
        candidates=candidates,
        prediction=_prediction(),
        objective=DesignObjective(target_family="Gs", avoid_families=("Gi/o",)),
        sort=False,
    )

    ascending = rank_mutation_design_results(results, descending=False)

    assert ascending[0].rank_score < ascending[1].rank_score


def test_mutation_results_to_rows_and_per_model_delta_rows_are_flat_tables() -> None:
    results = build_mutation_design_results(
        candidates=(_candidate(),),
        prediction=_prediction(),
        objective=DesignObjective(target_family="Gs", avoid_families=("Gi/o",)),
    )

    rows = mutation_results_to_rows(results)
    per_model = per_model_delta_rows(results)

    assert len(rows) == 1
    assert rows[0]["mutation"] == "R2A"
    assert len(per_model) == 2
    assert {row["model_name"] for row in per_model} == {"m1", "m2"}
    assert all("delta_target_margin" in row for row in per_model)


def test_prediction_compatibility_checks_family_names() -> None:
    wt = _prediction()
    mutant = DummyPrediction(
        model_names=wt.model_names,
        family_names=("Gs", "Gi/o"),
        weights=wt.weights,
        predictions=wt.predictions[:, :, :2],
    )

    with pytest.raises(ValueError, match="family_names"):
        build_mutation_design_result(
            candidate=_candidate(),
            wt_prediction=wt,
            mutant_prediction=mutant,
            objective=DesignObjective(target_family="Gs"),
            mutant_row_index=1,
        )


def test_prediction_compatibility_checks_model_names() -> None:
    wt = _prediction()
    mutant = DummyPrediction(
        model_names=("other", "m2"),
        family_names=wt.family_names,
        weights=wt.weights,
        predictions=wt.predictions,
    )

    with pytest.raises(ValueError, match="model_names"):
        build_mutation_design_result(
            candidate=_candidate(),
            wt_prediction=wt,
            mutant_prediction=mutant,
            objective=DesignObjective(target_family="Gs"),
            mutant_row_index=1,
        )


def test_row_index_bounds_are_checked() -> None:
    with pytest.raises(IndexError, match="outside n_rows"):
        build_mutation_design_result(
            candidate=_candidate(),
            wt_prediction=_prediction(),
            mutant_prediction=_prediction(),
            objective=DesignObjective(target_family="Gs"),
            mutant_row_index=99,
        )

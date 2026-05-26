import numpy as np
import pytest

from glens.design.selectivity import (
    DesignObjective,
    family_index,
    score_selectivity_delta,
    selectivity_margin,
    summarize_ensemble_delta,
    top_family,
)


def test_selectivity_margin_uses_strongest_non_target() -> None:
    scores = [0.40, 0.35, 0.10, 0.05]

    assert selectivity_margin(scores, "Gs") == pytest.approx(0.05)
    assert selectivity_margin(scores, "Gi/o") == pytest.approx(-0.05)


def test_score_selectivity_delta_rewards_margin_and_penalizes_avoid_increase() -> None:
    objective = DesignObjective(target_family="Gs", avoid_families=("Gi/o",))

    wt = [0.40, 0.35, 0.10, 0.05]
    mutant = [0.50, 0.30, 0.10, 0.05]

    result = score_selectivity_delta(wt, mutant, objective)

    assert result.wt_top_family == "Gs"
    assert result.mutant_top_family == "Gs"
    assert result.delta_target_score == pytest.approx(0.10)
    assert result.wt_target_margin == pytest.approx(0.05)
    assert result.mutant_target_margin == pytest.approx(0.20)
    assert result.delta_target_margin == pytest.approx(0.15)
    assert result.avoid_increase_penalty == pytest.approx(0.0)
    assert result.design_score > 0.0

    row = result.as_dict()
    assert row["delta_Gs"] == pytest.approx(0.10)
    assert row["delta_Gi_o"] == pytest.approx(-0.05)


def test_score_selectivity_delta_penalizes_off_target_increase() -> None:
    objective = DesignObjective(target_family="Gs", avoid_families=("Gi/o",))

    wt = [0.40, 0.35, 0.10, 0.05]
    bad_mutant = [0.45, 0.50, 0.10, 0.05]

    result = score_selectivity_delta(wt, bad_mutant, objective)

    assert result.mutant_top_family == "Gi/o"
    assert result.delta_target_score == pytest.approx(0.05)
    assert result.delta_target_margin < 0.0
    assert result.avoid_increase_penalty == pytest.approx(0.15)
    assert result.off_target_increase_penalty == pytest.approx(0.15)
    assert result.design_score < 0.0


def test_summarize_ensemble_delta_reports_agreement_and_uncertainty() -> None:
    objective = DesignObjective(target_family="Gs", avoid_families=("Gi/o",))
    wt = np.array(
        [
            [0.40, 0.35, 0.10, 0.05],
            [0.42, 0.33, 0.10, 0.05],
            [0.38, 0.36, 0.10, 0.05],
        ]
    )
    mutant = np.array(
        [
            [0.50, 0.30, 0.10, 0.05],
            [0.48, 0.31, 0.10, 0.05],
            [0.39, 0.37, 0.10, 0.05],
        ]
    )

    summary = summarize_ensemble_delta(wt.tolist(), mutant.tolist(), objective)

    assert summary.n_models == 3
    assert summary.mean_delta_target_score > 0.0
    assert summary.target_score_agreement == pytest.approx(1.0)
    assert summary.target_margin_agreement == pytest.approx(2 / 3)
    assert summary.sd_design_score > 0.0
    assert summary.uncertainty_adjusted_score < summary.mean_design_score


def test_invalid_family_raises_helpful_error() -> None:
    with pytest.raises(ValueError, match="Unknown family"):
        family_index("Gz")


def test_top_family_uses_canonical_order() -> None:
    assert top_family([0.1, 0.2, 0.3, 0.0]) == "Gq/11"

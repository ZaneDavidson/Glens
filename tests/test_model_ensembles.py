from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from glens.inference.ensembles import (
    FAMILY_NAMES,
    FamilyEnsemblePrediction,
    FamilyModelSpec,
    feature_matrix_for_model,
    load_family_model_artifact,
    load_family_model_ensemble,
    predict_family_model,
    predict_family_model_ensemble,
    read_family_model_specs,
    required_embedding_keys,
)


class SumPredictor:
    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    def predict(self, X: np.ndarray) -> np.ndarray:
        total = np.sum(X, axis=1) * self.scale
        return np.column_stack(
            [
                total,
                total + 0.1,
                total + 0.2,
                total + 0.3,
            ]
        )


def _dump_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def test_read_family_model_specs_resolves_relative_paths(tmp_path: Path) -> None:
    manifest = tmp_path / "ensemble.json"
    manifest.write_text(
        json.dumps(
            {
                "models": [
                    {"path": "a.joblib", "name": "a", "weight": 2.0},
                    {"path": "/abs/b.joblib", "name": "b"},
                ]
            }
        ),
        encoding="utf-8",
    )

    specs = read_family_model_specs(manifest)

    assert specs[0] == FamilyModelSpec(tmp_path / "a.joblib", "a", 2.0)
    assert specs[1] == FamilyModelSpec(Path("/abs/b.joblib"), "b", 1.0)


def test_load_global_family_model_and_predict(tmp_path: Path) -> None:
    model_path = tmp_path / "global.joblib"
    _dump_payload(
        model_path,
        {
            "model": SumPredictor(scale=0.1),
            "family_names": FAMILY_NAMES,
            "embedding_key": "X_coupling_face_concat",
        },
    )

    artifact = load_family_model_artifact(
        FamilyModelSpec(model_path, name="global", weight=1.5)
    )
    embeddings = {
        "X_coupling_face_concat": np.array([[1.0, 2.0], [3.0, 4.0]]),
    }

    pred = predict_family_model(artifact, embeddings, clip=False)

    assert artifact.name == "global"
    assert artifact.view_kind == "global"
    assert artifact.required_embedding_keys == ("X_coupling_face_concat",)
    assert pred.shape == (2, 4)
    assert pred[0, 0] == pytest.approx(0.3)


def test_load_blockwise_family_model_concatenates_blocks(tmp_path: Path) -> None:
    model_path = tmp_path / "blockwise.joblib"
    _dump_payload(
        model_path,
        {
            "model": SumPredictor(scale=1.0),
            "family_names": FAMILY_NAMES,
            "embedding_blocks": ["X_tm3_cyt_mean", "X_h8_mean"],
        },
    )

    artifact = load_family_model_artifact(model_path)
    embeddings = {
        "X_tm3_cyt_mean": np.array([[1.0], [2.0]]),
        "X_h8_mean": np.array([[10.0, 20.0], [30.0, 40.0]]),
    }

    X = feature_matrix_for_model(embeddings, artifact)
    pred = predict_family_model(artifact, embeddings, clip=False)

    assert artifact.view_kind == "blockwise"
    assert artifact.required_embedding_keys == ("X_tm3_cyt_mean", "X_h8_mean")
    assert X.shape == (2, 3)
    assert pred[0, 0] == pytest.approx(31.0)


def test_predict_family_model_clips_to_unit_interval(tmp_path: Path) -> None:
    model_path = tmp_path / "global.joblib"
    _dump_payload(
        model_path,
        {
            "model": SumPredictor(scale=1.0),
            "family_names": FAMILY_NAMES,
            "embedding_key": "X",
        },
    )
    artifact = load_family_model_artifact(model_path)

    pred = predict_family_model(artifact, {"X": np.array([[10.0]])}, clip=True)

    assert np.all(pred <= 1.0)
    assert np.all(pred >= 0.0)


def test_ensemble_prediction_exposes_tensor_weighted_mean_and_sd(tmp_path: Path) -> None:
    path_a = tmp_path / "a.joblib"
    path_b = tmp_path / "b.joblib"
    _dump_payload(
        path_a,
        {"model": SumPredictor(scale=0.1), "family_names": FAMILY_NAMES, "embedding_key": "X"},
    )
    _dump_payload(
        path_b,
        {"model": SumPredictor(scale=0.2), "family_names": FAMILY_NAMES, "embedding_key": "X"},
    )

    artifacts = load_family_model_ensemble(
        [
            FamilyModelSpec(path_a, name="a", weight=1.0),
            FamilyModelSpec(path_b, name="b", weight=3.0),
        ]
    )
    prediction = predict_family_model_ensemble(
        artifacts,
        {"X": np.array([[1.0, 2.0], [3.0, 4.0]])},
        clip=False,
    )

    assert prediction.predictions.shape == (2, 2, 4)
    assert prediction.model_names == ("a", "b")
    assert prediction.weighted_mean[0, 0] == pytest.approx((0.3 * 1 + 0.6 * 3) / 4)
    assert prediction.sd[0, 0] > 0.0
    assert prediction.scores_by_model_for_row(0).shape == (2, 4)
    assert prediction.as_summary_dict()["n_models"] == 2


def test_ensemble_prediction_normalizes_input_array_and_weights() -> None:
    prediction = FamilyEnsemblePrediction(
        model_names=("a",),
        family_names=FAMILY_NAMES,
        weights=(np.float64(2.0),),
        predictions=np.array([[[0.1, 0.2, 0.3, 0.4]]]),
    )

    assert isinstance(prediction.predictions, np.ndarray)
    assert prediction.predictions.dtype == np.float64
    assert prediction.weights == (2.0,)
    assert prediction.weighted_mean.shape == (1, 4)


def test_required_embedding_keys_returns_unique_sorted_keys(tmp_path: Path) -> None:
    path_a = tmp_path / "a.joblib"
    path_b = tmp_path / "b.joblib"
    _dump_payload(
        path_a,
        {"model": SumPredictor(), "family_names": FAMILY_NAMES, "embedding_key": "X_global"},
    )
    _dump_payload(
        path_b,
        {
            "model": SumPredictor(),
            "family_names": FAMILY_NAMES,
            "embedding_blocks": ["X_h8", "X_tm3"],
        },
    )

    artifacts = load_family_model_ensemble([path_a, path_b])

    assert required_embedding_keys(artifacts) == ("X_global", "X_h8", "X_tm3")


def test_load_family_model_rejects_missing_feature_metadata(tmp_path: Path) -> None:
    path = tmp_path / "bad.joblib"
    _dump_payload(path, {"model": SumPredictor(), "family_names": FAMILY_NAMES})

    with pytest.raises(ValueError, match="missing embedding_key or embedding_blocks"):
        load_family_model_artifact(path)


def test_ensemble_rejects_family_name_mismatch(tmp_path: Path) -> None:
    path_a = tmp_path / "a.joblib"
    path_b = tmp_path / "b.joblib"
    _dump_payload(
        path_a,
        {"model": SumPredictor(), "family_names": FAMILY_NAMES, "embedding_key": "X"},
    )
    _dump_payload(
        path_b,
        {"model": SumPredictor(), "family_names": ("Gs", "Gi/o"), "embedding_key": "X"},
    )

    with pytest.raises(ValueError, match="Family-name mismatch"):
        load_family_model_ensemble([path_a, path_b])

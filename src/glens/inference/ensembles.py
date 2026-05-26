"""Family-model ensemble prediction utilities.

This module owns model-side ensemble wiring. Design code should not need to know
whether a trained model expects one global embedding view or a concatenation of
regional embedding blocks.

Current saved model artifacts from 'family' and 'family-blockwise' contain
enough metadata to infer their required feature view:

* global family models store 'embedding_key'
* blockwise family models store 'embedding_blocks'

The ensemble implemented here is deliberately transparent: it loads trained
models, predicts family scores per model, and exposes the full prediction tensor
plus weighted summaries. It does not perform WT-vs-mutant scoring; that belongs
under 'glens.design'.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeAlias, cast
from numpy.typing import NDArray

import joblib
import numpy as np

FAMILY_NAMES: tuple[str, ...] = ("Gs", "Gi/o", "Gq/11", "G12/13")
FloatArray: TypeAlias = NDArray[np.float64]

EmbeddingTable: TypeAlias = Mapping[str, np.ndarray]


class PredictsFamilyScores(Protocol):
    """Protocol for fitted sklearn-like family score predictors."""

    def predict(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Predict family scores for rows in X."""
        ...

@dataclass(frozen=True)
class FamilyModelSpec:
    """User/manifest-level description of one model in ensemble."""

    path: Path
    name: str | None = None
    weight: float = 1.0

    @classmethod
    def from_mapping(
        cls,
        row: Mapping[str, Any],
        *,
        base_dir: Path | None = None,
    ) -> FamilyModelSpec:
        """Build a spec from a JSON manifest row."""
        raw_path = row.get("path")
        if raw_path is None:
            raise ValueError("Model spec is missing required field 'path'.")

        path = Path(str(raw_path))
        if base_dir is not None and not path.is_absolute():
            path = base_dir / path

        weight = float(row.get("weight", 1.0))
        name_value = row.get("name")
        name = None if name_value is None else str(name_value)

        return cls(path=path, name=name, weight=weight)


@dataclass(frozen=True)
class FamilyModelArtifact:
    """Trained family model and feature-view metadata."""

    name: str
    path: Path
    model: PredictsFamilyScores
    family_names: tuple[str, ...]
    weight: float
    embedding_key: str | None
    embedding_blocks: tuple[str, ...]
    metadata: Mapping[str, Any]

    @property
    def required_embedding_keys(self) -> tuple[str, ...]:
        """NPZ/dict keys required to construct the model's feature matrix."""
        if self.embedding_key is not None:
            return (self.embedding_key,)
        return self.embedding_blocks

    @property
    def view_kind(self) -> str:
        """Human-readable feature-view kind."""
        return "global" if self.embedding_key is not None else "blockwise"


@dataclass(frozen=True)
class FamilyEnsemblePrediction:
    """Per-model and aggregate family score predictions."""

    model_names: tuple[str, ...]
    family_names: tuple[str, ...]
    weights: tuple[float, ...]
    predictions: np.ndarray

    def __post_init__(self) -> None:
        arr = np.asarray(self.predictions, dtype=np.float64)
        weights = tuple(float(weight) for weight in self.weights)
        if arr.ndim != 3:
            raise ValueError(
                "predictions must have shape (n_models, n_rows, n_families)."
            )
        if arr.shape[0] != len(self.model_names):
            raise ValueError("model_names length does not match predictions.")
        if arr.shape[0] != len(weights):
            raise ValueError("weights length does not match predictions.")
        if arr.shape[2] != len(self.family_names):
            raise ValueError("family_names length does not match predictions.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Ensemble predictions must be finite.")
        if any(weight < 0.0 for weight in weights):
            raise ValueError("Ensemble weights must be non-negative.")
        if sum(weights) <= 0.0:
            raise ValueError("At least one ensemble weight must be positive.")

        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "predictions", arr)

    @property
    def n_models(self) -> int:
        return int(self.predictions.shape[0])

    @property
    def n_rows(self) -> int:
        return int(self.predictions.shape[1])

    @property
    def n_families(self) -> int:
        return int(self.predictions.shape[2])

    @property
    def normalized_weights(self) -> FloatArray:
        """Weights normalized to sum to one."""
        weights = np.asarray(self.weights, dtype=np.float64)
        return weights / float(np.sum(weights))

    @property
    def mean(self) -> FloatArray:
        """Unweighted mean predictions, shape '(n_rows, n_families)'."""
        return cast(FloatArray, np.mean(self.predictions, axis=0))

    @property
    def weighted_mean(self) -> FloatArray:
        """Weighted mean predictions, shape '(n_rows, n_families)'."""
        return cast(
            FloatArray,
            np.average(
                self.predictions,
                axis=0,
                weights=self.normalized_weights,
            ),
        )

    @property
    def sd(self) -> FloatArray:
        """Sample SD across models, shape '(n_rows, n_families)'."""
        if self.n_models <= 1:
            return np.zeros((self.n_rows, self.n_families), dtype=np.float64)
        return cast(FloatArray, np.std(self.predictions, axis=0, ddof=1))

    def scores_by_model_for_row(self, row_index: int) -> FloatArray:
        """Return model-by-family predictions for one row."""
        if row_index < 0 or row_index >= self.n_rows:
            raise IndexError(f"row_index {row_index} outside n_rows={self.n_rows}.")
        return cast(FloatArray, self.predictions[:, row_index, :])
    
    def as_summary_dict(self) -> dict[str, Any]:
        """Small metadata summary suitable for JSON reports."""
        return {
            "model_names": list(self.model_names),
            "family_names": list(self.family_names),
            "weights": [float(value) for value in self.weights],
            "normalized_weights": [
                float(value) for value in self.normalized_weights
            ],
            "n_models": self.n_models,
            "n_rows": self.n_rows,
            "n_families": self.n_families,
        }


def read_family_model_specs(path: Path) -> tuple[FamilyModelSpec, ...]:
    """Read an ensemble manifest JSON file.

    Expected shape::

        {
          "models": [
            {"path": "models/a.joblib", "name": "continuous_best", "weight": 1.0},
            {"path": "models/b.joblib", "name": "rank_ref", "weight": 0.5}
          ]
        }

    Relative model paths are resolved relative to manifest's parent.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Model manifest must be a JSON object.")

    rows = payload.get("models")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Model manifest must contain a non-empty 'models' list.")

    specs: list[FamilyModelSpec] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"models[{idx}] must be a JSON object.")
        specs.append(FamilyModelSpec.from_mapping(row, base_dir=path.parent))

    return tuple(specs)


def load_family_model_artifact(spec: FamilyModelSpec | str | Path) -> FamilyModelArtifact:
    """Load one saved family-model artifact from a joblib file."""
    model_spec = _coerce_spec(spec)
    if model_spec.weight < 0.0:
        raise ValueError("Model weight must be non-negative.")

    payload = joblib.load(model_spec.path)
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"Expected {model_spec.path} to contain a mapping payload, "
            f"got {type(payload).__name__}."
        )

    model = payload.get("model")
    if model is None or not hasattr(model, "predict"):
        raise ValueError(f"{model_spec.path} is missing a fitted 'model' with predict().")

    family_names = _coerce_family_names(payload.get("family_names", FAMILY_NAMES))

    embedding_key = payload.get("embedding_key")
    embedding_blocks = payload.get("embedding_blocks")

    if embedding_key is not None and embedding_blocks is not None:
        raise ValueError(
            f"{model_spec.path} has both embedding_key and embedding_blocks; "
            "expected exactly one feature-view description."
        )

    if embedding_key is None and embedding_blocks is None:
        raise ValueError(
            f"{model_spec.path} is missing embedding_key or embedding_blocks metadata."
        )

    block_tuple: tuple[str, ...] = ()
    key_value: str | None = None
    if embedding_key is not None:
        key_value = str(embedding_key)
        if key_value == "":
            raise ValueError("embedding_key must not be empty.")
    else:
        block_tuple = tuple(str(block) for block in cast(Sequence[Any], embedding_blocks))
        if not block_tuple or any(block == "" for block in block_tuple):
            raise ValueError("embedding_blocks must be a non-empty sequence.")

    name = model_spec.name or model_spec.path.stem

    return FamilyModelArtifact(
        name=name,
        path=model_spec.path,
        model=cast(PredictsFamilyScores, model),
        family_names=family_names,
        weight=float(model_spec.weight),
        embedding_key=key_value,
        embedding_blocks=block_tuple,
        metadata=payload,
    )


def load_family_model_ensemble(
    specs: Sequence[FamilyModelSpec | str | Path],
) -> tuple[FamilyModelArtifact, ...]:
    """Load and validate a family-model ensemble."""
    if not specs:
        raise ValueError("At least one model spec is required.")

    artifacts = tuple(load_family_model_artifact(spec) for spec in specs)
    _validate_common_family_names(artifacts)
    if sum(artifact.weight for artifact in artifacts) <= 0.0:
        raise ValueError("At least one model in the ensemble must have positive weight.")

    return artifacts


def feature_matrix_for_model(
    embeddings: EmbeddingTable,
    artifact: FamilyModelArtifact,
) -> np.ndarray:
    """Construct the feature matrix required by one family model."""
    if artifact.embedding_key is not None:
        return _embedding_array(embeddings, artifact.embedding_key)

    arrays = [
        _embedding_array(embeddings, key)
        for key in artifact.embedding_blocks
    ]
    n_rows = int(arrays[0].shape[0])
    for key, array in zip(artifact.embedding_blocks, arrays, strict=True):
        if int(array.shape[0]) != n_rows:
            raise ValueError(
                f"Embedding block {key!r} has {array.shape[0]} rows; "
                f"expected {n_rows}."
            )
    return np.concatenate(arrays, axis=1).astype(np.float32, copy=False)


def predict_family_model(
    artifact: FamilyModelArtifact,
    embeddings: EmbeddingTable,
    *,
    clip: bool = True,
) -> FloatArray:
    """Predict family scores for one loaded family model."""
    X = feature_matrix_for_model(embeddings, artifact)
    pred = np.asarray(artifact.model.predict(X), dtype=np.float64)

    if pred.ndim != 2:
        raise ValueError(
            f"Model {artifact.name!r} returned predictions with shape {pred.shape}; "
            "expected 2D array."
        )
    if pred.shape[0] != X.shape[0]:
        raise ValueError(
            f"Model {artifact.name!r} returned {pred.shape[0]} rows for "
            f"{X.shape[0]} feature rows."
        )
    if pred.shape[1] != len(artifact.family_names):
        raise ValueError(
            f"Model {artifact.name!r} returned {pred.shape[1]} family columns; "
            f"expected {len(artifact.family_names)}."
        )
    if not np.all(np.isfinite(pred)):
        raise ValueError(f"Model {artifact.name!r} produced non-finite predictions.")

    if clip:
        pred = np.clip(pred, 0.0, 1.0)

    return pred


def predict_family_model_ensemble(
    artifacts: Sequence[FamilyModelArtifact],
    embeddings: EmbeddingTable,
    *,
    clip: bool = True,
) -> FamilyEnsemblePrediction:
    """Predict family scores for every model in an ensemble."""
    if not artifacts:
        raise ValueError("At least one loaded artifact is required.")

    _validate_common_family_names(artifacts)
    predictions = [
        predict_family_model(artifact, embeddings, clip=clip)
        for artifact in artifacts
    ]

    first_shape = predictions[0].shape
    for artifact, pred in zip(artifacts, predictions, strict=True):
        if pred.shape != first_shape:
            raise ValueError(
                f"Prediction shape mismatch for model {artifact.name!r}: "
                f"{pred.shape} != {first_shape}."
            )

    return FamilyEnsemblePrediction(
        model_names=tuple(artifact.name for artifact in artifacts),
        family_names=artifacts[0].family_names,
        weights=tuple(float(artifact.weight) for artifact in artifacts),
        predictions=np.stack(predictions, axis=0),
    )


def predict_from_model_specs(
    specs: Sequence[FamilyModelSpec | str | Path],
    embeddings: EmbeddingTable,
    *,
    clip: bool = True,
) -> FamilyEnsemblePrediction:
    """Load an ensemble from specs and predict in one call."""
    artifacts = load_family_model_ensemble(specs)
    return predict_family_model_ensemble(artifacts, embeddings, clip=clip)


def required_embedding_keys(
    artifacts: Sequence[FamilyModelArtifact],
) -> tuple[str, ...]:
    """Return sorted unique embedding keys required by an ensemble."""
    keys: set[str] = set()
    for artifact in artifacts:
        keys.update(artifact.required_embedding_keys)
    return tuple(sorted(keys))


def _coerce_spec(spec: FamilyModelSpec | str | Path) -> FamilyModelSpec:
    if isinstance(spec, FamilyModelSpec):
        return spec
    return FamilyModelSpec(path=Path(spec))


def _coerce_family_names(value: Any) -> tuple[str, ...]:
    names = tuple(str(name) for name in value)
    if not names:
        raise ValueError("family_names must not be empty.")
    if len(set(names)) != len(names):
        raise ValueError(f"family_names contain duplicates: {names}")
    return names


def _validate_common_family_names(artifacts: Sequence[FamilyModelArtifact]) -> None:
    expected = artifacts[0].family_names
    for artifact in artifacts[1:]:
        if artifact.family_names != expected:
            raise ValueError(
                f"Family-name mismatch for {artifact.name!r}: "
                f"{artifact.family_names} != {expected}."
            )


def _embedding_array(embeddings: EmbeddingTable, key: str) -> np.ndarray:
    if key not in embeddings:
        available = ", ".join(sorted(embeddings))
        raise ValueError(
            f"Embedding table is missing key {key!r}. Available keys: {available}"
        )

    array = np.asarray(embeddings[key])
    if array.ndim != 2:
        raise ValueError(f"Embedding key {key!r} must be 2D, got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Embedding key {key!r} contains non-finite values.")

    return array.astype(np.float32, copy=False)

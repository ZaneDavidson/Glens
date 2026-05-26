"""Inference-time utilities for trained Glens models.

This package owns model artifact loading, prediction I/O, and ensemble
prediction. Training code belongs under 'glens.models'; mutation-design
scoring belongs under 'glens.design'.
"""

from .ensembles import (
    FAMILY_NAMES,
    FamilyEnsemblePrediction,
    FamilyModelArtifact,
    FamilyModelSpec,
    feature_matrix_for_model,
    load_family_model_artifact,
    load_family_model_ensemble,
    predict_family_model,
    predict_family_model_ensemble,
    predict_from_model_specs,
    read_family_model_specs,
    required_embedding_keys,
)
from .prediction_io import (
    load_embedding_npz,
    load_ensemble_prediction_npz,
)

__all__ = [
    "FAMILY_NAMES",
    "FamilyEnsemblePrediction",
    "FamilyModelArtifact",
    "FamilyModelSpec",
    "feature_matrix_for_model",
    "load_embedding_npz",
    "load_ensemble_prediction_npz",
    "load_family_model_artifact",
    "load_family_model_ensemble",
    "predict_family_model",
    "predict_family_model_ensemble",
    "predict_from_model_specs",
    "read_family_model_specs",
    "required_embedding_keys",
]
"""I/O helpers for precomputed model inputs and predictions.

* load an embedding NPZ for already-computed WT/mutant sequence feature rows
* load a precomputed ensemble prediction NPZ for direct testing

"""

# TODO: On-demand sequence embedding will plug in later by producing the same embedding
# table or prediction object and feeding to existing utilities.

from pathlib import Path
from typing import Any

import numpy as np

from glens.inference.ensembles import FAMILY_NAMES, FamilyEnsemblePrediction

ArrayDict = dict[str, np.ndarray[Any, Any]]


def load_embedding_npz(path: Path) -> ArrayDict:
    """Load all arrays from an embedding NPZ.

    The returned mapping may contain non-feature metadata arrays. Model ensemble
    prediction selects only the keys required by its artifacts.
    """
    arrays: ArrayDict = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            arrays[key] = np.asarray(data[key])
    return arrays


def load_ensemble_prediction_npz(path: Path) -> FamilyEnsemblePrediction:
    """Load a precomputed ensemble prediction NPZ.

    Required key
    ------------
    predictions:
        Float array with shape '(n_models, n_rows, n_families)'.

    Optional info keys
    -------------
    model_names:
        String array of length 'n_models'. Defaults to 'model_0...'.
    family_names:
        String array of length 'n_families'. Defaults to canonical families.
    weights:
        Float array of length 'n_models'. Defaults to all ones.
    """
    with np.load(path, allow_pickle=False) as data:
        if "predictions" not in data.files:
            raise ValueError(f"{path} is missing required key 'predictions'.")

        predictions = np.asarray(data["predictions"], dtype=np.float64)
        if predictions.ndim != 3:
            raise ValueError(
                f"'predictions' must be 3D, got shape {predictions.shape}."
            )

        n_models = int(predictions.shape[0])
        n_families = int(predictions.shape[2])

        model_names = (
            _string_tuple(np.asarray(data["model_names"]))
            if "model_names" in data.files
            else tuple(f"model_{idx}" for idx in range(n_models))
        )
        family_names = (
            _string_tuple(np.asarray(data["family_names"]))
            if "family_names" in data.files
            else FAMILY_NAMES
        )
        weights = (
            _float_tuple(np.asarray(data["weights"], dtype=np.float64))
            if "weights" in data.files
            else tuple(1.0 for _ in range(n_models))
        )

    if len(model_names) != n_models:
        raise ValueError(
            f"model_names has length {len(model_names)}; expected {n_models}."
        )
    if len(family_names) != n_families:
        raise ValueError(
            f"family_names has length {len(family_names)}; expected {n_families}."
        )
    if len(weights) != n_models:
        raise ValueError(f"weights has length {len(weights)}; expected {n_models}.")

    return FamilyEnsemblePrediction(
        model_names=model_names,
        family_names=family_names,
        weights=weights,
        predictions=predictions,
    )


def _string_tuple(values: np.ndarray[Any, Any]) -> tuple[str, ...]:
    flat = np.asarray(values).reshape(-1).tolist()
    return tuple(str(value) for value in flat)


def _float_tuple(values: np.ndarray[Any, Any]) -> tuple[float, ...]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1).tolist()
    return tuple(float(value) for value in flat)

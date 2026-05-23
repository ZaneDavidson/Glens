"""
Family-level GPCR-G protein selectivity fitting.

This module builds receptor-level family targets from the GPCRdb common coupling
map, joins them to an ESM embedding table, fits a RidgeCV with cross-validation,
and writes model, metrics, predictions, and optional training-tables.
"""

import csv
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import typer
from sklearn.base import clone
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

app = typer.Typer(no_args_is_help=True)

FAMILY_NAMES = ("Gs", "Gi/o", "Gq/11", "G12/13")
UNIPROT_ENTRY_RE = re.compile(r"^[a-z0-9]+_[a-z0-9]+$")
GPCRDB_ENTRY_RE = re.compile(r"/protein/([^/?#]+)")
DEFAULT_ALPHAS = np.logspace(-3, 4, 200).tolist()


class CVStrategy(str, Enum):
    KFOLD = "kfold"
    GROUP_KFOLD = "groupkfold"


class CVGroup(str, Enum):
    RECEPTOR_FAMILY = "receptor_family"
    GPCR_CLASS = "gpcr_class"


class RidgeMode(str, Enum):
    SHARED_ALPHA = "shared_alpha"
    ALPHA_PER_TARGET = "alpha_per_target"
    INDEPENDENT_TARGETS = "independent_targets"


class ReducerKind(str, Enum):
    NONE = "none"
    PCA = "pca"
    TRUNCATED_SVD = "truncated_svd"


@dataclass(frozen=True)
class FamilyTargets:
    gpcrdb_entry_name: np.ndarray
    y: np.ndarray
    mask: np.ndarray
    primary_family: np.ndarray
    receptor_family: np.ndarray
    gpcr_class: np.ndarray
    label_source: str


@dataclass(frozen=True)
class JoinedFamilyTable:
    X: np.ndarray
    y: np.ndarray
    mask: np.ndarray
    gpcrdb_entry_name: np.ndarray
    primary_family: np.ndarray
    receptor_family: np.ndarray
    gpcr_class: np.ndarray


@dataclass(frozen=True)
class CVPredictionResult:
    y_pred: np.ndarray
    baseline_predictions: dict[str, np.ndarray]
    fold_metrics: list[dict[str, Any]]


def _clean_header(value: str) -> str:
    return value.strip().lstrip("\ufeff")


def _fill_group_headers(row: Sequence[str]) -> list[str]:
    filled: list[str] = []
    current = ""
    for value in row:
        clean = _clean_header(str(value)) if value is not None else ""
        if clean and clean.lower() != "nan":
            current = clean
        filled.append(current)
    return filled


def _parse_numeric(value: Any) -> float:
    if value is None:
        return np.nan
    text = str(value).strip()
    if text in {"", "-", "nan", "NaN"}:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def _entry_name(value: str) -> str | None:
    match = GPCRDB_ENTRY_RE.search(value)
    entry = (match.group(1) if match else value).strip().lower()
    return entry if UNIPROT_ENTRY_RE.match(entry) else None


def _iter_map_rows(path: Path) -> Iterator[dict[tuple[str, str], str]]:
    """Yield rows keyed by (group header, subheader) from the two-row map header."""
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        group_row = next(reader, None)
        name_row = next(reader, None)
        if group_row is None or name_row is None:
            return

        group_headers = _fill_group_headers(group_row)
        subheaders = [_clean_header(cell) for cell in name_row]
        keys = list(zip(group_headers, subheaders, strict=False))

        for row in reader:
            yield {
                key: row[idx].strip() if idx < len(row) else ""
                for idx, key in enumerate(keys)
            }


def build_family_targets(
    coupling_map_csv: Path,
    *,
    source_lab: str = "GproteinDb",
    source_biosensor: str = "merged data",
) -> FamilyTargets:
    """Build receptor-level family target matrix from merged coupling-map rows.

    Targets come from the `% of 1' G protein family` columns and are scaled from
    0-100 to 0-1. Missing values remain NaN in y and False in mask.
    """
    ids: list[str] = []
    y_rows: list[list[float]] = []
    mask_rows: list[list[bool]] = []
    primary: list[str] = []
    receptor_families: list[str] = []
    gpcr_classes: list[str] = []
    seen: set[str] = set()

    for row in _iter_map_rows(coupling_map_csv):
        if row.get(("Source", "Lab")) != source_lab:
            continue
        if row.get(("Source", "Biosensor")) != source_biosensor:
            continue

        gpcrdb = row.get(("Receptor", "GPCRdb"), "")
        entry = _entry_name(gpcrdb)
        if entry is None or entry in seen:
            continue

        values = [
            _parse_numeric(row.get(("% of 1' G protein family", family)))
            for family in FAMILY_NAMES
        ]
        mask = [np.isfinite(value) for value in values]
        if not any(mask):
            continue

        # Do not map 0's to NaN!!
        scaled = [value / 100.0 if np.isfinite(value) else np.nan for value in values]
        ids.append(entry)
        y_rows.append(scaled)
        mask_rows.append(mask)
        primary.append(row.get(("Selectivity", "Primary family"), ""))
        receptor_families.append(row.get(("Receptor", "Rec. family"), ""))
        gpcr_classes.append(row.get(("Receptor", "Cl"), ""))
        seen.add(entry)

    if not y_rows:
        raise ValueError("No usable family targets were found in the coupling map.")

    return FamilyTargets(
        gpcrdb_entry_name=np.array(ids, dtype=str),
        y=np.array(y_rows, dtype=np.float32),
        mask=np.array(mask_rows, dtype=bool),
        primary_family=np.array(primary, dtype=str),
        receptor_family=np.array(receptor_families, dtype=str),
        gpcr_class=np.array(gpcr_classes, dtype=str),
        label_source=f"{source_lab}/{source_biosensor}",
    )


def load_embedding_artifact(
    path: Path,
    *,
    embedding_key: str = "X",
) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)

    required = {embedding_key, "gpcrdb_entry_name"}
    missing = required.difference(data.files)
    if missing:
        available = ", ".join(data.files)
        raise ValueError(
            f"Embedding artifact missing keys: {sorted(missing)}. "
            f"Available keys: {available}"
        )

    payload = {key: data[key] for key in data.files}

    # Fix the selected view to the key used downstream.
    payload["X"] = data[embedding_key]

    return payload


def join_embeddings_to_targets(
    embeddings: dict[str, np.ndarray],
    targets: FamilyTargets,
) -> JoinedFamilyTable:
    """Inner-join embedding table to labels by gpcrdb_entry_name."""
    emb_ids = embeddings["gpcrdb_entry_name"].astype(str)
    emb_index = {entry: idx for idx, entry in enumerate(emb_ids)}

    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    joined_ids: list[str] = []
    primary_rows: list[str] = []
    receptor_family_rows: list[str] = []
    gpcr_class_rows: list[str] = []

    for idx, entry in enumerate(targets.gpcrdb_entry_name.astype(str)):
        emb_idx = emb_index.get(entry)
        if emb_idx is None:
            continue
        x_rows.append(embeddings["X"][emb_idx])
        y_rows.append(targets.y[idx])
        mask_rows.append(targets.mask[idx])
        joined_ids.append(entry)
        primary_rows.append(targets.primary_family[idx])
        receptor_family_rows.append(targets.receptor_family[idx])
        gpcr_class_rows.append(targets.gpcr_class[idx])

    if not x_rows:
        raise ValueError("No overlap between embedding IDs and family target IDs.")

    return JoinedFamilyTable(
        X=np.vstack(x_rows).astype(np.float32, copy=False),
        y=np.vstack(y_rows).astype(np.float32, copy=False),
        mask=np.vstack(mask_rows).astype(bool, copy=False),
        gpcrdb_entry_name=np.array(joined_ids, dtype=str),
        primary_family=np.array(primary_rows, dtype=str),
        receptor_family=np.array(receptor_family_rows, dtype=str),
        gpcr_class=np.array(gpcr_class_rows, dtype=str),
    )


def _subset_joined_table(table: JoinedFamilyTable, keep: np.ndarray) -> JoinedFamilyTable:
    return JoinedFamilyTable(
        X=table.X[keep],
        y=table.y[keep],
        mask=table.mask[keep],
        gpcrdb_entry_name=table.gpcrdb_entry_name[keep],
        primary_family=table.primary_family[keep],
        receptor_family=table.receptor_family[keep],
        gpcr_class=table.gpcr_class[keep],
    )


def _complete_case_family_targets(table: JoinedFamilyTable) -> JoinedFamilyTable:
    keep = np.asarray(table.mask.all(axis=1) & np.isfinite(table.y).all(axis=1))
    if not np.any(keep):
        raise ValueError("No rows have complete family targets.")
    return _subset_joined_table(table, keep)


def _nonzero_target_mask(y: np.ndarray) -> np.ndarray:
    return np.isfinite(y).all(axis=1) & (np.nansum(y, axis=1) > 0.0)


def _exclude_zero_target_rows(table: JoinedFamilyTable) -> tuple[JoinedFamilyTable, int]:
    keep = _nonzero_target_mask(table.y)
    n_excluded = int(table.y.shape[0] - np.sum(keep))
    if not np.any(keep):
        raise ValueError("All complete-case rows have zero family targets.")
    return _subset_joined_table(table, keep), n_excluded


def _top1_evaluable_mask(y_true: np.ndarray) -> np.ndarray:
    """Rows with an interpretable top family."""
    return _nonzero_target_mask(y_true)


def _top1_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    evaluable = _top1_evaluable_mask(y_true)
    if not np.any(evaluable):
        return None
    return float(
        np.mean(
            np.argmax(y_true[evaluable], axis=1)
            == np.argmax(y_pred[evaluable], axis=1)
        )
    )


def _family_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    metrics: dict[str, Any] = {"per_family": {}}
    for idx, family in enumerate(FAMILY_NAMES):
        yt = y_true[:, idx]
        yp = y_pred[:, idx]
        metrics["per_family"][family] = {
            "mae": float(mean_absolute_error(yt, yp)),
            "rmse": float(mean_squared_error(yt, yp) ** 0.5),
            "r2": float(r2_score(yt, yp)),
        }

    top1_evaluable = _top1_evaluable_mask(y_true)
    metrics["overall"] = {
        "mae_macro": float(mean_absolute_error(y_true, y_pred)),
        "rmse_macro": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2_variance_weighted": float(
            r2_score(y_true, y_pred, multioutput="variance_weighted")
        ),
        "top1_family_accuracy": _top1_accuracy(y_true, y_pred),
        "top1_n_evaluable_rows": int(np.sum(top1_evaluable)),
        "top1_n_masked_rows": int(y_true.shape[0] - np.sum(top1_evaluable)),
    }
    return metrics


def _write_predictions_csv(
    path: Path,
    ids: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_true: np.ndarray | None = None,
    extra_columns: Mapping[str, np.ndarray] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["gpcrdb_entry_name"]
    if extra_columns is not None:
        fields.extend(extra_columns.keys())
    fields.extend(f"pred_{family}" for family in FAMILY_NAMES)
    if y_true is not None:
        fields.extend(f"true_{family}" for family in FAMILY_NAMES)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row_idx, entry in enumerate(ids.astype(str)):
            row: dict[str, str | float] = {"gpcrdb_entry_name": entry}
            if extra_columns is not None:
                row.update({
                    name: str(values[row_idx])
                    for name, values in extra_columns.items()
                })
            row.update({
                f"pred_{family}": float(y_pred[row_idx, family_idx])
                for family_idx, family in enumerate(FAMILY_NAMES)
            })
            if y_true is not None:
                row.update({
                    f"true_{family}": float(y_true[row_idx, family_idx])
                    for family_idx, family in enumerate(FAMILY_NAMES)
                })
            writer.writerow(row)


def _write_npz_targets(path: Path, targets: FamilyTargets) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        gpcrdb_entry_name=targets.gpcrdb_entry_name,
        y_family=targets.y,
        family_mask=targets.mask,
        family_names=np.array(FAMILY_NAMES, dtype=str),
        primary_family=targets.primary_family,
        receptor_family=targets.receptor_family,
        gpcr_class=targets.gpcr_class,
        label_source=np.array(targets.label_source, dtype=str),
    )


def _write_training_table_npz(path: Path, table: JoinedFamilyTable) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=table.X.astype(np.float32, copy=False),
        y_family=table.y.astype(np.float32, copy=False),
        gpcrdb_entry_name=table.gpcrdb_entry_name.astype(str),
        family_names=np.array(FAMILY_NAMES, dtype=str),
        primary_family=table.primary_family.astype(str),
        receptor_family=table.receptor_family.astype(str),
        gpcr_class=table.gpcr_class.astype(str),
    )


def _make_reducer(
    *,
    reducer: ReducerKind,
    reducer_components: int | None,
    random_state: int,
    svd_n_iter: int,
) -> Any | None:
    if reducer == ReducerKind.NONE:
        if reducer_components is not None:
            raise ValueError(
                "--reducer-components should only be passed when "
                "--reducer is pca or truncated_svd."
            )
        return None

    if reducer_components is None:
        raise ValueError(
            "--reducer-components is required when --reducer is "
            "pca or truncated_svd."
        )

    if reducer_components <= 0:
        raise ValueError("--reducer-components must be positive.")

    if reducer == ReducerKind.PCA:
        return PCA(
            n_components=reducer_components,
            svd_solver="randomized",
            random_state=random_state,
        )

    if reducer == ReducerKind.TRUNCATED_SVD:
        return TruncatedSVD(
            n_components=reducer_components,
            algorithm="randomized",
            n_iter=svd_n_iter,
            random_state=random_state,
        )

    raise ValueError(f"Unsupported reducer: {reducer}")


def _make_model(
    alphas: Sequence[float],
    *,
    ridge_mode: RidgeMode,
    reducer: ReducerKind,
    reducer_components: int | None,
    random_state: int,
    svd_n_iter: int,
) -> Any:
    alpha_grid = np.array(alphas, dtype=np.float64)

    if ridge_mode == RidgeMode.SHARED_ALPHA:
        regressor = RidgeCV(alphas=alpha_grid)
    elif ridge_mode == RidgeMode.ALPHA_PER_TARGET:
        regressor = RidgeCV(alphas=alpha_grid, alpha_per_target=True)
    elif ridge_mode == RidgeMode.INDEPENDENT_TARGETS:
        regressor = MultiOutputRegressor(RidgeCV(alphas=alpha_grid))
    else:
        raise ValueError(f"Unsupported ridge mode: {ridge_mode}")

    reducer_step = _make_reducer(
        reducer=reducer,
        reducer_components=reducer_components,
        random_state=random_state,
        svd_n_iter=svd_n_iter,
    )

    steps: list[Any] = [StandardScaler()]

    if reducer_step is not None:
        steps.append(reducer_step)

    steps.append(regressor)

    return make_pipeline(*steps)


def _cv_group_values(table: JoinedFamilyTable, group_by: CVGroup) -> np.ndarray:
    if group_by == CVGroup.RECEPTOR_FAMILY:
        values = table.receptor_family
    elif group_by == CVGroup.GPCR_CLASS:
        values = table.gpcr_class
    else:
        raise ValueError(f"Unsupported CV grouping: {group_by}")

    return np.array([str(value).strip() or "unknown" for value in values], dtype=str)


def _summarize_strings(values: np.ndarray, *, top_n: int = 10) -> dict[str, Any]:
    if values.size == 0:
        return {"n_groups": 0, "largest_groups": []}
    unique, counts = np.unique(values.astype(str), return_counts=True)
    order = np.argsort(counts)[::-1]

    return {
        "n_groups": int(unique.size),
        "largest_groups": [
            {"group": str(unique[idx]), "n": int(counts[idx])}
            for idx in order[:top_n]
        ],
    }


def _make_cv_splitter(
    *,
    cv_strategy: CVStrategy,
    cv_splits: int,
    random_state: int,
    table: JoinedFamilyTable,
    group_by: CVGroup,
) -> tuple[Any, np.ndarray | None, dict[str, Any] | None]:
    if cv_strategy == CVStrategy.KFOLD:
        return (
            KFold(n_splits=cv_splits, shuffle=True, random_state=random_state),
            None,
            None,
        )

    groups = _cv_group_values(table, group_by)
    group_summary = _summarize_strings(groups)

    if group_summary["n_groups"] < cv_splits:
        raise ValueError(
            f"Need at least cv_splits={cv_splits} unique groups for GroupKFold, "
            f"got {group_summary['n_groups']}."
        )

    return GroupKFold(n_splits=cv_splits), groups, group_summary


def _validate_reducer_components(
    *,
    reducer: ReducerKind,
    reducer_components: int | None,
    table: JoinedFamilyTable,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any] | None:
    if reducer == ReducerKind.NONE:
        return None

    if reducer_components is None:
        raise ValueError(
            "--reducer-components is required when --reducer is "
            "pca or truncated_svd."
        )

    min_train_rows = min(int(train_idx.size) for train_idx, _ in splits)
    n_features = int(table.X.shape[1])
    max_safe_components = min(min_train_rows - 1, n_features)

    if reducer_components > max_safe_components:
        raise ValueError(
            f"--reducer-components={reducer_components} is too large for "
            f"{reducer.value} with the current CV splits. The smallest training "
            f"fold has {min_train_rows} rows and X has {n_features} features; "
            f"use <= {max_safe_components}."
        )

    return {
        "reducer": reducer.value,
        "reducer_components": int(reducer_components),
        "min_train_rows": int(min_train_rows),
        "n_features_before_reducer": int(n_features),
        "max_safe_components": int(max_safe_components),
    }


def _category_mean_predictions(
    *,
    y_train: np.ndarray,
    train_categories: np.ndarray,
    test_categories: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    means: dict[str, np.ndarray] = {}
    for category in np.unique(train_categories.astype(str)):
        keep = train_categories.astype(str) == category
        means[str(category)] = np.mean(y_train[keep], axis=0)

    return np.vstack([
        means.get(str(category), fallback)
        for category in test_categories.astype(str)
    ]).astype(np.float32, copy=False)


def _baseline_predictions_for_fold(
    table: JoinedFamilyTable,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    y_train = table.y[train_idx]
    global_mean = np.mean(y_train, axis=0)
    global_median = np.median(y_train, axis=0)

    return {
        "global_mean": np.tile(global_mean, (test_idx.size, 1)),
        "global_median": np.tile(global_median, (test_idx.size, 1)),
        "gpcr_class_mean": _category_mean_predictions(
            y_train=y_train,
            train_categories=table.gpcr_class[train_idx],
            test_categories=table.gpcr_class[test_idx],
            fallback=global_mean,
        ),
        "receptor_family_mean": _category_mean_predictions(
            y_train=y_train,
            train_categories=table.receptor_family[train_idx],
            test_categories=table.receptor_family[test_idx],
            fallback=global_mean,
        ),
    }


def _clip_if_requested(y: np.ndarray, *, clip_predictions: bool) -> np.ndarray:
    return np.clip(y, 0.0, 1.0) if clip_predictions else y


def _top_family_labels(y: np.ndarray) -> np.ndarray:
    labels: list[str] = []
    for row in y:
        if not np.isfinite(row).all() or np.nansum(row) <= 0.0:
            labels.append("masked_zero")
        else:
            labels.append(FAMILY_NAMES[int(np.argmax(row))])
    return np.array(labels, dtype=str)


def _fold_diagnostics(
    *,
    fold_idx: int,
    table: JoinedFamilyTable,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    model_pred: np.ndarray,
    baseline_pred: Mapping[str, np.ndarray],
    cv_groups: np.ndarray | None,
) -> dict[str, Any]:
    y_test = table.y[test_idx]
    diagnostics: dict[str, Any] = {
        "fold": int(fold_idx),
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "test_target_sum_min": float(np.min(np.sum(y_test, axis=1))),
        "test_target_sum_mean": float(np.mean(np.sum(y_test, axis=1))),
        "test_target_sum_max": float(np.max(np.sum(y_test, axis=1))),
        "test_true_top_family_counts": _summarize_strings(_top_family_labels(y_test)),
        "test_receptor_family_summary": _summarize_strings(table.receptor_family[test_idx]),
        "test_gpcr_class_summary": _summarize_strings(table.gpcr_class[test_idx]),
        "model": _family_metrics(y_test, model_pred)["overall"],
        "baselines": {
            name: _family_metrics(y_test, pred)["overall"]
            for name, pred in baseline_pred.items()
        },
    }

    if cv_groups is not None:
        diagnostics["test_cv_group_summary"] = _summarize_strings(cv_groups[test_idx])

    return diagnostics


def _cross_val_predict_with_diagnostics(
    *,
    model: Any,
    table: JoinedFamilyTable,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    cv_groups: np.ndarray | None,
    clip_predictions: bool,
) -> CVPredictionResult:
    y_pred = np.empty_like(table.y, dtype=np.float32)
    baseline_predictions = {
        "global_mean": np.empty_like(table.y, dtype=np.float32),
        "global_median": np.empty_like(table.y, dtype=np.float32),
        "gpcr_class_mean": np.empty_like(table.y, dtype=np.float32),
        "receptor_family_mean": np.empty_like(table.y, dtype=np.float32),
    }
    fold_metrics: list[dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        fold_model = clone(model)
        fold_model.fit(table.X[train_idx], table.y[train_idx])

        fold_pred = fold_model.predict(table.X[test_idx]).astype(np.float32, copy=False)
        fold_report_pred = _clip_if_requested(fold_pred, clip_predictions=clip_predictions)
        y_pred[test_idx] = fold_report_pred

        fold_baseline_pred = _baseline_predictions_for_fold(table, train_idx, test_idx)
        fold_report_baselines = {
            name: _clip_if_requested(pred, clip_predictions=clip_predictions)
            for name, pred in fold_baseline_pred.items()
        }
        for name, pred in fold_report_baselines.items():
            baseline_predictions[name][test_idx] = pred

        fold_metrics.append(
            _fold_diagnostics(
                fold_idx=fold_idx,
                table=table,
                train_idx=train_idx,
                test_idx=test_idx,
                model_pred=fold_report_pred,
                baseline_pred=fold_report_baselines,
                cv_groups=cv_groups,
            )
        )

    return CVPredictionResult(
        y_pred=y_pred,
        baseline_predictions=baseline_predictions,
        fold_metrics=fold_metrics,
    )


def _jsonify_numeric(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonify_numeric(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonify_numeric(item) for item in value]
    return value


def _selected_alphas(fitted_model: Any) -> Any:
    regressor = fitted_model[-1]
    if isinstance(regressor, RidgeCV):
        return _jsonify_numeric(regressor.alpha_)
    if isinstance(regressor, MultiOutputRegressor):
        return [_jsonify_numeric(cast(RidgeCV, estimator).alpha_) for estimator in regressor.estimators_]
    return None


def _selected_reducer_summary(fitted_model: Any) -> dict[str, Any] | None:
    for step in fitted_model:
        if isinstance(step, (PCA, TruncatedSVD)):
            explained = getattr(step, "explained_variance_ratio_", None)
            singular_values = getattr(step, "singular_values_", None)

            summary: dict[str, Any] = {
                "kind": (
                    "pca"
                    if isinstance(step, PCA)
                    else "truncated_svd"
                ),
                "n_components_fitted": int(step.components_.shape[0]),
                "explained_variance_ratio_sum": (
                    float(np.sum(explained))
                    if explained is not None
                    else None
                ),
            }

            if explained is not None:
                summary["explained_variance_ratio_head"] = [
                    float(value)
                    for value in explained[:10]
                ]

            if singular_values is not None:
                summary["singular_values_head"] = [
                    float(value)
                    for value in singular_values[:10]
                ]

            return summary

    return None


@app.command()
def family(
    embeddings_npz: Path = typer.Argument(..., help="Canonical ESM embedding artifact NPZ."),
    coupling_map_csv: Path = typer.Argument(..., help="GPCR common coupling map CSV."),
    model_out: Path = typer.Argument(..., help="Output .joblib model path."),
    embedding_key: str = typer.Option(
        "X",
        "--embedding-key",
        help="NPZ key to use as model features, e.g. X, X_global_mean, X_global_mean_std.",
    ),
    predictions_csv: Path = typer.Option(
        Path("reports/family_ridgecv.predictions.csv"),
        help="Output CSV with cross-validated labeled predictions.",
    ),
    all_predictions_csv: Path = typer.Option(
        Path("reports/family_ridgecv.all_predictions.csv"),
        help="Output CSV with final-model predictions for all embedded receptors.",
    ),
    metrics_json: Path = typer.Option(
        Path("reports/family_ridgecv.metrics.json"),
        help="Output JSON with CV metrics.",
    ),
    targets_npz: Path | None = typer.Option(
        Path("data/processed/labels/family_targets.npz"),
        help="Optional output NPZ for extracted family labels. Pass none to skip.",
    ),
    training_table_npz: Path | None = typer.Option(
        Path("data/processed/model_tables/family_training_table.npz"),
        help="Optional output NPZ for joined training table. Pass none to skip.",
    ),
    cv_splits: int = typer.Option(5, min=2, help="Number of CV splits."),
    cv_strategy: CVStrategy = typer.Option(
        CVStrategy.KFOLD,
        "--cv-strategy",
        help="CV splitter to use: kfold or groupkfold.",
    ),
    group_by: CVGroup = typer.Option(
        CVGroup.RECEPTOR_FAMILY,
        "--group-by",
        help="Metadata field used when --cv-strategy groupkfold is selected.",
    ),
    random_state: int = typer.Option(13, help="Random seed for shuffled K-fold CV."),
    exclude_zero_targets: bool = typer.Option(
        True,
        "--exclude-zero-targets/--include-zero-targets",
        help="Exclude complete-case rows whose four family targets sum to zero.",
    ),
    ridge_mode: RidgeMode = typer.Option(
        RidgeMode.SHARED_ALPHA,
        "--ridge-mode",
        help="Ridge strategy: shared_alpha, alpha_per_target, or independent_targets.", #alpha-per-target should prob be default but need to investigate
    ),
    reducer: ReducerKind = typer.Option(
        ReducerKind.NONE,
        "--reducer",
        help="Optional feature reducer: none, pca, or truncated_svd.",
    ),
    reducer_components: int | None = typer.Option(
        None,
        "--reducer-components",
        help="Number of PCA/SVD components. Required unless --reducer none.",
    ),
    svd_n_iter: int = typer.Option(
        7,
        "--svd-n-iter",
        min=1,
        help="Power iterations for randomized TruncatedSVD.",
    ),
    clip_predictions: bool = typer.Option(
        True,
        "--clip/--no-clip",
        help="Clip predictions to [0, 1] for reporting artifacts.",
    ),
) -> None:
    """Fit the first family-level continuous selectivity baseline."""
    embeddings = load_embedding_artifact(
        embeddings_npz,
        embedding_key=embedding_key,
    )
    targets = build_family_targets(coupling_map_csv)

    if targets_npz is not None:
        _write_npz_targets(targets_npz, targets)

    joined_table = join_embeddings_to_targets(embeddings, targets)
    complete_table = _complete_case_family_targets(joined_table)

    n_zero_target_rows_excluded = 0
    if exclude_zero_targets:
        train_table, n_zero_target_rows_excluded = _exclude_zero_target_rows(complete_table)
    else:
        train_table = complete_table

    if training_table_npz is not None:
        _write_training_table_npz(training_table_npz, train_table)

    if train_table.X.shape[0] < cv_splits:
        raise ValueError(
            f"Need at least cv_splits={cv_splits} labeled rows, "
            f"got {train_table.X.shape[0]}."
        )

    cv, cv_groups, cv_group_summary = _make_cv_splitter(
        cv_strategy=cv_strategy,
        cv_splits=cv_splits,
        random_state=random_state,
        table=train_table,
        group_by=group_by,
    )
    splits = list(cv.split(train_table.X, train_table.y, groups=cv_groups))

    reducer_cv_summary = _validate_reducer_components(
        reducer=reducer,
        reducer_components=reducer_components,
        table=train_table,
        splits=splits,
    )

    model = _make_model(
        DEFAULT_ALPHAS,
        ridge_mode=ridge_mode,
        reducer=reducer,
        reducer_components=reducer_components,
        random_state=random_state,
        svd_n_iter=svd_n_iter,
    )

    cv_result = _cross_val_predict_with_diagnostics(
        model=model,
        table=train_table,
        splits=splits,
        cv_groups=cv_groups,
        clip_predictions=clip_predictions,
    )

    metrics = _family_metrics(train_table.y, cv_result.y_pred)
    metrics.update(
        {
            "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "model": (
                "StandardScaler + RidgeCV"
                if reducer == ReducerKind.NONE
                else f"StandardScaler + {reducer.value} + RidgeCV"
            ),
            "ridge_mode": ridge_mode.value,
            "reducer": reducer.value,
            "reducer_components": (
                int(reducer_components)
                if reducer_components is not None
                else None
            ),
            "reducer_cv_summary": reducer_cv_summary,
            "svd_n_iter": (
                int(svd_n_iter)
                if reducer == ReducerKind.TRUNCATED_SVD
                else None
            ),
            "alphas": list(DEFAULT_ALPHAS),
            "cv_splits": cv_splits,
            "cv_strategy": cv_strategy.value,
            "cv_group_by": (
                group_by.value if cv_strategy == CVStrategy.GROUP_KFOLD else None
            ),
            "cv_group_summary": cv_group_summary,
            "random_state": random_state if cv_strategy == CVStrategy.KFOLD else None,
            "embedding_key": embedding_key,
            "embedding_dim": int(embeddings["X"].shape[1]),
            "n_embedding_rows": int(embeddings["X"].shape[0]),
            "n_family_target_rows": int(targets.y.shape[0]),
            "n_complete_case_rows_before_zero_filter": int(complete_table.X.shape[0]),
            "exclude_zero_targets": bool(exclude_zero_targets),
            "n_zero_target_rows_excluded": int(n_zero_target_rows_excluded),
            "n_joined_labeled_rows": int(train_table.X.shape[0]),
            "family_names": list(FAMILY_NAMES),
            "target_source": targets.label_source,
            "target_transform": "% of 1' G protein family / 100; no softmax; no row normalization",
            "baseline_metrics": {
                name: _family_metrics(train_table.y, pred)
                for name, pred in cv_result.baseline_predictions.items()
            },
            "fold_metrics": cv_result.fold_metrics,
        }
    )

    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    prediction_metadata = {
        "primary_family": train_table.primary_family,
        "receptor_family": train_table.receptor_family,
        "gpcr_class": train_table.gpcr_class,
    }
    if cv_groups is not None:
        prediction_metadata["cv_group"] = cv_groups

    _write_predictions_csv(
        predictions_csv,
        train_table.gpcrdb_entry_name,
        cv_result.y_pred,
        y_true=train_table.y,
        extra_columns=prediction_metadata,
    )

    # Fit final model on all selected complete labeled rows.
    model.fit(train_table.X, train_table.y)
    selected_alphas = _selected_alphas(model)
    selected_reducer_summary = _selected_reducer_summary(model)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "family_names": FAMILY_NAMES,
            "ridge_mode": ridge_mode.value,
            "selected_alphas": selected_alphas,
            "reducer": reducer.value,
            "reducer_components": (
                int(reducer_components)
                if reducer_components is not None
                else None
            ),
            "selected_reducer_summary": selected_reducer_summary,
            "target_source": targets.label_source,
            "target_transform": "% of 1' G protein family / 100; no softmax; no row normalization",
            "exclude_zero_targets": bool(exclude_zero_targets),
            "embedding_key": embedding_key,
            "embedding_dim": int(embeddings["X"].shape[1]),
        },
        model_out,
    )

    all_pred = model.predict(embeddings["X"].astype(np.float32, copy=False))
    all_report_pred = _clip_if_requested(all_pred, clip_predictions=clip_predictions)
    _write_predictions_csv(
        all_predictions_csv,
        embeddings["gpcrdb_entry_name"],
        all_report_pred,
    )

    metrics["selected_alphas_final_model"] = selected_alphas
    metrics["selected_reducer_final_model"] = selected_reducer_summary
    metrics_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    typer.echo(f"Labeled training rows: {train_table.X.shape[0]}")
    typer.echo(f"Excluded zero-target rows: {n_zero_target_rows_excluded}")
    typer.echo(f"CV strategy: {cv_strategy.value}")
    typer.echo(f"Ridge mode: {ridge_mode.value}")
    typer.echo(f"Reducer: {reducer.value}")
    if reducer_components is not None:
        typer.echo(f"Reducer components: {reducer_components}")
    typer.echo(f"Wrote model: {model_out}")
    typer.echo(f"Wrote metrics: {metrics_json}")
    typer.echo(f"Wrote labeled CV predictions: {predictions_csv}")
    typer.echo(f"Wrote all receptor predictions: {all_predictions_csv}")

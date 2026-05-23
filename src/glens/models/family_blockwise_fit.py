"""Blockwise PCA/SVD family selectivity fitting.

This command keeps biologically meaningful ESM region views separate until cross-
validation time. Each fold fits one scaler + PCA/SVD per region block on the
training receptors only, concatenates the fold-local reduced block scores, then
standardizes the concatenated component table before RidgeCV.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import typer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import RidgeCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from glens.models.family_fit import (
    CVGroup,
    CVPredictionResult,
    CVStrategy,
    DEFAULT_ALPHAS,
    FAMILY_NAMES,
    ReducerKind,
    RidgeMode,
    _clip_if_requested,
    _complete_case_family_targets,
    _cross_val_predict_with_diagnostics,
    _exclude_zero_target_rows,
    _family_metrics,
    _make_cv_splitter,
    _selected_alphas,
    _write_npz_targets,
    _write_predictions_csv,
    _write_training_table_npz,
    build_family_targets,
    join_embeddings_to_targets,
)

DEFAULT_BLOCK_EMBEDDING_KEYS = (
    "X_tm3_cyt_mean",
    "X_tm5_cyt_mean",
    "X_tm6_cyt_mean",
    "X_tm7_cyt_mean",
    "X_icl2_mean",
    "X_icl3_mean",
    "X_h8_mean",
)


class BlockwiseReducer(BaseEstimator, TransformerMixin):
    """Fit independent fold-local reducers for contiguous feature blocks.

    Input X is expected to be a column-wise concatenation of the requested blocks,
    in the same order as ``block_names`` and ``block_sizes``. This transformer is
    intentionally sklearn-cloneable so that normal CV code can safely fit one
    blockwise reducer per training fold.
    """

    def __init__(
        self,
        *,
        block_names: Sequence[str],
        block_sizes: Sequence[int],
        reducer: ReducerKind,
        n_components: int,
        random_state: int,
        svd_n_iter: int,
    ) -> None:
        # Keep these parameters unmodified for sklearn.clone.
        self.block_names = block_names
        self.block_sizes = block_sizes
        self.reducer = reducer
        self.n_components = n_components
        self.random_state = random_state
        self.svd_n_iter = svd_n_iter

        self.block_slices_: list[slice] = []
        self.block_pipelines_: list[Any] = []
        self.n_features_in_: int | None = None
        self.output_dim_: int | None = None
        self.reducer_summary_: dict[str, Any] | None = None

    def _normalized_blocks(self) -> tuple[tuple[str, ...], tuple[int, ...]]:
        return (
            tuple(str(name) for name in self.block_names),
            tuple(int(size) for size in self.block_sizes),
        )

    def _new_reducer(self) -> Any:
        if self.reducer == ReducerKind.PCA:
            return PCA(
                n_components=int(self.n_components),
                svd_solver="randomized",
                random_state=int(self.random_state),
            )
        if self.reducer == ReducerKind.TRUNCATED_SVD:
            return TruncatedSVD(
                n_components=int(self.n_components),
                algorithm="randomized",
                n_iter=int(self.svd_n_iter),
                random_state=int(self.random_state),
            )
        raise ValueError(f"Unsupported blockwise reducer: {self.reducer}")

    def _block_slices(self, n_features: int) -> list[slice]:
        block_names, block_sizes = self._normalized_blocks()
        if not block_names:
            raise ValueError("At least one block is required for blockwise reduction.")
        if len(block_names) != len(block_sizes):
            raise ValueError("block_names and block_sizes must have the same length.")
        if any(size <= 0 for size in block_sizes):
            raise ValueError("All block sizes must be positive.")
        if sum(block_sizes) != n_features:
            raise ValueError(
                f"Feature width mismatch: blocks sum to {sum(block_sizes)} columns, "
                f"but X has {n_features}."
            )

        start = 0
        slices: list[slice] = []
        for size in block_sizes:
            stop = start + size
            slices.append(slice(start, stop))
            start = stop
        return slices

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> BlockwiseReducer:
        del y
        x = np.asarray(X)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D X, got shape {x.shape}.")

        n_components = int(self.n_components)
        if n_components <= 0:
            raise ValueError("n_components must be positive.")

        block_names, block_sizes = self._normalized_blocks()
        block_slices = self._block_slices(int(x.shape[1]))
        max_safe_components = min(int(x.shape[0]) - 1, min(block_sizes))
        if n_components > max_safe_components:
            raise ValueError(
                f"n_components={n_components} is too large for blockwise "
                f"{self.reducer.value}; with {x.shape[0]} training rows and "
                f"min block width {min(block_sizes)}, use <= {max_safe_components}."
            )

        pipelines: list[Any] = []
        block_summaries: list[dict[str, Any]] = []
        for block_name, block_size, block_slice in zip(
            block_names,
            block_sizes,
            block_slices,
            strict=True,
        ):
            # This scaler is fit only on the current fold's training block.
            # For SVD, centering here also makes randomized TruncatedSVD behave
            # more like PCA on standardized block features.
            pipeline = make_pipeline(StandardScaler(), self._new_reducer())
            pipeline.fit(x[:, block_slice])
            reducer_step = pipeline[-1]
            explained = getattr(reducer_step, "explained_variance_ratio_", None)
            singular_values = getattr(reducer_step, "singular_values_", None)

            summary: dict[str, Any] = {
                "block": block_name,
                "input_dim": int(block_size),
                "n_components_fitted": int(reducer_step.components_.shape[0]),
                "explained_variance_ratio_sum": (
                    float(np.sum(explained)) if explained is not None else None
                ),
            }
            if explained is not None:
                summary["explained_variance_ratio_head"] = [
                    float(value) for value in explained[:10]
                ]
            if singular_values is not None:
                summary["singular_values_head"] = [
                    float(value) for value in singular_values[:10]
                ]

            pipelines.append(pipeline)
            block_summaries.append(summary)

        self.block_slices_ = block_slices
        self.block_pipelines_ = pipelines
        self.n_features_in_ = int(x.shape[1])
        self.output_dim_ = int(n_components * len(block_names))
        self.reducer_summary_ = {
            "kind": f"blockwise_{self.reducer.value}",
            "n_blocks": int(len(block_names)),
            "block_names": list(block_names),
            "block_input_dims": [int(size) for size in block_sizes],
            "components_per_block": int(n_components),
            "output_dim": int(self.output_dim_),
            "blocks": block_summaries,
        }
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.block_pipelines_ or self.n_features_in_ is None:
            raise ValueError("BlockwiseReducer must be fit before transform.")
        x = np.asarray(X)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D X, got shape {x.shape}.")
        if int(x.shape[1]) != int(self.n_features_in_):
            raise ValueError(
                f"Expected {self.n_features_in_} columns, got {x.shape[1]}."
            )

        reduced = [
            pipeline.transform(x[:, block_slice])
            for pipeline, block_slice in zip(
                self.block_pipelines_,
                self.block_slices_,
                strict=True,
            )
        ]
        return np.hstack(reduced).astype(np.float32, copy=False)

    def get_feature_names_out(
        self,
        input_features: Sequence[str] | None = None,
    ) -> np.ndarray:
        del input_features
        block_names, _ = self._normalized_blocks()
        return np.array(
            [
                f"{block}:component_{idx + 1}"
                for block in block_names
                for idx in range(int(self.n_components))
            ],
            dtype=object,
        )


def _embedding_blocks(blocks: Sequence[str] | None) -> tuple[str, ...]:
    requested = tuple(block.strip() for block in (blocks or ()) if block.strip())
    return requested or DEFAULT_BLOCK_EMBEDDING_KEYS


def load_block_embedding_artifact(
    path: Path,
    *,
    embedding_blocks: Sequence[str],
) -> tuple[dict[str, np.ndarray], tuple[str, ...], tuple[int, ...]]:
    data = np.load(path, allow_pickle=False)
    blocks = tuple(embedding_blocks)
    required = {"gpcrdb_entry_name", *blocks}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(
            f"Embedding artifact missing keys: {sorted(missing)}. "
            f"Available keys: {', '.join(data.files)}"
        )

    arrays = [np.asarray(data[key]) for key in blocks]
    n_rows = int(arrays[0].shape[0])
    for key, array in zip(blocks, arrays, strict=True):
        if array.ndim != 2:
            raise ValueError(f"Embedding block {key!r} must be 2D, got {array.shape}.")
        if int(array.shape[0]) != n_rows:
            raise ValueError(
                f"Embedding block {key!r} has {array.shape[0]} rows; "
                f"expected {n_rows}."
            )

    payload = {key: data[key] for key in data.files}
    payload["X"] = np.concatenate(arrays, axis=1).astype(np.float32, copy=False)
    return payload, blocks, tuple(int(array.shape[1]) for array in arrays)


def _make_regressor(alphas: Sequence[float], *, ridge_mode: RidgeMode) -> Any:
    alpha_grid = np.array(alphas, dtype=np.float64)
    if ridge_mode == RidgeMode.SHARED_ALPHA:
        return RidgeCV(alphas=alpha_grid)
    if ridge_mode == RidgeMode.ALPHA_PER_TARGET:
        return RidgeCV(alphas=alpha_grid, alpha_per_target=True)
    if ridge_mode == RidgeMode.INDEPENDENT_TARGETS:
        return MultiOutputRegressor(RidgeCV(alphas=alpha_grid))
    raise ValueError(f"Unsupported ridge mode: {ridge_mode}")


def _make_model(
    *,
    ridge_mode: RidgeMode,
    reducer: ReducerKind,
    components_per_block: int,
    block_names: Sequence[str],
    block_sizes: Sequence[int],
    random_state: int,
    svd_n_iter: int,
) -> Any:
    if reducer == ReducerKind.NONE:
        raise ValueError("Blockwise fit requires --blockwise-reducer pca or truncated_svd.")

    return make_pipeline(
        BlockwiseReducer(
            block_names=block_names,
            block_sizes=block_sizes,
            reducer=reducer,
            n_components=components_per_block,
            random_state=random_state,
            svd_n_iter=svd_n_iter,
        ),
        # Standardize the concatenated PC/SVD scores before RidgeCV so components
        # or blocks with larger retained variance do not get an accidental lower
        # effective ridge penalty.
        StandardScaler(),
        _make_regressor(DEFAULT_ALPHAS, ridge_mode=ridge_mode),
    )


def _validate_components(
    *,
    reducer: ReducerKind,
    components_per_block: int,
    block_names: Sequence[str],
    block_sizes: Sequence[int],
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    if reducer == ReducerKind.NONE:
        raise ValueError("--blockwise-reducer must be pca or truncated_svd.")
    if components_per_block <= 0:
        raise ValueError("--blockwise-components must be positive.")
    if not block_names:
        raise ValueError("At least one --embedding-block is required.")
    if len(block_names) != len(block_sizes):
        raise ValueError("Block names and block sizes differ in length.")

    min_train_rows = min(int(train_idx.size) for train_idx, _ in splits)
    max_safe_components = min(min_train_rows - 1, min(int(size) for size in block_sizes))
    if components_per_block > max_safe_components:
        raise ValueError(
            f"--blockwise-components={components_per_block} is too large. "
            f"Smallest training fold has {min_train_rows} rows and smallest block "
            f"has {min(block_sizes)} features; use <= {max_safe_components}."
        )

    return {
        "reducer": f"blockwise_{reducer.value}",
        "components_per_block": int(components_per_block),
        "min_train_rows": int(min_train_rows),
        "n_blocks": int(len(block_names)),
        "block_names": [str(name) for name in block_names],
        "block_input_dims": [int(size) for size in block_sizes],
        "n_features_before_reducer": int(sum(block_sizes)),
        "n_features_after_reducer": int(components_per_block * len(block_names)),
        "max_safe_components_per_block": int(max_safe_components),
    }


def _selected_reducer_summary(fitted_model: Any) -> dict[str, Any] | None:
    reducer = fitted_model[0]
    summary = getattr(reducer, "reducer_summary_", None)
    return cast(dict[str, Any] | None, summary)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def family_blockwise(
    embeddings_npz: Path = typer.Argument(..., help="Canonical ESM embedding artifact NPZ."),
    coupling_map_csv: Path = typer.Argument(..., help="GPCR common coupling map CSV."),
    model_out: Path = typer.Argument(..., help="Output .joblib model path."),
    embedding_blocks: list[str] | None = typer.Option(
        None,
        "--embedding-block",
        help=(
            "Repeatable NPZ region-view key. Defaults to X_tm3_cyt_mean, "
            "X_tm5_cyt_mean, X_tm6_cyt_mean, X_icl2_mean, and X_h8_mean."
        ),
    ),
    blockwise_reducer: ReducerKind = typer.Option(
        ReducerKind.PCA,
        "--blockwise-reducer",
        help="Per-block reducer fit inside each CV fold: pca or truncated_svd.",
    ),
    blockwise_components: int = typer.Option(
        16,
        "--blockwise-components",
        min=1,
        help="Number of PCA/SVD components to keep per block.",
    ),
    predictions_csv: Path = typer.Option(
        Path("reports/family_blockwise_ridgecv.predictions.csv"),
        help="Output CSV with cross-validated labeled predictions.",
    ),
    all_predictions_csv: Path = typer.Option(
        Path("reports/family_blockwise_ridgecv.all_predictions.csv"),
        help="Output CSV with final-model predictions for all embedded receptors.",
    ),
    metrics_json: Path = typer.Option(
        Path("reports/family_blockwise_ridgecv.metrics.json"),
        help="Output JSON with CV metrics.",
    ),
    targets_npz: Path | None = typer.Option(
        Path("data/processed/labels/family_targets.npz"),
        help="Optional output NPZ for extracted family labels. Pass none to skip.",
    ),
    training_table_npz: Path | None = typer.Option(
        Path("data/processed/model_tables/family_blockwise_training_table.npz"),
        help="Optional output NPZ for joined block-concat training table. Pass none to skip.",
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
        help="Ridge strategy: shared_alpha, alpha_per_target, or independent_targets.",
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
    """Fit RidgeCV on fold-local blockwise PCA/SVD region features."""
    blocks = _embedding_blocks(embedding_blocks)
    embeddings, block_names, block_sizes = load_block_embedding_artifact(
        embeddings_npz,
        embedding_blocks=blocks,
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
    reducer_cv_summary = _validate_components(
        reducer=blockwise_reducer,
        components_per_block=blockwise_components,
        block_names=block_names,
        block_sizes=block_sizes,
        splits=splits,
    )

    model = _make_model(
        ridge_mode=ridge_mode,
        reducer=blockwise_reducer,
        components_per_block=blockwise_components,
        block_names=block_names,
        block_sizes=block_sizes,
        random_state=random_state,
        svd_n_iter=svd_n_iter,
    )

    cv_result: CVPredictionResult = _cross_val_predict_with_diagnostics(
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
                f"blockwise StandardScaler + blockwise {blockwise_reducer.value} "
                "+ StandardScaler + RidgeCV"
            ),
            "ridge_mode": ridge_mode.value,
            "blockwise_reducer": blockwise_reducer.value,
            "blockwise_components": int(blockwise_components),
            "reducer_cv_summary": reducer_cv_summary,
            "svd_n_iter": (
                int(svd_n_iter)
                if blockwise_reducer == ReducerKind.TRUNCATED_SVD
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
            "embedding_blocks": list(block_names),
            "default_block_embedding_keys": list(DEFAULT_BLOCK_EMBEDDING_KEYS),
            "feature_block_sizes": [int(size) for size in block_sizes],
            "embedding_dim_before_reducer": int(embeddings["X"].shape[1]),
            "embedding_dim_after_reducer": int(blockwise_components * len(block_names)),
            "n_embedding_rows": int(embeddings["X"].shape[0]),
            "n_family_target_rows": int(targets.y.shape[0]),
            "n_complete_case_rows_before_zero_filter": int(complete_table.X.shape[0]),
            "exclude_zero_targets": bool(exclude_zero_targets),
            "n_zero_target_rows_excluded": int(n_zero_target_rows_excluded),
            "n_joined_labeled_rows": int(train_table.X.shape[0]),
            "family_names": list(FAMILY_NAMES),
            "target_source": targets.label_source,
            "target_transform": (
                "% of 1' G protein family / 100; "
                "no softmax; no row normalization"
            ),
            "baseline_metrics": {
                name: _family_metrics(train_table.y, pred)
                for name, pred in cv_result.baseline_predictions.items()
            },
            "fold_metrics": cv_result.fold_metrics,
        }
    )
    _write_json(metrics_json, metrics)

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
            "blockwise_reducer": blockwise_reducer.value,
            "blockwise_components": int(blockwise_components),
            "selected_reducer_summary": selected_reducer_summary,
            "target_source": targets.label_source,
            "target_transform": (
                "% of 1' G protein family / 100; "
                "no softmax; no row normalization"
            ),
            "exclude_zero_targets": bool(exclude_zero_targets),
            "embedding_blocks": list(block_names),
            "feature_block_sizes": [int(size) for size in block_sizes],
            "embedding_dim_before_reducer": int(embeddings["X"].shape[1]),
            "embedding_dim_after_reducer": int(blockwise_components * len(block_names)),
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
    _write_json(metrics_json, metrics)

    typer.echo(f"Labeled training rows: {train_table.X.shape[0]}")
    typer.echo(f"Excluded zero-target rows: {n_zero_target_rows_excluded}")
    typer.echo(f"CV strategy: {cv_strategy.value}")
    typer.echo(f"Ridge mode: {ridge_mode.value}")
    typer.echo(f"Blockwise reducer: {blockwise_reducer.value}")
    typer.echo(f"Components per block: {blockwise_components}")
    typer.echo(f"Embedding blocks: {', '.join(block_names)}")
    typer.echo(f"Wrote model: {model_out}")
    typer.echo(f"Wrote metrics: {metrics_json}")
    typer.echo(f"Wrote labeled CV predictions: {predictions_csv}")
    typer.echo(f"Wrote all receptor predictions: {all_predictions_csv}")

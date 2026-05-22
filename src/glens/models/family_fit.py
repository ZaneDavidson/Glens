"""
Family-level GPCR-G protein selectivity fitting.

This module builds receptor-level family targets from the GPCRdb common coupling
map, joins them to an ESM embedding table, fits a RidgeCV with cross-validation,
and writes model, metrics, predictions, and optional training-tables.
"""

import csv
import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import typer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

app = typer.Typer(no_args_is_help=True)

FAMILY_NAMES = ("Gs", "Gi/o", "Gq/11", "G12/13")
UNIPROT_ENTRY_RE = re.compile(r"^[a-z0-9]+_[a-z0-9]+$")
GPCRDB_ENTRY_RE = re.compile(r"/protein/([^/?#]+)")
DEFAULT_ALPHAS = np.logspace(-3, 4, 200).tolist()


@dataclass(frozen=True)
class FamilyTargets:
    gpcrdb_entry_name: np.ndarray
    y: np.ndarray
    mask: np.ndarray
    primary_family: np.ndarray
    label_source: str


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

# Should we model an unknown class? 
def _top1_evaluable_mask(y_true: np.ndarray) -> np.ndarray:
    """Return rows with an interpretable top family.

    In the case of no available family coupling scores, argmax([0, 0, 0, 0]) returns 0,
    incorrectly scoring the receptor as Gs primary.  
    """
    return np.isfinite(y_true).all(axis=1) & (np.nansum(y_true, axis=1) > 0.0)


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
        seen.add(entry)

    if not y_rows:
        raise ValueError("No usable family targets were found in the coupling map.")

    return FamilyTargets(
        gpcrdb_entry_name=np.array(ids, dtype=str),
        y=np.array(y_rows, dtype=np.float32),
        mask=np.array(mask_rows, dtype=bool),
        primary_family=np.array(primary, dtype=str),
        label_source=f"{source_lab}/{source_biosensor}",
    )


def load_embedding_artifact(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    required = {"X", "gpcrdb_entry_name"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"Embedding artifact missing keys: {sorted(missing)}")
    return {key: data[key] for key in data.files}


def join_embeddings_to_targets(
    embeddings: dict[str, np.ndarray],
    targets: FamilyTargets,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Inner-join embedding table to labels by gpcrdb_entry_name."""
    emb_ids = embeddings["gpcrdb_entry_name"].astype(str)
    emb_index = {entry: idx for idx, entry in enumerate(emb_ids)}

    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    joined_ids: list[str] = []

    for idx, entry in enumerate(targets.gpcrdb_entry_name.astype(str)):
        emb_idx = emb_index.get(entry)
        if emb_idx is None:
            continue
        x_rows.append(embeddings["X"][emb_idx])
        y_rows.append(targets.y[idx])
        mask_rows.append(targets.mask[idx])
        joined_ids.append(entry)

    if not x_rows:
        raise ValueError("No key matches between embedding IDs and family target IDs.")

    X = np.vstack(x_rows).astype(np.float32, copy=False)
    y = np.vstack(y_rows).astype(np.float32, copy=False)
    mask = np.vstack(mask_rows).astype(bool, copy=False)
    ids = np.array(joined_ids, dtype=str)
    return X, y, mask, ids


def _complete_case_family_targets(
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = mask.all(axis=1) & np.isfinite(y).all(axis=1)
    if not np.any(keep):
        raise ValueError("No rows have complete family targets.")
    return X[keep], y[keep], ids[keep]


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
        "r2_variance_weighted": float(r2_score(y_true, y_pred, multioutput="variance_weighted")),
        "top1_family_accuracy": _top1_accuracy(y_true, y_pred),
        "top1_n_evaluable_rows": int(np.sum(top1_evaluable)),
        "top1_n_masked_rows": int(y_true.shape[0] - np.sum(top1_evaluable)), # incl the mask rule?
    }
    return metrics


def _write_predictions_csv(
    path: Path,
    ids: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_true: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["gpcrdb_entry_name"]
    fields.extend(f"pred_{family}" for family in FAMILY_NAMES)
    if y_true is not None:
        fields.extend(f"true_{family}" for family in FAMILY_NAMES)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row_idx, entry in enumerate(ids.astype(str)):
            row: dict[str, str | float] = {"gpcrdb_entry_name": entry}
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
        label_source=np.array(targets.label_source, dtype=str),
    )


def _write_training_table_npz(
    path: Path,
    X: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=X.astype(np.float32, copy=False),
        y_family=y.astype(np.float32, copy=False),
        gpcrdb_entry_name=ids.astype(str),
        family_names=np.array(FAMILY_NAMES, dtype=str),
    )


def _make_model(alphas: Sequence[float]) -> Any:
    return make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.array(alphas, dtype=np.float64)),
    )


@app.command()
def family(
    embeddings_npz: Path = typer.Argument(..., help="Embedding .npz"),
    coupling_map_csv: Path = typer.Argument(..., help="GPCRdb common coupling map .csv"),
    model_out: Path = typer.Argument(..., help="Output .joblib model path."),
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
    cv_splits: int = typer.Option(5, min=2, help="Number of K-fold CV splits."),
    random_state: int = typer.Option(13, help="Random seed for CV splits."),
    clip_predictions: bool = typer.Option(
        True,
        "--clip/--no-clip",
        help="Clip predictions to [0, 1] for reporting artifacts.",
    ),
) -> None:
    """Fit the first family-level continuous selectivity baseline."""
    embeddings = load_embedding_artifact(embeddings_npz)
    targets = build_family_targets(coupling_map_csv)

    if targets_npz is not None:
        _write_npz_targets(targets_npz, targets)

    X_joined, y_joined, mask_joined, joined_ids = join_embeddings_to_targets(
        embeddings,
        targets,
    )
    X_train, y_train, train_ids = _complete_case_family_targets(
        X_joined,
        y_joined,
        mask_joined,
        joined_ids,
    )

    if training_table_npz is not None:
        _write_training_table_npz(training_table_npz, X_train, y_train, train_ids)

    if X_train.shape[0] < cv_splits:
        raise ValueError(
            f"Need at least cv_splits={cv_splits} labeled rows, got {X_train.shape[0]}."
        )

    model = _make_model(DEFAULT_ALPHAS)
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    cv_pred = cross_val_predict(model, X_train, y_train, cv=cv)
    report_pred = np.clip(cv_pred, 0.0, 1.0) if clip_predictions else cv_pred

    metrics = _family_metrics(y_train, report_pred)
    metrics.update(
        {
            "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "model": "StandardScaler + RidgeCV",
            "alphas": list(DEFAULT_ALPHAS),
            "cv_splits": cv_splits,
            "random_state": random_state,
            "n_embedding_rows": int(embeddings["X"].shape[0]),
            "n_family_target_rows": int(targets.y.shape[0]),
            "n_joined_labeled_rows": int(X_train.shape[0]),
            "family_names": list(FAMILY_NAMES),
            "target_source": targets.label_source,
            "target_transform": "% of 1' G protein family / 100; no softmax; no row normalization",
        }
    )

    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    _write_predictions_csv(predictions_csv, train_ids, report_pred, y_true=y_train)

    # Fit final model on all complete labeled rows.
    model.fit(X_train, y_train)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "family_names": FAMILY_NAMES,
            "target_source": targets.label_source,
            "target_transform": "% of 1' G protein family / 100; no softmax; no row normalization",
        },
        model_out,
    )

    all_pred = model.predict(embeddings["X"].astype(np.float32, copy=False))
    all_report_pred = np.clip(all_pred, 0.0, 1.0) if clip_predictions else all_pred
    _write_predictions_csv(
        all_predictions_csv,
        embeddings["gpcrdb_entry_name"],
        all_report_pred,
    )

    typer.echo(f"Labeled training rows: {X_train.shape[0]}")
    typer.echo(f"Wrote model: {model_out}")
    typer.echo(f"Wrote metrics: {metrics_json}")
    typer.echo(f"Wrote labeled CV predictions: {predictions_csv}")
    typer.echo(f"Wrote all receptor predictions: {all_predictions_csv}")

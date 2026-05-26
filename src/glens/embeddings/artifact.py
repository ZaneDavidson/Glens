"""Embedding artifact construction."""

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from glens.data.sequences import SequenceRecord
from glens.embeddings.model import (
    DEFAULT_MAX_RESIDUES,
    DEFAULT_MODEL_ID,
    ResidueEmbeddingResult,
    embed_residue_sequences,
)
from glens.embeddings.views import (
    EmbeddingViews,
    build_global_views,
    build_region_views,
    merge_view_metadata,
    merge_views,
    stack_view_rows,
)
from glens.models.regions import (
    RegionMasks,
    empty_region_masks,
    load_region_cache,
    reconcile_length_arguments,
    validate_region_masks,
)

BACKCOMPAT_EMBEDDING_KEY = "X_global_mean"
PROGRESS_STATUS_WIDTH = 42


@dataclass(frozen=True)
class EmbeddingRunConfig:
    """Non-source-specific embedding run metadata."""

    model_id: str = DEFAULT_MODEL_ID
    max_residues: int = DEFAULT_MAX_RESIDUES
    stride: int | None = None
    batch_size: int = 8
    source_name: str = "sequence_records"
    source_uri: str | None = None
    region_mask_json: Path | None = None


def write_sequence_embedding_artifact(
    records: Sequence[SequenceRecord],
    output_npz: Path,
    *,
    tokenizer: Any,
    model: Any,
    device: Any,
    config: EmbeddingRunConfig,
    write_audit_csv: bool = True,
    write_metadata_json: bool = True,
) -> None:
    """Embed records, build all configured views, and write a compressed NPZ.

    This is the core automation used by both GPCRdb coupling-map embedding and
    arbitrary WT/mutant sequence-table embedding.
    """
    if not records:
        raise ValueError("At least one sequence record is required.")
    if output_npz.suffix != ".npz":
        raise ValueError("Output path must end in .npz")

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    region_cache = (
        load_region_cache(config.region_mask_json)
        if config.region_mask_json is not None
        else None
    )

    enriched_records = [_with_embedding_counts(row, config) for row in records]
    view_blocks: list[dict[str, np.ndarray]] = []
    view_rows: list[EmbeddingViews] = []

    with tqdm(
        total=len(enriched_records),
        desc="Embedding sequences",
        unit="sequence",
        dynamic_ncols=False,
        ncols=118,
        bar_format=(
            "{desc}: {percentage:3.0f}%|{bar:32}| "
            "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        ),
    ) as bar:
        for start in range(0, len(enriched_records), config.batch_size):
            batch_records = enriched_records[start : start + config.batch_size]
            batch_ids = ", ".join(row.sequence_id for row in batch_records[:3])
            if len(batch_records) > 3:
                batch_ids += ", ..."
            bar.set_postfix_str(_progress_status(f"embed batch: {batch_ids}"))

            residue_results = embed_residue_sequences(
                (row.sequence for row in batch_records),
                tokenizer,
                model,
                batch_size=config.batch_size,
                max_residues=config.max_residues,
                stride=config.stride,
                device=device,
            )
            if isinstance(residue_results, ResidueEmbeddingResult):
                residue_results = [residue_results]

            batch_views = [
                _views_for_result(
                    record=row,
                    result=result,
                    region_cache=region_cache,
                )
                for row, result in zip(batch_records, residue_results, strict=True)
            ]
            view_names = list(batch_views[0].arrays.keys())
            view_blocks.append(stack_view_rows(batch_views, view_names))
            view_rows.extend(batch_views)
            bar.update(len(batch_records))

    view_arrays = _stack_view_blocks(view_blocks)
    x_alias = view_arrays[BACKCOMPAT_EMBEDDING_KEY]
    if x_alias.shape[0] != len(enriched_records):
        raise RuntimeError(
            f"Embedding row mismatch: X has {x_alias.shape[0]} rows, "
            f"but records has {len(enriched_records)} rows."
        )

    _write_npz(
        output_npz,
        enriched_records,
        view_arrays,
        extra_arrays=_region_extra_arrays(view_rows),
    )

    audit_csv = _audit_csv_path(output_npz) if write_audit_csv else None
    if audit_csv is not None:
        _write_audit_csv(audit_csv, enriched_records)

    if write_metadata_json:
        _write_metadata_json(
            output_npz,
            records=enriched_records,
            config=config,
            view_arrays=view_arrays,
            view_metadata=merge_view_metadata(view_rows),
            audit_csv=audit_csv,
        )


def _with_embedding_counts(
    record: SequenceRecord,
    config: EmbeddingRunConfig,
) -> SequenceRecord:
    metadata = dict(record.metadata)
    metadata["sequence_length"] = str(len(record.sequence))
    n_windows = _n_windows(len(record.sequence), config.max_residues, config.stride)
    metadata["n_windows"] = str(n_windows)
    metadata["n_chunks"] = str(n_windows)
    return SequenceRecord(
        sequence_id=record.sequence_id,
        sequence=record.sequence,
        source=record.source,
        gpcrdb_entry_name=record.gpcrdb_entry_name,
        uniprot_accession=record.uniprot_accession,
        uniprot_id=record.uniprot_id,
        metadata=metadata,
    )


def _views_for_result(
    *,
    record: SequenceRecord,
    result: ResidueEmbeddingResult,
    region_cache: dict[str, RegionMasks] | None,
) -> EmbeddingViews:
    global_views = build_global_views(result)
    if region_cache is None:
        return global_views

    cache_key = record.gpcrdb_entry_name or record.sequence_id
    masks = region_cache.get(cache_key)
    if masks is None:
        masks = empty_region_masks(cache_key, result.residue_embeddings.shape[0])
    else:
        masks = reconcile_length_arguments(
            masks,
            sequence_length=result.residue_embeddings.shape[0],
        )
        validate_region_masks(
            masks,
            sequence_length=result.residue_embeddings.shape[0],
        )

    region_views = build_region_views(result, masks)
    return merge_views(global_views, region_views)


def _write_npz(
    output_npz: Path,
    records: Sequence[SequenceRecord],
    view_arrays: dict[str, np.ndarray],
    *,
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> None:
    x_alias = view_arrays[BACKCOMPAT_EMBEDDING_KEY]
    payload = {
        "X": x_alias.astype(np.float32, copy=False),
        **{
            key: value.astype(np.float32, copy=False)
            for key, value in view_arrays.items()
        },
        "sequence_id": np.array([row.sequence_id for row in records], dtype=str),
        "source": np.array([row.source for row in records], dtype=str),
        # Backward-compatible training join key. Source-agnostic batches use
        # sequence_id as fallback, while GPCRdb/UniProt records preserve entry names.
        "gpcrdb_entry_name": np.array(
            [row.gpcrdb_entry_name or row.sequence_id for row in records],
            dtype=str,
        ),
        "uniprot_accession": np.array(
            [row.uniprot_accession or "" for row in records],
            dtype=str,
        ),
        "uniprot_id": np.array([row.uniprot_id or "" for row in records], dtype=str),
        "sequence_length": np.array(
            [int(row.metadata["sequence_length"]) for row in records],
            dtype=np.int32,
        ),
        "n_chunks": np.array(
            [int(row.metadata["n_chunks"]) for row in records],
            dtype=np.int16,
        ),
        "n_windows": np.array(
            [int(row.metadata["n_windows"]) for row in records],
            dtype=np.int16,
        ),
    }
    if extra_arrays is not None:
        payload.update(extra_arrays)
    np.savez_compressed(output_npz, **payload)


def _write_metadata_json(
    output_npz: Path,
    *,
    records: Sequence[SequenceRecord],
    config: EmbeddingRunConfig,
    view_arrays: dict[str, np.ndarray],
    view_metadata: dict[str, object],
    audit_csv: Path | None,
) -> None:
    x_alias = view_arrays[BACKCOMPAT_EMBEDDING_KEY]
    metadata = {
        "source_name": config.source_name,
        "source_uri": config.source_uri,
        "output_npz": str(output_npz),
        "audit_csv": str(audit_csv) if audit_csv is not None else None,
        "model_id": config.model_id,
        "embedding_key": "X",
        "embedding_alias_of": BACKCOMPAT_EMBEDDING_KEY,
        "embedding_shape": list(x_alias.shape),
        "embedding_dim": int(x_alias.shape[1]),
        "dtype": str(x_alias.dtype),
        "pooling": "global_and_region_views_from_reconstructed_residue_tokens",
        "chunking": "windowed_residue_reconstruction",
        "max_residues": config.max_residues,
        "stride": config.stride,
        "batch_size": config.batch_size,
        "n_records": len(records),
        "view_shapes": {key: list(value.shape) for key, value in view_arrays.items()},
        "view_metadata": view_metadata,
        "region_mask_json": str(config.region_mask_json) if config.region_mask_json else None,
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    _metadata_json_path(output_npz).write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_audit_csv(audit_csv: Path, records: Sequence[SequenceRecord]) -> None:
    fieldnames = [
        "sequence_id",
        "source",
        "gpcrdb_entry_name",
        "uniprot_accession",
        "uniprot_id",
        "sequence_length",
        "n_chunks",
        "n_windows",
    ]
    with audit_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    "sequence_id": row.sequence_id,
                    "source": row.source,
                    "gpcrdb_entry_name": row.gpcrdb_entry_name or "",
                    "uniprot_accession": row.uniprot_accession or "",
                    "uniprot_id": row.uniprot_id or "",
                    "sequence_length": row.metadata["sequence_length"],
                    "n_chunks": row.metadata["n_chunks"],
                    "n_windows": row.metadata["n_windows"],
                }
            )


def _stack_view_blocks(view_blocks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not view_blocks:
        raise RuntimeError("No embedding view blocks were produced.")
    keys = list(view_blocks[0].keys())
    return {
        key: np.vstack([block[key] for block in view_blocks]).astype(
            np.float32,
            copy=False,
        )
        for key in keys
    }


def _region_extra_arrays(view_rows: Sequence[EmbeddingViews]) -> dict[str, np.ndarray]:
    if not view_rows:
        return {}
    region_summary = view_rows[0].metadata.get("region_residue_counts")
    if not isinstance(region_summary, Mapping):
        return {}

    region_names = sorted(str(name) for name in region_summary.keys())
    count_rows: list[list[int]] = []
    for views in view_rows:
        counts_obj = views.metadata.get("region_residue_counts")
        if not isinstance(counts_obj, Mapping):
            count_rows.append([0 for _ in region_names])
            continue
        count_rows.append([int(counts_obj.get(name, 0)) for name in region_names])

    counts = np.array(count_rows, dtype=np.int16)
    return {
        "region_names": np.array(region_names, dtype=str),
        "region_residue_counts": counts,
        "region_missing_mask": counts == 0,
    }


def _n_windows(sequence_length: int, max_residues: int, stride: int | None) -> int:
    if sequence_length <= max_residues:
        return 1
    if stride is None:
        stride = max_residues
    if stride > max_residues:
        raise ValueError("stride must be <= max_residues to avoid coverage gaps.")
    if stride == max_residues:
        return (sequence_length + max_residues - 1) // max_residues

    starts = list(range(0, max(1, sequence_length - max_residues + 1), stride))
    final_start = max(0, sequence_length - max_residues)
    if starts[-1] != final_start:
        starts.append(final_start)
    return len(starts)


def _metadata_json_path(output_npz: Path) -> Path:
    return output_npz.with_suffix(".metadata.json")


def _audit_csv_path(output_npz: Path) -> Path:
    return output_npz.with_suffix(".audit.csv")


def _progress_status(text: str, width: int = PROGRESS_STATUS_WIDTH) -> str:
    text = text.replace("\n", " ")
    if len(text) > width:
        text = text[: width - 1] + "…"
    return f"{text:<{width}}"

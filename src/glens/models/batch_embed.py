"""
Typer CLI for embedding unique GPCRs from GPCRdb common coupling map.
"""

# TODO: I want to make this less GPCRdb facing. Yes, that is where data is ingested now but
# what about new assays? Needs functionality for embedding just from a UniProt accession list.

import csv
import json
import re
from collections.abc import Iterable, Iterator, Mapping
from datetime import UTC, datetime
from itertools import chain
from pathlib import Path

import numpy as np
import requests
import typer
from tqdm.auto import tqdm

from glens.models.embed_model import (
    DEFAULT_MAX_RESIDUES,
    DEFAULT_MODEL_ID,
    ResidueEmbeddingResult,
    embed_residue_sequences,
    load_plm,
)
from glens.models.embed_views import (
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

app = typer.Typer(no_args_is_help=True)
UNIPROT_ENTRY_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{entry_id}.fasta"
UNIPROT_ENTRY_JSON_URL = "https://rest.uniprot.org/uniprotkb/{entry_id}.json"
UNIPROT_ENTRY_RE = re.compile(r"^[a-z0-9]+_[a-z0-9]+$")
GPCRDB_ENTRY_RE = re.compile(r"/protein/([^/?#]+)")
BACKCOMPAT_EMBEDDING_KEY = "X_global_mean"


def _clean_header(value: str) -> str:
    return value.strip().lstrip("\ufeff")


def _column_index(row: list[str], name: str) -> int | None:
    cleaned = [_clean_header(cell) for cell in row]
    return cleaned.index(name) if name in cleaned else None


def _iter_gpcrdb_entry_names(path: Path, column: str = "GPCRdb") -> Iterator[str]:
    """
    Yield unique GPCRdb/UniProt-style entry names from a coupling map CSV.
    """
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        first = next(reader, None)
        if first is None:
            return

        second = next(reader, None)
        second_idx = _column_index(second, column) if second is not None else None
        first_idx = _column_index(first, column)

        if second_idx is not None:
            idx = second_idx
            rows: Iterable[list[str]] = reader
        elif first_idx is not None:
            idx = first_idx
            rows = chain([second], reader) if second is not None else reader
        else:
            raise ValueError(f"Column {column!r} not found in {path}")

        seen: set[str] = set()
        for row in rows:
            if row is None or idx >= len(row):
                continue
            value = row[idx].strip()
            match = GPCRDB_ENTRY_RE.search(value)
            entry_name = (match.group(1) if match else value).strip().lower()

            if not UNIPROT_ENTRY_RE.match(entry_name):  # Skip malformed rows
                continue

            if entry_name in seen:
                continue

            seen.add(entry_name)
            yield entry_name


def _parse_fasta_sequence(text: str) -> str:
    return "".join(
        line.strip()
        for line in text.splitlines()
        if line and not line.startswith(">")
    )


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


# TODO: When constructing metadata, refactor some of this so we can grab taxon and organism
def fetch_uniprot_sequence(
    entry_name: str,
    session: requests.Session,
    timeout: float = 30.0,
) -> dict[str, str]:
    """
    Resolve a GPCRdb entry name to a UniProt sequence.
    """
    entry_id = entry_name.upper()

    json_response = session.get(
        UNIPROT_ENTRY_JSON_URL.format(entry_id=entry_id),
        timeout=timeout,
    )
    json_response.raise_for_status()
    record = json_response.json()

    accession = record.get("primaryAccession")
    uniprot_id = record.get("uniProtkbId", entry_id)

    if not accession:
        raise LookupError(f"UniProt record for {entry_name!r} did not include an accession")

    sequence = record.get("sequence", {}).get("value")

    if not sequence:
        fasta_response = session.get(
            UNIPROT_ENTRY_FASTA_URL.format(entry_id=accession),
            timeout=timeout,
        )
        fasta_response.raise_for_status()
        sequence = _parse_fasta_sequence(fasta_response.text)

    if not sequence:
        raise LookupError(
            f"UniProt record for {entry_name!r} resolved to "
            f"{uniprot_id!r} / {accession!r}, but no sequence was found"
        )

    return {
        "gpcrdb_entry_name": entry_name,
        "uniprot_accession": accession,
        "uniprot_id": uniprot_id,
        "sequence": sequence,
    }


def _metadata_json_path(output_npz: Path) -> Path:
    return output_npz.with_suffix(".metadata.json")


def _audit_csv_path(output_npz: Path) -> Path:
    return output_npz.with_suffix(".audit.csv")


def _write_npz(
    output_npz: Path,
    records: list[dict[str, str]],
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
        "gpcrdb_entry_name": np.array(
            [row["gpcrdb_entry_name"] for row in records],
            dtype=str,
        ),
        "uniprot_accession": np.array(
            [row["uniprot_accession"] for row in records],
            dtype=str,
        ),
        "uniprot_id": np.array(
            [row["uniprot_id"] for row in records],
            dtype=str,
        ),
        "sequence_length": np.array(
            [int(row["sequence_length"]) for row in records],
            dtype=np.int32,
        ),
        "n_chunks": np.array(
            [int(row["n_windows"]) for row in records],
            dtype=np.int16,
        ),
        "n_windows": np.array(
            [int(row["n_windows"]) for row in records],
            dtype=np.int16,
        ),
    }

    if extra_arrays is not None:
        payload.update(extra_arrays)

    np.savez_compressed(output_npz, **payload)


def _write_metadata_json(
    output_npz: Path,
    *,
    input_csv: Path,
    model_id: str,
    max_residues: int,
    stride: int | None,
    view_arrays: dict[str, np.ndarray],
    view_metadata: dict[str, object],
    gpcrdb_column: str,
    audit_csv: Path | None,
    region_mask_json: Path | None,
) -> None:
    x_alias = view_arrays[BACKCOMPAT_EMBEDDING_KEY]

    metadata = {
        "source_csv": str(input_csv),
        "output_npz": str(output_npz),
        "audit_csv": str(audit_csv) if audit_csv is not None else None,
        "model_id": model_id,
        "embedding_key": "X",
        "embedding_alias_of": BACKCOMPAT_EMBEDDING_KEY,
        "embedding_shape": list(x_alias.shape),
        "embedding_dim": int(x_alias.shape[1]),
        "dtype": str(x_alias.dtype),
        "pooling": "global_and_region_views_from_reconstructed_residue_tokens",
        "chunking": "windowed_residue_reconstruction",
        "max_residues": max_residues,
        "stride": stride,
        "view_shapes": {
            key: list(value.shape)
            for key, value in view_arrays.items()
        },
        "view_metadata": view_metadata,
        "region_mask_json": str(region_mask_json) if region_mask_json is not None else None,
        "gpcrdb_column": gpcrdb_column,
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }

    _metadata_json_path(output_npz).write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_audit_csv(
    audit_csv: Path,
    records: list[dict[str, str]],
) -> None:
    fieldnames = [
        "gpcrdb_entry_name",
        "uniprot_accession",
        "uniprot_id",
        "sequence_length",
        "n_chunks",
        "n_windows",
    ]

    with audit_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in records:
            writer.writerow({field: row[field] for field in fieldnames})


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


def _region_extra_arrays(view_rows: list[EmbeddingViews]) -> dict[str, np.ndarray]:
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

        count_rows.append([
            int(counts_obj.get(name, 0))
            for name in region_names
        ])

    counts = np.array(count_rows, dtype=np.int16)

    return {
        "region_names": np.array(region_names, dtype=str),
        "region_residue_counts": counts,
        "region_missing_mask": counts == 0,
    }


def _views_for_result(
    *,
    entry_name: str,
    result: ResidueEmbeddingResult,
    region_cache: dict[str, RegionMasks] | None,
) -> EmbeddingViews:
    global_views = build_global_views(result)

    if region_cache is None:
        return global_views

    masks = region_cache.get(entry_name)
    if masks is None:
        masks = empty_region_masks(
            entry_name,
            result.residue_embeddings.shape[0],
        )
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


# TODO: Change output artifact from csv to npz for ml flow
@app.command()
def coupling_map(
    input_csv: Path = typer.Argument(..., help="GPCR common coupling map CSV."),
    output_npz: Path = typer.Argument(..., help="Output compressed NPZ embedding artifact."),
    gpcrdb_column: str = typer.Option("GPCRdb", help="Column containing GPCRdb URLs/entry names."),
    model_id: str = typer.Option(DEFAULT_MODEL_ID, help="Hugging Face ESM-2 model id."),
    batch_size: int = typer.Option(8, min=1, help="Sequences per embedding batch."),
    device: str = typer.Option("auto", help="auto, cuda, mps, or cpu."),
    max_residues: int = typer.Option(DEFAULT_MAX_RESIDUES, min=1, help="Maximum residues accepted by the model."),
    stride: int | None = typer.Option(
        None,
        min=1,
        help="Window stride for residue reconstruction. Default none means non-overlapping windows.",
    ),
    region_mask_json: Path | None = typer.Option(
        None,
        "--region-mask-json",
        help="Optional local region-mask cache JSON for building region-aware embedding views.",
    ),
    write_audit_csv: bool = typer.Option(
        True,
        "--audit/--no-audit",
        help="Write a small output audit CSV, containing protein entry name, uniprot accession and id, and number of embedding chunks.",
    ),
    write_metadata_json: bool = typer.Option(
        True,
        "--md/--no-md",
        help="""
        Write metadata containing information from the embedding model and its parameters,
        including model type, embedding shape and dimension, and output .npz dimensions.
        """,
    ),
) -> None:
    """
    Fetch UniProt sequences for unique receptors, and write embeddings.
    """

    entry_names = list(_iter_gpcrdb_entry_names(input_csv, gpcrdb_column))
    if not entry_names:
        raise typer.BadParameter("No receptor entries found in the coupling map.")

    if output_npz.suffix != ".npz":
        raise typer.BadParameter("Output path must end in .npz")

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    region_cache = load_region_cache(region_mask_json) if region_mask_json is not None else None

    tokenizer, model, torch_device = load_plm(model_id, device=device)
    typer.echo(f"Embedding {len(entry_names)} unique receptors on {torch_device}.")

    records: list[dict[str, str]] = []
    view_blocks: list[dict[str, np.ndarray]] = []
    view_rows: list[EmbeddingViews] = []
    batch_rows: list[dict[str, str]] = []

    with requests.Session() as session:
        with tqdm(
            total=len(entry_names),
            desc="Embedding receptors",
            unit="receptor",
            dynamic_ncols=True,
        ) as bar:
            for idx, entry_name in enumerate(entry_names, start=1):
                bar.set_postfix_str(f"fetch {entry_name}")

                row = fetch_uniprot_sequence(entry_name, session)
                sequence_length = len(row["sequence"])
                n_windows = _n_windows(sequence_length, max_residues, stride)

                row["sequence_length"] = str(sequence_length)
                row["n_windows"] = str(n_windows)
                row["n_chunks"] = str(n_windows)  # backward-compatible audit name

                batch_rows.append(row)

                is_full_batch = len(batch_rows) == batch_size
                is_last_batch = idx == len(entry_names)
                if not is_full_batch and not is_last_batch:
                    continue

                batch_ids = ", ".join(
                    row["gpcrdb_entry_name"] for row in batch_rows[:3]
                )
                if len(batch_rows) > 3:
                    batch_ids += ", ..."

                bar.set_postfix_str(f"embed batch: {batch_ids}")

                residue_results = embed_residue_sequences(
                    (row["sequence"] for row in batch_rows),
                    tokenizer,
                    model,
                    batch_size=batch_size,
                    max_residues=max_residues,
                    stride=stride,
                    device=torch_device,
                )

                if isinstance(residue_results, ResidueEmbeddingResult):
                    residue_results = [residue_results]

                batch_views = [
                    _views_for_result(
                        entry_name=row["gpcrdb_entry_name"],
                        result=result,
                        region_cache=region_cache,
                    )
                    for row, result in zip(batch_rows, residue_results, strict=True)
                ]

                view_names = list(batch_views[0].arrays.keys())
                view_blocks.append(stack_view_rows(batch_views, view_names))
                view_rows.extend(batch_views)
                records.extend(batch_rows)

                bar.update(len(batch_rows))
                batch_rows = []

    view_arrays = _stack_view_blocks(view_blocks)
    x_alias = view_arrays[BACKCOMPAT_EMBEDDING_KEY]

    if x_alias.shape[0] != len(records):
        raise RuntimeError(
            f"Embedding row mismatch: X has {x_alias.shape[0]} rows, "
            f"but records has {len(records)} rows."
        )

    _write_npz(
        output_npz,
        records,
        view_arrays,
        extra_arrays=_region_extra_arrays(view_rows),
    )

    audit_csv = _audit_csv_path(output_npz) if write_audit_csv else None
    if audit_csv is not None:
        _write_audit_csv(
            audit_csv,
            records,
        )

    if write_metadata_json:
        _write_metadata_json(
            output_npz,
            input_csv=input_csv,
            model_id=model_id,
            max_residues=max_residues,
            stride=stride,
            view_arrays=view_arrays,
            view_metadata=merge_view_metadata(view_rows),
            gpcrdb_column=gpcrdb_column,
            audit_csv=audit_csv,
            region_mask_json=region_mask_json,
        )

    typer.echo(f"Wrote embeddings: {output_npz}")
    if audit_csv is not None:
        typer.echo(f"Wrote audit CSV: {audit_csv}")
    if write_metadata_json:
        typer.echo(f"Wrote metadata JSON: {_metadata_json_path(output_npz)}")

"""
Typer CLI for embedding unique GPCRs from GPCRdb common coupling map.
"""

#TODO: I want to make this less GPCRdb facing. Yes, that is where data is ingested now but
# what about new assays? Needs functionality for embedding just from a UniProt accession list.

import csv
import json
import re
from datetime import UTC, datetime

import numpy as np
from collections.abc import Iterable, Iterator
from itertools import chain
from pathlib import Path
from tqdm.auto import tqdm

import requests
import typer

from glens.models.embed_model import DEFAULT_MAX_RESIDUES, DEFAULT_MODEL_ID, embed_sequences, load_plm

app = typer.Typer(no_args_is_help=True)
UNIPROT_ENTRY_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{entry_id}.fasta"
UNIPROT_ENTRY_JSON_URL = "https://rest.uniprot.org/uniprotkb/{entry_id}.json"
UNIPROT_ENTRY_RE = re.compile(r"^[a-z0-9]+_[a-z0-9]+$")
GPCRDB_ENTRY_RE = re.compile(r"/protein/([^/?#]+)")


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

            if not UNIPROT_ENTRY_RE.match(entry_name): # Skip malformed rows
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

def _n_chunks(sequence_length: int, max_residues: int) -> int:
    return (sequence_length + max_residues - 1) // max_residues

#TODO: When constructing metadata, refactor some of this so we can grab taxon and organism
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
    X: np.ndarray,
) -> None:
    payload = {
        "X": X.astype(np.float32, copy=False),
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
            [int(row["n_chunks"]) for row in records],
            dtype=np.int16,
        ),
    }

    np.savez_compressed(output_npz, **payload)


def _write_metadata_json(
    output_npz: Path,
    *,
    input_csv: Path,
    model_id: str,
    max_residues: int,
    X: np.ndarray,
    gpcrdb_column: str,
    audit_csv: Path | None,
) -> None:
    metadata = {
        "source_csv": str(input_csv),
        "output_npz": str(output_npz),
        "audit_csv": str(audit_csv) if audit_csv is not None else None,
        "model_id": model_id,
        "embedding_key": "X",
        "embedding_shape": list(X.shape),
        "embedding_dim": int(X.shape[1]),
        "dtype": str(X.dtype),
        "pooling": "mean_residue_tokens",
        "chunking": "non_overlapping_length_weighted",
        "max_residues": max_residues,
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
    ]

    with audit_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in records:
            writer.writerow({field: row[field] for field in fieldnames})


#TODO: Change output artifact from csv to npz for ml flow
@app.command()
def coupling_map(
    input_csv: Path = typer.Argument(..., help="GPCR common coupling map CSV."),
    output_npz: Path = typer.Argument(..., help="Output compressed NPZ embedding artifact."),
    gpcrdb_column: str = typer.Option("GPCRdb", help="Column containing GPCRdb URLs/entry names."),
    model_id: str = typer.Option(DEFAULT_MODEL_ID, help="Hugging Face ESM-2 model id."),
    batch_size: int = typer.Option(8, min=1, help="Sequences per embedding batch."),
    device: str = typer.Option("auto", help="auto, cuda, mps, or cpu."),
    max_residues: int = typer.Option(DEFAULT_MAX_RESIDUES, min=1, help="Maximum residues accepted by the model."), # is this even useful?
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

    tokenizer, model, torch_device = load_plm(model_id, device=device)
    typer.echo(f"Embedding {len(entry_names)} unique receptors on {torch_device}.")

    records: list[dict[str, str]] = []
    embedding_blocks: list[np.ndarray] = []
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
                row["sequence_length"] = str(sequence_length)
                row["n_chunks"] = str(_n_chunks(sequence_length, max_residues))

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

                embeddings = embed_sequences(
                    (row["sequence"] for row in batch_rows),
                    tokenizer,
                    model,
                    batch_size=batch_size,
                    max_residues=max_residues,
                    device=torch_device,
                )

                embedding_blocks.append(np.asarray(embeddings, dtype=np.float32))
                records.extend(batch_rows)

                bar.update(len(batch_rows))
                batch_rows = []

    if not embedding_blocks:
        raise RuntimeError("No embeddings were produced.")

    X = np.vstack(embedding_blocks).astype(np.float32, copy=False)

    if X.shape[0] != len(records):
        raise RuntimeError(
            f"Embedding row mismatch: X has {X.shape[0]} rows, "
            f"but records has {len(records)} rows."
        )

    _write_npz(
        output_npz,
        records,
        X,
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
            X=X,
            gpcrdb_column=gpcrdb_column,
            audit_csv=audit_csv,
        )

    typer.echo(f"Wrote embeddings: {output_npz}")
    if audit_csv is not None:
        typer.echo(f"Wrote audit CSV: {audit_csv}")
    if write_metadata_json:
        typer.echo(f"Wrote metadata JSON: {_metadata_json_path(output_npz)}")
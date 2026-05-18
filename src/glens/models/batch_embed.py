"""
Typer CLI for embedding unique GPCRs from GPCRdb common coupling map.
"""

#TODO: I want to make this less GPCRdb facing. Yes, that is where data is ingested now but
# what about new assays? Needs functionality for embedding just from a UniProt accession list.

from __future__ import annotations

import csv
import re
import numpy as np
from collections.abc import Iterable, Iterator
from itertools import chain
from pathlib import Path
from typing import Any
from tqdm.auto import tqdm

import requests
import typer

from glens.models.embed_model import DEFAULT_MAX_RESIDUES, DEFAULT_MODEL_ID, embed_sequences, load_plm

app = typer.Typer(no_args_is_help=True)
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
GPCRDB_ENTRY_RE = re.compile(r"/protein/([^/?#]+)")


def _clean_header(value: str) -> str:
    return value.strip().lstrip("\ufeff")


def _column_index(row: list[str], name: str) -> int | None:
    cleaned = [_clean_header(cell) for cell in row]
    return cleaned.index(name) if name in cleaned else None


def iter_gpcrdb_entry_names(path: Path, column: str = "GPCRdb") -> Iterator[str]:
    """Yield unique GPCRdb/UniProt-style entry names from a coupling map CSV."""
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
            entry_name = match.group(1) if match else value
            entry_name = entry_name.strip().lower()
            if not entry_name or entry_name in seen:
                continue
            seen.add(entry_name)
            yield entry_name


def fetch_uniprot_sequence(
    entry_name: str,
    session: requests.Session,
    timeout: float = 30.0,
) -> dict[str, str]:
    """Resolve a GPCRdb entry name like ``5ht1a_human`` to a UniProt sequence."""
    params = {
        "query": f"id:{entry_name.upper()}",
        "fields": "accession,id,sequence",
        "format": "json",
        "size": 1,
    }
    response = session.get(UNIPROT_SEARCH_URL, params=params, timeout=timeout)
    response.raise_for_status()
    results: list[dict[str, Any]] = response.json().get("results", [])

    if not results:
        params["query"] = f"{entry_name.upper()} AND organism_id:9606"
        response = session.get(UNIPROT_SEARCH_URL, params=params, timeout=timeout)
        response.raise_for_status()
        results = response.json().get("results", [])

    if not results:
        raise LookupError(f"No UniProt sequence found for {entry_name!r}")

    record = results[0]
    sequence = record.get("sequence", {}).get("value")
    if not sequence:
        raise LookupError(f"UniProt record for {entry_name!r} did not include a sequence")

    return {
        "gpcrdb_entry_name": entry_name,
        "uniprot_accession": record.get("primaryAccession", ""),
        "uniprot_id": record.get("uniProtkbId", entry_name.upper()),
        "sequence": sequence,
    }


def _embedding_header(dim: int) -> list[str]:
    return [f"esm2_{i:04d}" for i in range(dim)]


def _write_embedding_batch(
    writer: csv.DictWriter[str],
    rows: list[dict[str, str]],
    embeddings: np.ndarray,
    include_sequence: bool,
) -> None:
    for row, emb in zip(rows, embeddings, strict=True):
        out: dict[str, str | float] = {
            "gpcrdb_entry_name": row["gpcrdb_entry_name"],
            "uniprot_accession": row["uniprot_accession"],
            "uniprot_id": row["uniprot_id"],
        }
        if include_sequence:
            out["sequence"] = row["sequence"]
        out.update({f"esm2_{i:04d}": float(value) for i, value in enumerate(emb)})
        writer.writerow(out)


@app.command()
def coupling_map(
    input_csv: Path = typer.Argument(..., help="GPCR common coupling map CSV."),
    output_csv: Path = typer.Argument(..., help="Output embedding CSV."),
    gpcrdb_column: str = typer.Option("GPCRdb", help="Column containing GPCRdb URLs/entry names."),
    model_id: str = typer.Option(DEFAULT_MODEL_ID, help="Hugging Face ESM-2 model id."),
    batch_size: int = typer.Option(8, min=1, help="Sequences per embedding batch."),
    device: str = typer.Option("auto", help="auto, cuda, mps, or cpu."),
    max_residues: int = typer.Option(DEFAULT_MAX_RESIDUES, min=1, help="Maximum residues accepted by the model."), # is this even useful?
    include_sequence: bool = typer.Option(False, help="Include fetched sequences in the output CSV."),
) -> None:
    """Fetch UniProt sequences for unique receptors, and write embeddings."""
    entry_names = list(iter_gpcrdb_entry_names(input_csv, gpcrdb_column))
    if not entry_names:
        raise typer.BadParameter("No receptor entries found in the coupling map.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tokenizer, model, torch_device = load_plm(model_id, device=device)
    typer.echo(f"Embedding {len(entry_names)} unique receptors on {torch_device}.")

    writer: csv.DictWriter[str] | None = None
    batch_rows: list[dict[str, str]] = []

    with requests.Session() as session, output_csv.open("w", newline="") as handle:
        with tqdm(
            total=len(entry_names),
            desc="Embedding receptors",
            unit="receptor",
            dynamic_ncols=True,
        ) as bar:
            for idx, entry_name in enumerate(entry_names, start=1):
                bar.set_postfix_str(f"fetch {entry_name}")
                batch_rows.append(fetch_uniprot_sequence(entry_name, session))

                is_full_batch = len(batch_rows) == batch_size
                is_last_batch = idx == len(entry_names)
                if not is_full_batch and not is_last_batch:
                    continue

                batch_ids = ", ".join(row["gpcrdb_entry_name"] for row in batch_rows[:3])
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

                if writer is None:
                    meta = ["gpcrdb_entry_name", "uniprot_accession", "uniprot_id"]
                    if include_sequence:
                        meta.append("sequence")

                    writer = csv.DictWriter(
                        handle,
                        fieldnames=meta + _embedding_header(embeddings.shape[1]),
                    )
                    writer.writeheader()

                bar.set_postfix_str("write output")
                _write_embedding_batch(writer, batch_rows, embeddings, include_sequence)

                bar.update(len(batch_rows))
                batch_rows = []
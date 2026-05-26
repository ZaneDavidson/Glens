"""Source-agnostic sequence records and sequence-table readers."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SequenceRecord:
    """One amino-acid sequence plus optional source metadata.

    The embedding layer consumes these records without caring whether they came
    from GPCRdb/UniProt, a FASTA-derived WT sequence, a mutant sequence table,
    or a future assay source.
    """

    sequence_id: str
    sequence: str
    source: str = "sequence_table"
    gpcrdb_entry_name: str | None = None
    uniprot_accession: str | None = None
    uniprot_id: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)


def read_sequence_table_csv(
    path: Path,
    *,
    id_column: str = "sequence_id",
    sequence_column: str = "sequence",
    default_gpcrdb_entry_name: str | None = None,
) -> tuple[SequenceRecord, ...]:
    """Read a generic sequence table into ``SequenceRecord`` rows.

    Required columns are controlled by ``id_column`` and ``sequence_column``.
    Optional recognized metadata columns are:

    - source
    - gpcrdb_entry_name
    - uniprot_accession
    - uniprot_id

    All remaining columns are preserved in ``SequenceRecord.metadata`` as
    strings. This keeps WT/mutant design sequence tables auditable without
    adding design-specific code to the embedding package.
    """
    records: list[SequenceRecord] = []
    seen_ids: set[str] = set()

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header.")

        fieldnames = [_clean_header(name) for name in reader.fieldnames]
        raw_to_clean = dict(zip(reader.fieldnames, fieldnames, strict=True))

        if id_column not in fieldnames:
            raise ValueError(
                f"CSV is missing id column {id_column!r}. "
                f"Available columns: {', '.join(fieldnames)}"
            )
        if sequence_column not in fieldnames:
            raise ValueError(
                f"CSV is missing sequence column {sequence_column!r}. "
                f"Available columns: {', '.join(fieldnames)}"
            )

        known = {
            id_column,
            sequence_column,
            "source",
            "gpcrdb_entry_name",
            "uniprot_accession",
            "uniprot_id",
        }

        for row_number, raw_row in enumerate(reader, start=2):
            row = {
                raw_to_clean[key]: (value or "").strip()
                for key, value in raw_row.items()
                if key is not None
            }
            sequence_id = row.get(id_column, "").strip()
            sequence = normalize_sequence(row.get(sequence_column, ""))

            if not sequence_id:
                raise ValueError(f"Row {row_number} is missing {id_column!r}.")
            if sequence_id in seen_ids:
                raise ValueError(f"Duplicate sequence id {sequence_id!r} at row {row_number}.")
            if not sequence:
                raise ValueError(f"Row {row_number} has an empty sequence.")

            seen_ids.add(sequence_id)
            gpcrdb_entry_name = row.get("gpcrdb_entry_name") or default_gpcrdb_entry_name
            metadata = {
                key: value
                for key, value in row.items()
                if key not in known and value != ""
            }

            records.append(
                SequenceRecord(
                    sequence_id=sequence_id,
                    sequence=sequence,
                    source=row.get("source") or "sequence_table",
                    gpcrdb_entry_name=gpcrdb_entry_name,
                    uniprot_accession=row.get("uniprot_accession") or None,
                    uniprot_id=row.get("uniprot_id") or None,
                    metadata=metadata,
                )
            )

    if not records:
        raise ValueError(f"No sequence rows were found in {path}.")

    return tuple(records)


def normalize_sequence(sequence: str) -> str:
    """Remove whitespace and uppercase a sequence string."""
    return "".join(str(sequence).split()).upper()


def _clean_header(value: str) -> str:
    return str(value).strip().lstrip("\ufeff")

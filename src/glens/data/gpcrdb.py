"""GPCRdb/common-coupling-map parsing helpers."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Iterator, Sequence
from itertools import chain
from pathlib import Path

UNIPROT_ENTRY_RE = re.compile(r"^[a-z0-9]+_[a-z0-9]+$")
GPCRDB_ENTRY_RE = re.compile(r"/protein/([^/?#]+)")


def iter_gpcrdb_entry_names(path: Path, column: str = "GPCRdb") -> Iterator[str]:
    """Yield unique GPCRdb/UniProt-style entry names from a coupling-map CSV.

    GPCRdb common coupling maps use a two-row header where the first row is a
    group label and the second row contains names like ``GPCRdb``. This parser
    also supports ordinary one-row CSVs with a direct ``GPCRdb`` column.
    """
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        first = next(reader, None)
        if first is None:
            return

        second = next(reader, None)
        second_idx = _column_index(second, column) if second is not None else None
        first_idx = _column_index(first, column)

        if second_idx is not None:
            idx = second_idx
            rows: Iterable[Sequence[str] | None] = reader
        elif first_idx is not None:
            idx = first_idx
            rows = chain([second], reader) if second is not None else reader
        else:
            raise ValueError(f"Column {column!r} not found in {path}")

        seen: set[str] = set()
        for row in rows:
            if row is None or idx >= len(row):
                continue

            entry_name = entry_name_from_gpcrdb_value(row[idx])
            if entry_name is None or entry_name in seen:
                continue

            seen.add(entry_name)
            yield entry_name


def entry_name_from_gpcrdb_value(value: str) -> str | None:
    """Extract a normalized GPCRdb/UniProt entry name from a cell value."""
    text = str(value).strip()
    match = GPCRDB_ENTRY_RE.search(text)
    entry = (match.group(1) if match else text).strip().lower()
    return entry if UNIPROT_ENTRY_RE.match(entry) else None


def _column_index(row: Sequence[str] | None, name: str) -> int | None:
    if row is None:
        return None
    cleaned = [_clean_header(cell) for cell in row]
    return cleaned.index(name) if name in cleaned else None


def _clean_header(value: str) -> str:
    return str(value).strip().lstrip("\ufeff")

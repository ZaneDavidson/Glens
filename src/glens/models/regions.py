"""
GPCR region mask utilities for region-aware ESM embedding views.

This module depends on a cache built from GPCRdb residue annotations, which can be generated with the `gpcrdb-cache` command.
This cache is a .json file containing masks for each relevent GPCR region per receptor. A REST client is included as a convenience,
but embedding runs should be cache-dependent only.
"""

import csv
import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import requests
import typer

app = typer.Typer(no_args_is_help=True)

GPCRDB_RESIDUES_URL = "https://gpcrdb.org/services/residues/{entry_name}/"
UNIPROT_ENTRY_RE = re.compile(r"^[a-z0-9]+_[a-z0-9]+$")
GPCRDB_ENTRY_RE = re.compile(r"/protein/([^/?#]+)")

BASE_SEGMENTS = (
    "N-term",
    "TM1",
    "ICL1",
    "TM2",
    "ECL1",
    "TM3",
    "ICL2",
    "TM4",
    "ECL2",
    "TM5",
    "ICL3",
    "TM6",
    "ECL3",
    "TM7",
    "H8",
    "C-term",
)

TM_SEGMENTS = ("TM1", "TM2", "TM3", "TM4", "TM5", "TM6", "TM7")
ICL_SEGMENTS = ("ICL1", "ICL2", "ICL3")
ECL_SEGMENTS = ("ECL1", "ECL2", "ECL3")
CORE_SEGMENTS = TM_SEGMENTS + ICL_SEGMENTS + ECL_SEGMENTS + ("H8",)

DERIVED_REGIONS = (
    "7tm_core",
    "tm_all",
    "icl_all",
    "intracellular",
    "tm3_cyt",
    "tm5_cyt",
    "tm6_cyt",
    "tm7_cyt",
    "cytoplasmic_tm_ends",
    "coupling_face",
)

REGION_NAMES = BASE_SEGMENTS + DERIVED_REGIONS


@dataclass(frozen=True)
class SegmentSpan:
    name: str
    start: int  # zero-based, inclusive
    end: int  # zero-based, exclusive
    source: str = "gpcrdb_residues"


@dataclass(frozen=True)
class RegionMasks:
    gpcrdb_entry_name: str
    sequence_length: int
    masks: dict[str, np.ndarray]
    spans: list[SegmentSpan]
    missing_regions: list[str]
    source: str = "gpcrdb_residues"


def _clean_header(value: str) -> str:
    return value.strip().lstrip("\ufeff")


def _column_index(row: list[str], name: str) -> int | None:
    cleaned = [_clean_header(cell) for cell in row]
    return cleaned.index(name) if name in cleaned else None


def _iter_gpcrdb_entry_names(path: Path, column: str = "GPCRdb") -> Iterator[str]:
    """Yield unique GPCRdb/UniProt-style entry names from a two-row coupling map."""
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

            if not UNIPROT_ENTRY_RE.match(entry_name):
                continue
            if entry_name in seen:
                continue

            seen.add(entry_name)
            yield entry_name


def _field(record: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in record and record[name] not in (None, ""):
            return record[name]
    return None


def _segment_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("slug", "name", "display_name", "label"):
            inner = value.get(key)
            if inner not in (None, ""):
                return str(inner)
    return str(value)


def normalize_segment_name(value: Any) -> str | None:
    """Normalize GPCRdb/annotation segment labels to a compact vocabulary."""
    text = _segment_text(value)
    if text is None:
        return None

    raw = text.strip()
    if not raw:
        return None

    compact = raw.upper().replace(" ", "").replace("-", "").replace("_", "")

    # Common GPCRdb labels are already TM1, ICL2, ECL3, H8, etc.
    for idx in range(1, 8):
        if compact in {f"TM{idx}", f"TRANSMEMBRANE{idx}", f"TRANSMEMBRANESEGMENT{idx}"}:
            return f"TM{idx}"
    for idx in range(1, 4):
        if compact in {f"ICL{idx}", f"INTRACELLULARLOOP{idx}", f"INTRACELLULAR{idx}"}:
            return f"ICL{idx}"
        if compact in {f"ECL{idx}", f"EXTRACELLULARLOOP{idx}", f"EXTRACELLULAR{idx}"}:
            return f"ECL{idx}"

    if compact in {"H8", "HELIX8", "HELIXVIII"}:
        return "H8"
    if compact in {"NTERM", "NTERMINUS", "NTERMINAL", "NTERMINALTERMINUS"}:
        return "N-term"
    if compact in {"CTERM", "CTERMINUS", "CTERMINAL", "CTERMINALTERMINUS"}:
        return "C-term"

    return None


def _sequence_number(record: Mapping[str, Any]) -> int | None:
    value = _field(record, "sequence_number", "seq_number", "position", "pos")
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _record_segment(record: Mapping[str, Any]) -> str | None:
    value = _field(record, "protein_segment", "segment", "segment_slug", "protein_segment_slug")
    return normalize_segment_name(value)


def _empty_masks(sequence_length: int) -> dict[str, np.ndarray]:
    return {
        name: np.zeros(sequence_length, dtype=bool)
        for name in REGION_NAMES
    }


def _spans_from_mask(name: str, mask: np.ndarray, *, source: str) -> list[SegmentSpan]:
    spans: list[SegmentSpan] = []
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return spans

    start = int(indices[0])
    prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        spans.append(SegmentSpan(name=name, start=start, end=prev + 1, source=source))
        start = idx
        prev = idx

    spans.append(SegmentSpan(name=name, start=start, end=prev + 1, source=source))
    return spans


def _terminal_slice(mask: np.ndarray, *, side: str, window: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return out

    if side == "first":
        keep = indices[:window]
    elif side == "last":
        keep = indices[-window:]
    else:
        raise ValueError(f"Unsupported side: {side}")

    out[keep] = True
    return out


def _derive_regions(masks: dict[str, np.ndarray], *, cyt_window: int) -> None:
    masks["tm_all"] = np.logical_or.reduce([masks[name] for name in TM_SEGMENTS])
    masks["icl_all"] = np.logical_or.reduce([masks[name] for name in ICL_SEGMENTS])

    masks["7tm_core"] = np.logical_or.reduce([masks[name] for name in CORE_SEGMENTS])
    masks["intracellular"] = np.logical_or.reduce([
        masks["ICL1"],
        masks["ICL2"],
        masks["ICL3"],
        masks["H8"],
    ])

    masks["tm3_cyt"] = _terminal_slice(masks["TM3"], side="last", window=cyt_window)
    masks["tm5_cyt"] = _terminal_slice(masks["TM5"], side="last", window=cyt_window)
    masks["tm6_cyt"] = _terminal_slice(masks["TM6"], side="first", window=cyt_window)
    masks["tm7_cyt"] = _terminal_slice(masks["TM7"], side="last", window=cyt_window)

    masks["cytoplasmic_tm_ends"] = np.logical_or.reduce([
        masks["tm3_cyt"],
        masks["tm5_cyt"],
        masks["tm6_cyt"],
        masks["tm7_cyt"],
    ])

    masks["coupling_face"] = np.logical_or.reduce([
        masks["ICL2"],
        masks["ICL3"],
        masks["H8"],
        masks["cytoplasmic_tm_ends"],
    ])


def region_masks_from_gpcrdb_residues(
    gpcrdb_entry_name: str,
    residue_records: Sequence[Mapping[str, Any]],
    *,
    sequence_length: int | None = None,
    cyt_window: int = 10,
    source: str = "gpcrdb_residues",
) -> RegionMasks:
    """Convert GPCRdb residue records to base and derived boolean masks."""
    positions: list[tuple[int, str]] = []
    max_position = 0

    for record in residue_records:
        number = _sequence_number(record)
        if number is None:
            continue
        max_position = max(max_position, number)

        segment = _record_segment(record)
        if segment is None:
            continue
        positions.append((number - 1, segment))

    inferred_length = max_position if sequence_length is None else int(sequence_length)
    if inferred_length <= 0:
        raise ValueError(f"Could not infer sequence length for {gpcrdb_entry_name!r}.")

    masks = _empty_masks(inferred_length)

    for zero_based, segment in positions:
        if 0 <= zero_based < inferred_length:
            masks[segment][zero_based] = True

    _derive_regions(masks, cyt_window=cyt_window)

    spans: list[SegmentSpan] = []
    for name in BASE_SEGMENTS:
        spans.extend(_spans_from_mask(name, masks[name], source=source))

    missing_regions = [
        name
        for name in REGION_NAMES
        if not np.any(masks[name])
    ]

    return RegionMasks(
        gpcrdb_entry_name=gpcrdb_entry_name,
        sequence_length=inferred_length,
        masks=masks,
        spans=spans,
        missing_regions=missing_regions,
        source=source,
    )


def empty_region_masks(
    gpcrdb_entry_name: str,
    sequence_length: int,
    *,
    source: str = "missing_region_cache",
) -> RegionMasks:
    masks = _empty_masks(sequence_length)
    return RegionMasks(
        gpcrdb_entry_name=gpcrdb_entry_name,
        sequence_length=sequence_length,
        masks=masks,
        spans=[],
        missing_regions=list(REGION_NAMES),
        source=source,
    )


def reconcile_length_arguments(
    masks: RegionMasks,
    sequence_length: int,
    *,
    allow_n_terminal_trim: bool = True,
) -> RegionMasks:
    """Reconcile cached region coordinates to the actual length of the sequence to embed.

    The cache source (GPCRdb) may include an N-terminal extension that UNIPROT omits.
    When the cache is longer than the actual sequence, support a conservative
    N-terminal trim. Old coordinate offset idx maps to new coordinate 0.
    """
    sequence_length = int(sequence_length)

    if masks.sequence_length == sequence_length:
        return masks

    if masks.sequence_length < sequence_length:
        raise ValueError(
            f"Region mask length mismatch for {masks.gpcrdb_entry_name}: "
            f"cache has {masks.sequence_length}, embedded sequence has {sequence_length}. "
            "Cache is shorter than the embedded sequence; refusing to pad masks."
        )

    if not allow_n_terminal_trim:
        raise ValueError(
            f"Region mask length mismatch for {masks.gpcrdb_entry_name}: "
            f"cache has {masks.sequence_length}, embedded sequence has {sequence_length}."
        )

    offset = masks.sequence_length - sequence_length

    trimmed_masks: dict[str, np.ndarray] = {}
    for name, mask in masks.masks.items():
        trimmed_masks[name] = mask[offset : offset + sequence_length].astype(
            bool,
            copy=False,
        )

        if trimmed_masks[name].shape != (sequence_length,):
            raise RuntimeError(
                f"Internal trimming error for {masks.gpcrdb_entry_name} region {name}: "
                f"got {trimmed_masks[name].shape}, expected {(sequence_length,)}."
            )

    trimmed_spans: list[SegmentSpan] = []
    for span in masks.spans:
        start = max(0, span.start - offset)
        end = min(sequence_length, span.end - offset)

        if end <= start:
            continue

        trimmed_spans.append(
            SegmentSpan(
                name=span.name,
                start=start,
                end=end,
                source=f"{span.source}|nterm_trim_{offset}",
            )
        )

    missing_regions = [
        name
        for name in REGION_NAMES
        if not np.any(trimmed_masks[name])
    ]

    return RegionMasks(
        gpcrdb_entry_name=masks.gpcrdb_entry_name,
        sequence_length=sequence_length,
        masks=trimmed_masks,
        spans=trimmed_spans,
        missing_regions=missing_regions,
        source=f"{masks.source}|nterm_trim_{offset}",
    )


def validate_region_masks(masks: RegionMasks, *, sequence_length: int | None = None) -> None:
    expected_length = masks.sequence_length if sequence_length is None else int(sequence_length)
    if masks.sequence_length != expected_length:
        raise ValueError(
            f"Region mask length mismatch for {masks.gpcrdb_entry_name}: "
            f"cache has {masks.sequence_length}, expected {expected_length}."
        )

    for name, mask in masks.masks.items():
        if mask.shape != (expected_length,):
            raise ValueError(
                f"Mask {name!r} for {masks.gpcrdb_entry_name} has shape {mask.shape}, "
                f"expected {(expected_length,)}."
            )

    tm_stack = np.vstack([masks.masks[name] for name in TM_SEGMENTS])
    if np.any(tm_stack.sum(axis=0) > 1):
        raise ValueError(f"Overlapping TM segment masks for {masks.gpcrdb_entry_name}.")

    loop_stack = np.vstack([masks.masks[name] for name in ICL_SEGMENTS + ECL_SEGMENTS])
    if np.any(tm_stack.sum(axis=0) & loop_stack.sum(axis=0)):
        raise ValueError(f"TM and loop masks overlap for {masks.gpcrdb_entry_name}.")


def _mask_to_ranges(mask: np.ndarray) -> list[list[int]]:
    return [[span.start, span.end] for span in _spans_from_mask("", mask, source="cache")]


def _ranges_to_mask(ranges: Sequence[Sequence[int]], sequence_length: int) -> np.ndarray:
    mask = np.zeros(sequence_length, dtype=bool)
    for start, end in ranges:
        start_i = int(start)
        end_i = int(end)
        if start_i < 0 or end_i < start_i or end_i > sequence_length:
            raise ValueError(f"Invalid region range [{start_i}, {end_i}) for length {sequence_length}.")
        mask[start_i:end_i] = True
    return mask


def region_masks_to_cache_record(masks: RegionMasks) -> dict[str, Any]:
    return {
        "gpcrdb_entry_name": masks.gpcrdb_entry_name,
        "sequence_length": masks.sequence_length,
        "source": masks.source,
        "regions": {
            name: _mask_to_ranges(mask)
            for name, mask in masks.masks.items()
            if np.any(mask)
        },
        "spans": [asdict(span) for span in masks.spans],
        "missing_regions": list(masks.missing_regions),
    }


def region_masks_from_cache_record(record: Mapping[str, Any]) -> RegionMasks:
    entry = str(record["gpcrdb_entry_name"])
    sequence_length = int(record["sequence_length"])
    masks = _empty_masks(sequence_length)

    for name, ranges in dict(record.get("regions", {})).items():
        if name not in masks:
            continue
        masks[name] = _ranges_to_mask(ranges, sequence_length)

    spans = [
        SegmentSpan(
            name=str(span["name"]),
            start=int(span["start"]),
            end=int(span["end"]),
            source=str(span.get("source", record.get("source", "cache"))),
        )
        for span in record.get("spans", [])
    ]

    missing_regions = [
        name
        for name in REGION_NAMES
        if not np.any(masks[name])
    ]

    return RegionMasks(
        gpcrdb_entry_name=entry,
        sequence_length=sequence_length,
        masks=masks,
        spans=spans,
        missing_regions=missing_regions,
        source=str(record.get("source", "cache")),
    )


def load_region_cache(path: Path) -> dict[str, RegionMasks]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("receptors", {})
    if isinstance(records, list):
        iterable = records
    else:
        iterable = records.values()

    cache = {
        str(record["gpcrdb_entry_name"]): region_masks_from_cache_record(record)
        for record in iterable
    }

    for masks in cache.values():
        validate_region_masks(masks)

    return cache


def write_region_cache(
    path: Path,
    masks_by_entry: Mapping[str, RegionMasks],
    *,
    source: str,
    cyt_window: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source": source,
        "cyt_window": cyt_window,
        "region_names": list(REGION_NAMES),
        "base_segments": list(BASE_SEGMENTS),
        "derived_regions": list(DERIVED_REGIONS),
        "receptors": {
            entry: region_masks_to_cache_record(masks)
            for entry, masks in sorted(masks_by_entry.items())
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def fetch_gpcrdb_residues(
    entry_name: str,
    session: requests.Session,
    *,
    base_url: str = GPCRDB_RESIDUES_URL,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    url = base_url.format(entry_name=entry_name)
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise TypeError(f"Expected a list from {url}, got {type(data).__name__}.")
    return data


@app.command("gpcrdb-cache")
def gpcrdb_cache(
    input_csv: Path = typer.Argument(..., help="Coupling map CSV containing GPCRdb entry names."),
    output_json: Path = typer.Argument(..., help="Output region-mask cache JSON."),
    gpcrdb_column: str = typer.Option("GPCRdb", help="Column containing GPCRdb URLs/entry names."),
    cyt_window: int = typer.Option(10, min=1, help="Residues from cytoplasmic TM ends."),
    timeout: float = typer.Option(30.0, min=1.0, help="Request timeout in seconds."),
) -> None:
    """Build a local region-mask cache from GPCRdb residue annotations."""
    entry_names = list(_iter_gpcrdb_entry_names(input_csv, gpcrdb_column))
    if not entry_names:
        raise typer.BadParameter("No receptor entries found in the input CSV.")

    masks_by_entry: dict[str, RegionMasks] = {}
    failures: dict[str, str] = {}

    with requests.Session() as session:
        for entry_name in entry_names:
            try:
                records = fetch_gpcrdb_residues(entry_name, session, timeout=timeout)
                masks = region_masks_from_gpcrdb_residues(
                    entry_name,
                    records,
                    cyt_window=cyt_window,
                )
                validate_region_masks(masks)
                masks_by_entry[entry_name] = masks
            except Exception as err:  # pragma: no cover - command diagnostics
                failures[entry_name] = str(err)

    if failures:
        typer.echo(f"Warning: failed to fetch masks for {len(failures)} receptors.")
        for entry_name, message in list(failures.items())[:10]:
            typer.echo(f"  {entry_name}: {message}")

    if not masks_by_entry:
        raise RuntimeError("No region masks were built.")

    write_region_cache(
        output_json,
        masks_by_entry,
        source="gpcrdb_rest_residues",
        cyt_window=cyt_window,
    )
    typer.echo(f"Wrote region mask cache: {output_json}")
    typer.echo(f"Cached receptors: {len(masks_by_entry)}")


@app.command("validate-cache")
def validate_cache(
    cache_json: Path = typer.Argument(..., help="Region-mask cache JSON."),
) -> None:
    """Validate a local region-mask cache and print a compact summary."""
    cache = load_region_cache(cache_json)
    n = len(cache)
    missing_counts: dict[str, int] = {name: 0 for name in REGION_NAMES}
    residue_counts: dict[str, list[int]] = {name: [] for name in REGION_NAMES}

    for masks in cache.values():
        validate_region_masks(masks)
        for name in REGION_NAMES:
            count = int(np.sum(masks.masks[name]))
            residue_counts[name].append(count)
            if count == 0:
                missing_counts[name] += 1

    typer.echo(f"Validated region cache: {cache_json}")
    typer.echo(f"Receptors: {n}")
    for name in ("7tm_core", "intracellular", "cytoplasmic_tm_ends", "coupling_face", "ICL2", "ICL3", "H8"):
        counts = np.array(residue_counts[name], dtype=np.int32)
        typer.echo(
            f"{name}: missing={missing_counts[name]}, "
            f"min={int(counts.min())}, mean={float(counts.mean()):.1f}, max={int(counts.max())}"
        )

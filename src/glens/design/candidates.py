"""Mutation-candidate construction for design workflows.

Two complementary mutation-entry modes are implemented in this module:

1. User-specified interrogation
   A user types or uploads a list (csv) of specific mutations to interrogate.

2. Search / scan mode
   Generates many candidate single substitutions across selected
   positions or regions, then scores and ranks them in batch.
"""

import csv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from glens.design.mutations import (
    CANONICAL_AMINO_ACIDS,
    PointMutation,
    generate_single_mutants,
    mutation_label,
    parse_point_mutation,
    region_labels_by_position,
    selected_positions_from_region_masks,
)


@dataclass(frozen=True)
class MutationCandidate:
    """Mutation data and provenance for downstream scoring/reporting."""

    mutation: PointMutation
    source: str
    note: str = ""

    @property
    def label(self) -> str:
        """One-based compact mutation label, e.g. 'R135A'."""
        return mutation_label(self.mutation)

    def as_dict(self) -> dict[str, Any]:
        """Return a flat, CSV-friendly row."""
        return {
            "mutation": self.label,
            "source": self.source,
            "note": self.note,
            "sequence_index": self.mutation.sequence_index,
            "position": self.mutation.one_based_position,
            "wt_aa": self.mutation.wt_aa,
            "mutant_aa": self.mutation.mutant_aa,
            "region": "" if self.mutation.region is None else self.mutation.region,
        }


@dataclass(frozen=True)
class MutationParseError:
    """Parsing/validation error from user mutation list."""

    line_number: int
    raw_text: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "line_number": self.line_number,
            "raw_text": self.raw_text,
            "message": self.message,
        }


@dataclass(frozen=True)
class ParsedMutationList:
    """Result of parsing user mutation list."""

    candidates: tuple[MutationCandidate, ...]
    errors: tuple[MutationParseError, ...]

    @property
    def ok(self) -> bool:
        """No parse/validation errors."""
        return not self.errors

    def raise_if_errors(self) -> None:
        """Raise an ValueError if any mutation lines failed."""
        if not self.errors:
            return
        preview = "; ".join(
            f"line {error.line_number}: {error.message}"
            for error in self.errors[:5]
        )
        if len(self.errors) > 5:
            preview += f"; ... {len(self.errors) - 5} more"
        raise ValueError(preview)


def parse_point_mutation_lines(
    lines: Iterable[str],
    *,
    sequence: str | None = None,
    index_base: int = 1,
    fail_fast: bool = False,
    comment_prefix: str = "#",
    source: str = "user_list",
) -> ParsedMutationList:
    """Parse a typed/uploaded list of standard point-mutation labels.

    Blank lines and comments (full and inline) are ignored.
    """
    candidates: list[MutationCandidate] = []
    errors: list[MutationParseError] = []

    for line_number, raw_line in enumerate(lines, start=1):
        raw_text = raw_line.rstrip("\n")
        token, note = _extract_mutation_token(
            raw_text,
            allow_inline_comments=True, # messy fix, inline comments should always be ignored
            comment_prefix=comment_prefix,
        )

        if token == "":
            continue

        try:
            mutation = parse_point_mutation(
                token,
                sequence=sequence,
                index_base=index_base,
            )
        except ValueError as exc:
            error = MutationParseError(
                line_number=line_number,
                raw_text=raw_text,
                message=str(exc),
            )
            if fail_fast:
                raise ValueError(
                    f"Failed to parse mutation list at line {line_number}: {exc}"
                ) from exc
            errors.append(error)
            continue

        candidates.append(
            MutationCandidate(
                mutation=mutation,
                source=source,
                note=note,
            )
        )

    return ParsedMutationList(
        candidates=tuple(candidates),
        errors=tuple(errors),
    )


def parse_point_mutation_text(
    text: str,
    *,
    sequence: str | None = None,
    index_base: int = 1,
    fail_fast: bool = False,
    source: str = "user_list",
) -> ParsedMutationList:
    """Parse point mutations from a newline-delimited text block."""
    return parse_point_mutation_lines(
        text.splitlines(),
        sequence=sequence,
        index_base=index_base,
        fail_fast=fail_fast,
        source=source,
    )


def parse_point_mutation_file(
    path: Path,
    *,
    sequence: str | None = None,
    index_base: int = 1,
    fail_fast: bool = False,
    source: str = "user_list",
) -> ParsedMutationList:
    """Parse point mutations from a newline-delimited .txt file."""
    return parse_point_mutation_lines(
        path.read_text(encoding="utf-8").splitlines(),
        sequence=sequence,
        index_base=index_base,
        fail_fast=fail_fast,
        source=source,
    )


def parse_point_mutation_csv(
    path: Path,
    *,
    sequence: str | None = None,
    mutation_column: str = "mutation",
    note_column: str | None = None,
    index_base: int = 1,
    fail_fast: bool = False,
    source: str = "user_csv",
) -> ParsedMutationList:
    """Parse point mutations from a .csv file with a mutation column."""
    candidates: list[MutationCandidate] = []
    errors: list[MutationParseError] = []

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or mutation_column not in reader.fieldnames:
            available = ", ".join(reader.fieldnames or [])
            raise ValueError(
                f"CSV is missing mutation column {mutation_column!r}. "
                f"Available columns: {available}"
            )

        for row_number, row in enumerate(reader, start=2):
            raw = str(row.get(mutation_column, "")).strip()
            if raw == "":
                continue

            try:
                mutation = parse_point_mutation(
                    raw,
                    sequence=sequence,
                    index_base=index_base,
                )
            except ValueError as exc:
                error = MutationParseError(
                    line_number=row_number,
                    raw_text=raw,
                    message=str(exc),
                )
                if fail_fast:
                    raise ValueError(
                        f"Failed to parse mutation CSV at row {row_number}: {exc}"
                    ) from exc
                errors.append(error)
                continue

            note = "" if note_column is None else str(row.get(note_column, "")).strip()
            candidates.append(
                MutationCandidate(
                    mutation=mutation,
                    source=source,
                    note=note,
                )
            )

    return ParsedMutationList(
        candidates=tuple(candidates),
        errors=tuple(errors),
    )


def build_single_mutation_scan_candidates(
    sequence: str,
    *,
    positions: Sequence[int] | None = None,
    region_masks: Mapping[str, Sequence[bool] | np.ndarray] | None = None,
    selected_regions: Sequence[str] | None = None,
    amino_acids: Sequence[str] = CANONICAL_AMINO_ACIDS,
    source: str = "single_mutant_scan",
) -> tuple[MutationCandidate, ...]:
    """Generate mutation candidates for a batched single-mutation scan.

    Mutations are generated across selected positions or regions. Mutants will preserve their region annotation.
    """
    if selected_regions is not None and region_masks is None:
        raise ValueError("region_masks is required when selected_regions is passed.")

    region_by_position: dict[int, str] | None = None
    scan_positions = None if positions is None else list(positions)

    if region_masks is not None:
        region_by_position = region_labels_by_position(
            region_masks,
            selected_regions=selected_regions,
        )
        region_positions = selected_positions_from_region_masks(
            region_masks,
            selected_regions=tuple(region_masks) if selected_regions is None else selected_regions,
        )
        if scan_positions is None:
            scan_positions = region_positions
        else:
            region_position_set = set(region_positions)
            scan_positions = [
                position for position in scan_positions if position in region_position_set
            ]

    mutations = generate_single_mutants(
        sequence,
        positions=scan_positions,
        region_by_position=region_by_position,
        amino_acids=amino_acids,
        skip_wt=True,
    )

    return tuple(
        MutationCandidate(mutation=mutation, source=source)
        for mutation in mutations
    )


def candidates_to_rows(
    candidates: Sequence[MutationCandidate],
) -> list[dict[str, Any]]:
    """Convert mutation candidates to flat dictionaries for tables."""
    return [candidate.as_dict() for candidate in candidates]


def _extract_mutation_token(
    raw_text: str,
    *,
    allow_inline_comments: bool,
    comment_prefix: str,
) -> tuple[str, str]:
    stripped = raw_text.strip()
    if stripped == "":
        return "", ""

    if stripped.startswith(comment_prefix):
        return "", ""

    note = ""
    body = stripped
    if allow_inline_comments and comment_prefix in stripped:
        body, note = stripped.split(comment_prefix, maxsplit=1)
        body = body.strip()
        note = note.strip()

    if body == "":
        return "", note

    # Allow either a bare mutation label or a simple first-token line such as
    # "R135A, literature candidate". Employ full .csv parser later?
    token = body.replace(",", " ").split()[0].strip()
    return token, note

"""WT/mutant sequence table construction for the design workflow.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from glens.design.candidates import MutationCandidate
from glens.design.mutations import (
    CANONICAL_AMINO_ACIDS,
    apply_point_mutation,
)


@dataclass(frozen=True)
class DesignSequenceRow:
    """One WT or mutant sequence row in design-batch order."""

    row_index: int
    sequence_id: str
    sequence: str
    is_wt: bool
    candidate: MutationCandidate | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a flat row suitable for CSV output."""
        base: dict[str, Any] = {
            "row_index": self.row_index,
            "sequence_id": self.sequence_id,
            "is_wt": self.is_wt,
            "sequence_length": len(self.sequence),
            "sequence": self.sequence,
        }

        if self.candidate is None:
            base.update(
                {
                    "mutation": "WT",
                    "source": "wild_type",
                    "note": "",
                    "position": "",
                    "sequence_index": "",
                    "wt_aa": "",
                    "mutant_aa": "",
                    "region": "",
                }
            )
            return base

        candidate_row = self.candidate.as_dict()
        base.update(candidate_row)
        return base


def read_sequence_file(path: Path) -> str:
    """Read a WT sequence from FASTA or plain text.
    """
    text = path.read_text(encoding="utf-8")
    return normalize_sequence_text(text)


def normalize_sequence_text(text: str) -> str:
    """Normalize FASTA/plain sequence text and validate canonical residues."""
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == "" or line.startswith(">"):
            continue
        lines.append(line)

    if not lines:
        stripped = "".join(str(text).split())
        sequence = stripped.upper()
    else:
        sequence = "".join("".join(lines).split()).upper()

    if not sequence:
        raise ValueError("WT sequence must not be empty.")

    invalid = sorted(set(sequence).difference(CANONICAL_AMINO_ACIDS))
    if invalid:
        joined = ", ".join(invalid)
        raise ValueError(f"WT sequence contains non-canonical residues: {joined}")

    return sequence


def build_wt_mutant_sequence_rows(
    *,
    wt_sequence: str,
    candidates: Sequence[MutationCandidate],
    wt_sequence_id: str = "WT",
) -> tuple[DesignSequenceRow, ...]:
    """Build canonical sequence rows with residue validation.
    """
    normalized_wt = normalize_sequence_text(wt_sequence)
    rows: list[DesignSequenceRow] = [
        DesignSequenceRow(
            row_index=0,
            sequence_id=wt_sequence_id,
            sequence=normalized_wt,
            is_wt=True,
            candidate=None,
        )
    ]

    for idx, candidate in enumerate(candidates, start=1):
        mutant_sequence = apply_point_mutation(
            normalized_wt,
            candidate.mutation,
            validate_wt=True,
        )
        rows.append(
            DesignSequenceRow(
                row_index=idx,
                sequence_id=f"{wt_sequence_id}|{candidate.label}",
                sequence=mutant_sequence,
                is_wt=False,
                candidate=candidate,
            )
        )

    return tuple(rows)


def sequence_rows_to_dicts(
    rows: Sequence[DesignSequenceRow],
) -> list[dict[str, Any]]:
    """Convert sequence rows to flat dictionaries."""
    return [row.as_dict() for row in rows]


def sequence_rows_to_fasta(
    rows: Sequence[DesignSequenceRow],
    *,
    line_width: int = 80,
) -> str:
    """Render sequence rows as FASTA text."""
    if line_width <= 0:
        raise ValueError("line_width must be positive.")

    blocks: list[str] = []
    for row in rows:
        header = row.sequence_id
        if row.candidate is not None:
            header += f" mutation={row.candidate.label}"
            if row.candidate.note:
                header += f" note={_sanitize_fasta_note(row.candidate.note)}"

        blocks.append(f">{header}")
        blocks.extend(_wrap_sequence(row.sequence, line_width=line_width))

    return "\n".join(blocks) + "\n"


def validate_sequence_rows(
    rows: Sequence[DesignSequenceRow],
) -> None:
    """Validate row-order and sequence-length invariants."""
    if not rows:
        raise ValueError("At least one sequence row is required.")
    if rows[0].row_index != 0 or not rows[0].is_wt:
        raise ValueError("First sequence row must be WT at row_index 0.")

    expected_length = len(rows[0].sequence)
    for expected_idx, row in enumerate(rows):
        if row.row_index != expected_idx:
            raise ValueError(
                f"Sequence row index mismatch: expected {expected_idx}, "
                f"got {row.row_index}."
            )
        if len(row.sequence) != expected_length:
            raise ValueError(
                f"Sequence row {row.row_index} has length {len(row.sequence)}; "
                f"expected {expected_length}."
            )
        normalize_sequence_text(row.sequence)


def _wrap_sequence(sequence: str, *, line_width: int) -> list[str]:
    return [
        sequence[start : start + line_width]
        for start in range(0, len(sequence), line_width)
    ]


def _sanitize_fasta_note(note: str) -> str:
    return " ".join(str(note).split()).replace(">", "")

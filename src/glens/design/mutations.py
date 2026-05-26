"""Point-mutation generation primitives.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

CANONICAL_AMINO_ACIDS: tuple[str, ...] = tuple("ACDEFGHIKLMNPQRSTVWY")
_MUTATION_RE = re.compile(r"^\s*([A-Za-z])(\d+)([A-Za-z])\s*$")


@dataclass(frozen=True)
class PointMutation:
    """A single amino-acid substitution.

    'sequence_index' is always zero indexed. Use :func:'mutation_label' when a
    human-readable mutation label is needed.
    """

    sequence_index: int
    wt_aa: str
    mutant_aa: str
    region: str | None = None

    def __post_init__(self) -> None:
        if self.sequence_index < 0:
            raise ValueError("sequence_index must be non-negative.")
        _validate_single_residue(self.wt_aa, field_name="wt_aa")
        _validate_single_residue(self.mutant_aa, field_name="mutant_aa")
        if self.wt_aa == self.mutant_aa:
            raise ValueError("Requested mutation must change an amino acid.")

    @property
    def one_based_position(self) -> int:
        """Position for display."""
        return self.sequence_index + 1


def parse_point_mutation(
    text: str,
    *,
    sequence: str | None = None,
    index_base: int = 1,
    region: str | None = None,
) -> PointMutation:
    """Parse a compact mutation label such as 'R135A'."""
    if index_base not in {0, 1}:
        raise ValueError("index_base must be 0 or 1.")

    match = _MUTATION_RE.match(text)
    if match is None:
        raise ValueError(
            f"Invalid mutation label {text!r}; expected format like 'R135A'."
        )

    wt_aa, raw_position, mutant_aa = match.groups()
    sequence_index = int(raw_position) - index_base
    mutation = PointMutation(
        sequence_index=sequence_index,
        wt_aa=wt_aa.upper(),
        mutant_aa=mutant_aa.upper(),
        region=region,
    )

    if sequence is not None:
        _validate_mutation_against_sequence(sequence, mutation)

    return mutation


def mutation_label(mutation: PointMutation, *, index_base: int = 1) -> str:
    """Return a compact mutation label such as 'R135A'."""
    if index_base not in {0, 1}:
        raise ValueError("index_base must be 0 or 1.")
    return f"{mutation.wt_aa}{mutation.sequence_index + index_base}{mutation.mutant_aa}"


def apply_point_mutation(
    sequence: str,
    mutation: PointMutation,
    *,
    validate_wt: bool = True,
) -> str:
    """Return 'sequence' with one point mutation applied."""
    seq = _normalize_sequence(sequence)
    _validate_mutation_bounds(seq, mutation)

    if validate_wt:
        _validate_mutation_against_sequence(seq, mutation)

    residues = list(seq)
    residues[mutation.sequence_index] = mutation.mutant_aa
    return "".join(residues)


def generate_single_mutants(
    sequence: str,
    *,
    positions: Sequence[int] | None = None,
    region_by_position: Mapping[int, str] | None = None,
    amino_acids: Sequence[str] = CANONICAL_AMINO_ACIDS,
    skip_wt: bool = True,
) -> list[PointMutation]:
    """Generate all possible single residue substitutions over selected positions, excluding WT."""
    seq = _normalize_sequence(sequence)
    scan_positions = list(range(len(seq))) if positions is None else list(positions)
    residues = tuple(_normalize_amino_acids(amino_acids))

    mutations: list[PointMutation] = []
    for position in scan_positions:
        if position < 0 or position >= len(seq):
            raise ValueError(
                f"Position {position} is outside sequence length {len(seq)}."
            )

        wt_aa = seq[position]
        region = None if region_by_position is None else region_by_position.get(position)
        for mutant_aa in residues:
            if skip_wt and mutant_aa == wt_aa:
                continue
            mutations.append(
                PointMutation(
                    sequence_index=position,
                    wt_aa=wt_aa,
                    mutant_aa=mutant_aa,
                    region=region,
                )
            )

    return mutations


def positions_from_boolean_mask(mask: Sequence[bool] | np.ndarray) -> list[int]:
    """Return positions where a boolean mask is true."""
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D boolean mask, got shape {arr.shape}.")
    return [int(idx) for idx in np.flatnonzero(arr)]


def region_labels_by_position(
    region_masks: Mapping[str, Sequence[bool] | np.ndarray],
    *,
    selected_regions: Sequence[str] | None = None,
    joiner: str = ";",
) -> dict[int, str]:
    """Build a region-label map from boolean masks."""
    names = tuple(selected_regions) if selected_regions is not None else tuple(region_masks)
    labels: dict[int, list[str]] = {}

    for region_name in names:
        if region_name not in region_masks:
            available = ", ".join(region_masks)
            raise ValueError(
                f"Region mask {region_name!r} not found. Available: {available}"
            )

        for position in positions_from_boolean_mask(region_masks[region_name]):
            labels.setdefault(position, []).append(region_name)

    return {
        position: joiner.join(region_names)
        for position, region_names in sorted(labels.items())
    }


def selected_positions_from_region_masks(
    region_masks: Mapping[str, Sequence[bool] | np.ndarray],
    *,
    selected_regions: Sequence[str],
) -> list[int]:
    """Return sorted positions covered by selected region masks."""
    region_labels = region_labels_by_position(
        region_masks,
        selected_regions=selected_regions,
    )
    return sorted(region_labels)


def _normalize_sequence(sequence: str) -> str:
    seq = "".join(str(sequence).split()).upper()
    if not seq:
        raise ValueError("Sequence must not be empty.")
    invalid = sorted(set(seq).difference(CANONICAL_AMINO_ACIDS))
    if invalid:
        joined = ", ".join(invalid)
        raise ValueError(f"Sequence contains non-canonical residues: {joined}")
    return seq


def _normalize_amino_acids(amino_acids: Sequence[str]) -> list[str]:
    residues = [str(residue).upper() for residue in amino_acids]
    if not residues:
        raise ValueError("At least one candidate amino acid is required.")

    seen: set[str] = set()
    normalized: list[str] = []
    for residue in residues:
        _validate_single_residue(residue, field_name="amino_acids")
        if residue not in seen:
            normalized.append(residue)
            seen.add(residue)
    return normalized


def _validate_single_residue(value: str, *, field_name: str) -> None:
    residue = str(value).upper()
    if len(residue) != 1 or residue not in CANONICAL_AMINO_ACIDS:
        valid = "".join(CANONICAL_AMINO_ACIDS)
        raise ValueError(f"{field_name} must be one canonical amino acid: {valid}")


def _validate_mutation_bounds(sequence: str, mutation: PointMutation) -> None:
    if mutation.sequence_index >= len(sequence):
        raise ValueError(
            f"Mutation position {mutation.sequence_index} is outside sequence "
            f"length {len(sequence)}."
        )


def _validate_mutation_against_sequence(sequence: str, mutation: PointMutation) -> None:
    seq = _normalize_sequence(sequence)
    _validate_mutation_bounds(seq, mutation)
    observed = seq[mutation.sequence_index]
    if observed != mutation.wt_aa:
        raise ValueError(
            f"WT residue mismatch at one-based position {mutation.one_based_position}: "
            f"mutation expects {mutation.wt_aa}, sequence has {observed}."
        )

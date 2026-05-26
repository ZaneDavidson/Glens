import numpy as np
import pytest

from glens.design.mutations import (
    PointMutation,
    apply_point_mutation,
    generate_single_mutants,
    mutation_label,
    parse_point_mutation,
    positions_from_boolean_mask,
    region_labels_by_position,
    selected_positions_from_region_masks,
)


def test_parse_and_apply_point_mutation_uses_one_based_labels_by_default() -> None:
    sequence = "ACDEFG"

    mutation = parse_point_mutation("C2W", sequence=sequence)
    mutant = apply_point_mutation(sequence, mutation)

    assert mutation.sequence_index == 1
    assert mutation.one_based_position == 2
    assert mutation.wt_aa == "C"
    assert mutation.mutant_aa == "W"
    assert mutation_label(mutation) == "C2W"
    assert mutant == "AWDEFG"


def test_parse_point_mutation_can_use_zero_based_labels() -> None:
    sequence = "ACDEFG"

    mutation = parse_point_mutation("C1W", sequence=sequence, index_base=0)

    assert mutation.sequence_index == 1
    assert mutation_label(mutation, index_base=0) == "C1W"
    assert mutation_label(mutation, index_base=1) == "C2W"


def test_parse_rejects_mismatched_wt_residue() -> None:
    with pytest.raises(ValueError, match="WT residue mismatch"):
        parse_point_mutation("A2W", sequence="ACDEFG")


def test_point_mutation_rejects_identity_substitution() -> None:
    with pytest.raises(ValueError, match="must change"):
        PointMutation(sequence_index=0, wt_aa="A", mutant_aa="A")


def test_generate_single_mutants_scans_selected_positions_with_region_labels() -> None:
    sequence = "ACD"
    mutations = generate_single_mutants(
        sequence,
        positions=[0, 2],
        amino_acids=("A", "C", "D"),
        region_by_position={0: "tm3", 2: "icl2"},
    )

    labels = [mutation_label(mutation) for mutation in mutations]

    assert labels == ["A1C", "A1D", "D3A", "D3C"]
    assert [mutation.region for mutation in mutations] == ["tm3", "tm3", "icl2", "icl2"]


def test_generate_single_mutants_checks_position_bounds() -> None:
    with pytest.raises(ValueError, match="outside sequence length"):
        generate_single_mutants("ACD", positions=[3])


def test_positions_from_boolean_mask() -> None:
    assert positions_from_boolean_mask([False, True, False, True]) == [1, 3]


def test_region_labels_by_position_preserves_overlaps() -> None:
    masks = {
        "tm6": np.array([False, True, True, False]),
        "icl3": np.array([False, False, True, True]),
    }

    labels = region_labels_by_position(masks)

    assert labels == {1: "tm6", 2: "tm6;icl3", 3: "icl3"}


def test_selected_positions_from_region_masks_limits_to_requested_regions() -> None:
    masks = {
        "tm6": [False, True, True, False],
        "icl3": [False, False, True, True],
        "h8": [True, False, False, False],
    }

    positions = selected_positions_from_region_masks(
        masks,
        selected_regions=("tm6", "h8"),
    )

    assert positions == [0, 1, 2]

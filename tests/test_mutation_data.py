import pandas as pd
import pytest

from glens.mutation_data import (
    apply_mutation_string,
    build_mutation_manifest_frame,
    canonical_mutation_string,
    parse_mutation_string,
)


def test_parse_and_canonicalize_multi_mutation() -> None:
    specs = parse_mutation_string("R131Q+D130A")
    assert canonical_mutation_string(specs) == "D130A+R131Q"


def test_apply_mutation_checks_wt_residue() -> None:
    assert apply_mutation_string("MDR", "D2A") == "MAR"
    with pytest.raises(ValueError, match="WT residue mismatch"):
        apply_mutation_string("MDR", "R2A")


def test_build_mutation_manifest_frame_maps_generic_numbers() -> None:
    receptors = pd.DataFrame({
        "receptor_id": ["demo_human"],
        "sequence": ["MDRY"],
    })
    numbering = pd.DataFrame({
        "receptor_id": ["demo_human", "demo_human"],
        "seq_pos": [2, 3],
        "aa": ["D", "R"],
        "region": ["TM3", "TM3"],
        "gpcrdb_number": ["3x49", "3x50"],
        "is_intracellular_face": [False, False],
        "is_candidate_interface": [True, True],
    })
    mutations = pd.DataFrame({
        "receptor_id": ["demo_human"],
        "mutation_string": ["D2A+R3Q"],
    })

    manifest = build_mutation_manifest_frame(mutations, receptors, numbering)

    assert len(manifest) == 2
    assert manifest["variant_id"].iloc[0] == "demo_human__D2A_R3Q"
    assert set(manifest["gpcrdb_number"]) == {"3x49", "3x50"}
    assert set(manifest["mutant_sequence"]) == {"MAQY"}
    assert manifest["is_candidate_interface"].all()

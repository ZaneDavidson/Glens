import pandas as pd

from glens.embedding_data import (
    EMBEDDING_COLUMNS,
    build_mutation_embedding_delta_frame,
    build_region_embedding_frame,
    build_residue_embedding_frame,
    residue_embedding,
)


def test_residue_embedding_backend_is_deterministic() -> None:
    assert residue_embedding("D") == residue_embedding("D")
    assert residue_embedding("D") != residue_embedding("A")
    assert len(residue_embedding("D")) == len(EMBEDDING_COLUMNS)


def test_region_pool_table_keeps_requested_pools() -> None:
    receptors = pd.DataFrame({"receptor_id": ["demo_human"], "sequence": ["MDRY"]})
    numbering = pd.DataFrame({
        "receptor_id": ["demo_human", "demo_human"],
        "seq_pos": [2, 3],
        "aa": ["D", "R"],
        "region": ["TM3", "ICL2"],
        "gpcrdb_number": ["3x49", ""],
        "is_intracellular_face": [False, True],
        "is_candidate_interface": [True, True],
    })

    residue_frame = build_residue_embedding_frame(receptors, numbering)
    region_frame = build_region_embedding_frame(residue_frame, ["ICL2", "H8"])

    assert set(region_frame["region_pool"]) == {"ICL2", "H8"}
    assert int(region_frame.loc[region_frame["region_pool"] == "ICL2", "residue_count"].iloc[0]) == 1
    assert int(region_frame.loc[region_frame["region_pool"] == "H8", "residue_count"].iloc[0]) == 0


def test_mutation_embedding_delta_frame_uses_local_window() -> None:
    receptors = pd.DataFrame({"receptor_id": ["demo_human"], "sequence": ["MDRY"]})
    mutation_manifest = pd.DataFrame({
        "variant_id": ["demo_human__D2A"],
        "receptor_id": ["demo_human"],
        "canonical_mutation": ["D2A"],
        "substitution_index": [1],
        "seq_pos": [2],
        "gpcrdb_number": ["3x49"],
        "region": ["TM3"],
        "wt_aa": ["D"],
        "mut_aa": ["A"],
        "mutant_sequence": ["MARY"],
    })

    deltas = build_mutation_embedding_delta_frame(mutation_manifest, receptors, window_radius=1)

    assert len(deltas) == 1
    assert int(deltas["window_start"].iloc[0]) == 1
    assert int(deltas["window_end"].iloc[0]) == 3
    assert any(abs(float(deltas[col].iloc[0])) > 0 for col in deltas.columns if col.startswith("delta_embedding_"))

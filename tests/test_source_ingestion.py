from pathlib import Path

import pandas as pd

try:
    from glens.source_ingestion import (
        amino_acid_properties_frame,
        gpcrdb_name_to_uniprot_id,
        labels_for_core,
        parse_common_coupling_map,
    )
except ModuleNotFoundError:
    from src.glens.source_ingestion import (
        amino_acid_properties_frame,
        gpcrdb_name_to_uniprot_id,
        labels_for_core,
        parse_common_coupling_map,
    )


def write_minimal_common_map(path: Path) -> None:
    columns = [f"col_{idx}" for idx in range(67)]
    header = [""] * 67
    header[1] = "Uniprot"
    header[3] = "GPCRdb"
    data = [""] * 67
    data[1] = "5HT1A"
    data[2] = "5-HT1A"
    data[3] = "https://gproteindb.org/protein/5ht1a_human"
    data[4] = "5-Hydroxytryptamine"
    data[5] = "A"
    data[7] = "Bouvier"
    data[8] = "GEMTA"
    data[11] = "Avet C et al. eLife 2022"
    data[12] = "5-hydroxytryptamine"
    data[13] = "Phys"
    data[14] = "-"
    data[15] = "3"
    data[16] = "2"
    data[17] = "-"
    data[18] = "Gi/o"
    data[20] = "nc"
    data[21] = "1'"
    data[22] = "2'"
    data[23] = "nc"
    data[24] = "0"
    data[25] = "100"
    data[26] = "81"
    data[27] = "0"
    data[28] = "log(Emax/EC50)"
    data[29] = "0"
    data[30] = "8.3"
    data[31] = "6.7"
    data[32] = "0"
    data[33] = "GoB"
    data[34] = "0"
    data[37] = "92"
    data[43] = "100"
    data[50] = "log(Emax/EC50)"
    data[54] = "7.6"
    data[60] = "8.3"
    pd.DataFrame([header, data], columns=columns).to_csv(path, index=False)


def test_common_coupling_map_to_long_labels(tmp_path: Path) -> None:
    common_map = tmp_path / "common_map.csv"
    write_minimal_common_map(common_map)

    frames = parse_common_coupling_map(common_map)
    core_labels = labels_for_core(frames.labels)

    assert set(frames.receptor_seeds["receptor_id"]) == {"5ht1a_human"}
    assert len(frames.assay_observations) == 1
    family_labels = frames.labels[frames.labels["evidence_kind"] == "family"]
    assert set(family_labels["g_alpha_family"]) == {"Gs", "Gi/o", "Gq/11", "G12/13"}
    assert set(family_labels["label_state"]) == {"positive", "negative_nc"}
    assert "log_Emax_EC50" in set(core_labels["readout_type"])


def test_static_ingestion_tables_are_model_ready() -> None:
    aa = amino_acid_properties_frame()
    assert set(aa["aa"]) == set("ACDEFGHIKLMNPQRSTVWY")
    assert {"hydropathy_kd", "charge_ph7", "side_chain_volume_a3"}.issubset(set(aa.columns))


def test_gpcrdb_name_to_uniprot_id() -> None:
    assert gpcrdb_name_to_uniprot_id("5ht1a_human") == "5HT1A_HUMAN"
    assert gpcrdb_name_to_uniprot_id("adrb2_human") == "ADRB2_HUMAN"

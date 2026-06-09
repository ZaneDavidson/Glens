from pathlib import Path

import pandas as pd

from glens.build_core_datasets import build_core_datasets, infer_label_state


def test_build_pass1_sample(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = root / "configs" / "pass1.yaml"
    summary = build_core_datasets(config)

    receptor_manifest = pd.read_parquet(summary["receptor_manifest"])
    label_fact_table = pd.read_parquet(summary["label_fact_table"])
    generic_number_maps = pd.read_parquet(summary["generic_number_maps"])

    assert set(receptor_manifest["receptor_id"]) == {"adrb2_human", "m3_human"}
    assert {"positive", "negative_nc", "missing", "weak"}.issubset(set(label_fact_table["label_state"]))
    assert "Gi/o" in set(label_fact_table["g_alpha_family"])
    assert generic_number_maps["is_candidate_interface"].any()


def test_nc_is_not_missing() -> None:
    nc = infer_label_state(pd.Series({"label_state": "nc", "value_raw": "nc"}))
    missing = infer_label_state(pd.Series({"label_state": "missing", "value_raw": ""}))
    assert nc == "negative_nc"
    assert missing == "missing"

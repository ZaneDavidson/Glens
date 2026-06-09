from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

AA20 = set("ACDEFGHIKLMNPQRSTVWY")
LABEL_STATES = {"positive", "negative_nc", "missing", "weak", "unknown"}

SUBTYPE_TO_FAMILY = {
    "GNAS": "Gs", "GNAL": "Gs", "GSA": "Gs", "GOLF": "Gs",
    "GNAI1": "Gi/o", "GNAI2": "Gi/o", "GNAI3": "Gi/o", "GNAO1": "Gi/o",
    "GNAZ": "Gi/o", "GNAT1": "Gi/o", "GNAT2": "Gi/o", "GNAI": "Gi/o", "GNAO": "Gi/o",
    "GNAQ": "Gq/11", "GNA11": "Gq/11", "GNA14": "Gq/11", "GNA15": "Gq/11", "GNA16": "Gq/11",
    "GNA12": "G12/13", "GNA13": "G12/13",
}
FAMILY_ALIASES = {
    "GS": "Gs", "G_S": "Gs", "GOLF": "Gs",
    "GI": "Gi/o", "GIO": "Gi/o", "GI/O": "Gi/o", "GI-O": "Gi/o", "GI_O": "Gi/o", "GO": "Gi/o", "GZ": "Gi/o",
    "GQ": "Gq/11", "GQ/11": "Gq/11", "GQ11": "Gq/11", "G15": "Gq/11", "G16": "Gq/11",
    "G12": "G12/13", "G13": "G12/13", "G12/13": "G12/13", "G12_13": "G12/13",
}


@dataclass(frozen=True)
class dataConfig:
    receptors_csv: Path
    labels_csv: Path
    generic_numbers_csv: Path | None
    receptor_manifest: Path | None
    label_fact_table: Path | None
    generic_number_maps: Path | None
    species: str | None = "human"
    gpcr_class: str | None = "A"


def build_core_datasets(config_path: str | Path) -> dict[str, str]:
    cfg = load_config(config_path)
    receptors = build_receptor_manifest(cfg.receptors_csv, cfg.species, cfg.gpcr_class)
    labels = build_label_fact_table(cfg.labels_csv, receptors)
    numbering = build_generic_number_maps(cfg.generic_numbers_csv, receptors)

    write_parquet(receptors, cfg.receptor_manifest)
    write_parquet(labels, cfg.label_fact_table)
    write_parquet(numbering, cfg.generic_number_maps)

    return {
        "receptor_manifest": str(cfg.receptor_manifest),
        "label_fact_table": str(cfg.label_fact_table),
        "generic_number_maps": str(cfg.generic_number_maps),
    }


def load_config(path: str | Path) -> dataConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    base = path.parent.parent if path.parent.name == "configs" else Path.cwd()
    inputs = raw.get("inputs", {})
    outputs = raw.get("outputs", {})
    filters = raw.get("filters", {})

    def p(value: str | None) -> Path | None:
        if value in (None, ""):
            return None
        candidate = Path(value)
        return candidate if candidate.is_absolute() else base / candidate

    receptors_csv = p(inputs.get("receptors_csv"))
    labels_csv = p(inputs.get("labels_csv"))
    if receptors_csv is None or labels_csv is None:
        raise ValueError("Config must define inputs.receptors_csv and inputs.labels_csv")

    return dataConfig(
        receptors_csv=receptors_csv,
        labels_csv=labels_csv,
        generic_numbers_csv=p(inputs.get("generic_numbers_csv")),
        receptor_manifest=p(outputs.get("receptor_manifest", "data/interim/receptor_manifest.parquet")),
        label_fact_table=p(outputs.get("label_fact_table", "data/interim/label_fact_table.parquet")),
        generic_number_maps=p(outputs.get("generic_number_maps", "data/interim/generic_number_maps.parquet")),
        species=filters.get("species"),
        gpcr_class=filters.get("gpcr_class"),
    )


def build_receptor_manifest(path: str | Path, species: str | None = "human", gpcr_class: str | None = "A") -> pd.DataFrame:
    df = read_csv(path)
    df = normalize_columns(df)
    df = fix_columns_to_target(df, "receptor_id", ["entry_name", "gpcrdb_entry_name", "name", "target_id"])
    df = fix_columns_to_target(df, "uniprot_id", ["accession", "uniprot", "accession_id"])
    df = fix_columns_to_target(df, "gpcrdb_entry_name", ["entry_name", "protein", "slug"])
    df = fix_columns_to_target(df, "iuphar_name", ["iuphar", "receptor_name", "name"])
    df = fix_columns_to_target(df, "receptor_family", ["family", "protein_family", "subfamily"])
    df = fix_columns_to_target(df, "gpcr_class", ["class", "receptor_class"])

    require_columns(df, ["receptor_id", "sequence"])
    df["receptor_id"] = df["receptor_id"].astype(str).str.strip().str.lower()
    df["sequence"] = df["sequence"].astype(str).str.replace(r"\s+", "", regex=True).str.upper()
    df["sequence_length"] = df["sequence"].str.len()
    df["has_gpcrdb_numbering"] = False

    for col in ["uniprot_id", "gpcrdb_entry_name", "iuphar_name", "species", "gpcr_class", "receptor_family"]:
        if col not in df.columns:
            df[col] = pd.NA

    if species and "species" in df.columns:
        df = df[df["species"].astype(str).str.lower().eq(species.lower()) | df["species"].isna()].copy()
    if gpcr_class and "gpcr_class" in df.columns:
        df = df[df["gpcr_class"].astype(str).str.upper().eq(gpcr_class.upper()) | df["gpcr_class"].isna()].copy()

    bad = df.loc[~df["sequence"].map(is_protein_sequence), ["receptor_id", "sequence"]]
    if not bad.empty:
        ids = ", ".join(bad["receptor_id"].head(5).astype(str))
        raise ValueError(f"Invalid amino-acid sequence for receptor(s): {ids}")

    out_cols = [
        "receptor_id", "uniprot_id", "gpcrdb_entry_name", "iuphar_name", "species",
        "gpcr_class", "receptor_family", "sequence", "sequence_length", "has_gpcrdb_numbering",
    ]
    return df[out_cols].drop_duplicates("receptor_id").sort_values("receptor_id").reset_index(drop=True)


def build_generic_number_maps(path: str | Path | None, receptors: pd.DataFrame) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return fallback_numbering(receptors)

    df = read_csv(path)
    df = normalize_columns(df)
    df = fix_columns_to_target(df, "receptor_id", ["entry_name", "gpcrdb_entry_name", "protein"])
    df = fix_columns_to_target(df, "seq_pos", ["sequence_number", "position", "residue_number", "pos"])
    df = fix_columns_to_target(df, "aa", ["amino_acid", "residue", "wt_aa"])
    df = fix_columns_to_target(df, "gpcrdb_number", ["generic_number", "display_generic_number", "gpcrdb"])
    require_columns(df, ["receptor_id", "seq_pos", "aa"])

    if "region" not in df.columns:
        df["region"] = "unknown"
    if "gpcrdb_number" not in df.columns:
        df["gpcrdb_number"] = pd.NA

    df["receptor_id"] = df["receptor_id"].astype(str).str.strip().str.lower()
    df = df[df["receptor_id"].isin(set(receptors["receptor_id"]))].copy()
    df["seq_pos"] = pd.to_numeric(df["seq_pos"], errors="raise").astype(int)
    df["aa"] = df["aa"].astype(str).str.strip().str.upper().str[0]
    df["region"] = df["region"].astype(str).map(normalize_region)
    df["is_intracellular_face"] = df["region"].map(is_intracellular_region)
    df["is_candidate_interface"] = df.apply(is_candidate_interface_row, axis=1)

    out_cols = [
        "receptor_id", "seq_pos", "aa", "region", "gpcrdb_number",
        "is_intracellular_face", "is_candidate_interface",
    ]
    out = df[out_cols].drop_duplicates(["receptor_id", "seq_pos"]).sort_values(["receptor_id", "seq_pos"]).reset_index(drop=True)
    return out


def fallback_numbering(receptors: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in receptors.itertuples(index=False):
        for idx, aa in enumerate(str(rec.sequence), start=1):
            rows.append({
                "receptor_id": rec.receptor_id,
                "seq_pos": idx,
                "aa": aa,
                "region": "unknown",
                "gpcrdb_number": pd.NA,
                "is_intracellular_face": False,
                "is_candidate_interface": False,
            })
    return pd.DataFrame(rows)


def build_label_fact_table(path: str | Path, receptors: pd.DataFrame) -> pd.DataFrame:
    df = read_csv(path)
    df = normalize_columns(df)
    df = fix_columns_to_target(df, "receptor_id", ["entry_name", "gpcrdb_entry_name", "target_id", "protein"])
    df = fix_columns_to_target(df, "variant_id", ["variant", "mutation_id"])
    df = fix_columns_to_target(df, "dataset", ["source", "resource"])
    df = fix_columns_to_target(df, "g_alpha_subtype", ["gprotein", "g_protein", "g_alpha", "subtype"])
    df = fix_columns_to_target(df, "g_alpha_family", ["family", "g_family", "transducer_family"])
    df = fix_columns_to_target(df, "readout_type", ["parameter", "metric", "measure"])
    df = fix_columns_to_target(df, "value_raw", ["value", "score", "activity", "coupling"])
    require_columns(df, ["receptor_id"])

    df["receptor_id"] = df["receptor_id"].astype(str).str.strip().str.lower()
    df = df[df["receptor_id"].isin(set(receptors["receptor_id"]))].copy()

    defaults = {
        "variant_id": "WT",
        "dataset": "unknown",
        "assay_type": "unknown",
        "ligand_context": "unknown",
        "cell_context": "unknown",
        "g_alpha_subtype": pd.NA,
        "g_alpha_family": pd.NA,
        "readout_type": "unknown",
        "value_raw": pd.NA,
        "label_state": pd.NA,
        "evidence_weight": 1.0,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    df["variant_id"] = df["variant_id"].fillna("WT").astype(str)
    df["g_alpha_subtype"] = df["g_alpha_subtype"].astype("string")
    df["g_alpha_family"] = df.apply(infer_family, axis=1)
    df["label_state"] = df.apply(infer_label_state, axis=1)
    df["value_normalized"] = df.apply(normalize_value, axis=1)
    df["evidence_weight"] = pd.to_numeric(df["evidence_weight"], errors="coerce").fillna(1.0)

    out_cols = [
        "receptor_id", "variant_id", "dataset", "assay_type", "ligand_context", "cell_context",
        "g_alpha_subtype", "g_alpha_family", "readout_type", "value_raw", "value_normalized",
        "label_state", "evidence_weight",
    ]
    return df[out_cols].sort_values(["receptor_id", "variant_id", "g_alpha_family"]).reset_index(drop=True)


def infer_family(row: pd.Series) -> str:
    family = row.get("g_alpha_family")
    if pd.notna(family) and str(family).strip():
        key = str(family).strip().upper().replace(" ", "").replace("-", "/")
        return FAMILY_ALIASES.get(key, str(family).strip())
    subtype = row.get("g_alpha_subtype")
    if pd.isna(subtype):
        return "unknown"
    key = str(subtype).strip().upper().replace("-", "").replace(" ", "")
    return SUBTYPE_TO_FAMILY.get(key, "unknown")


def infer_label_state(row: pd.Series) -> str:
    raw_state = row.get("label_state")
    raw_value = row.get("value_raw")
    state = "" if pd.isna(raw_state) else str(raw_state).strip().lower()
    value = "" if pd.isna(raw_value) else str(raw_value).strip().lower()
    token = state or value

    if token in {"positive", "pos", "yes", "true", "coupled", "coupling", "1"}:
        return "positive"
    if token in {"negative_nc", "nc", "non-coupling", "noncoupling", "non_coupling", "tested_negative", "0"}:
        return "negative_nc"
    if token in {"missing", "na", "n/a", "nan", "", "none", "not_tested", "untested", "-"}:
        return "missing"
    if token in {"weak", "low", "low_confidence", "ambiguous"}:
        return "weak"
    # to_numeric expects a Series/array-like for typing; wrap the raw value in a Series
    number = pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0]
    if pd.notna(number):
        return "positive" if float(number) > 0 else "negative_nc"
    return "unknown"


def normalize_value(row: pd.Series) -> float | None:
    state = row["label_state"]
    # to_numeric expects a Series/array-like for typing; wrap the raw value in a Series
    value = pd.to_numeric(pd.Series([row.get("value_raw")]), errors="coerce").iloc[0]
    if pd.notna(value):
        return float(value)
    if state == "positive":
        return 1.0
    if state == "negative_nc":
        return 0.0
    return None


def normalize_region(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip().upper().replace(" ", "")
    aliases = {
        "TM1": "TM1", "TM2": "TM2", "TM3": "TM3", "TM4": "TM4", "TM5": "TM5", "TM6": "TM6", "TM7": "TM7",
        "ICL1": "ICL1", "IL1": "ICL1", "ICL2": "ICL2", "IL2": "ICL2", "ICL3": "ICL3", "IL3": "ICL3",
        "H8": "H8", "HELIX8": "H8", "ECL1": "ECL1", "ECL2": "ECL2", "ECL3": "ECL3",
        "N-TERM": "N-term", "NTERM": "N-term", "C-TERM": "C-term", "CTERM": "C-term",
    }
    return aliases.get(text, str(value).strip() if str(value).strip() else "unknown")


def is_intracellular_region(region: str) -> bool:
    return region in {"ICL1", "ICL2", "ICL3", "H8", "C-term"}


def is_candidate_interface_row(row: pd.Series) -> bool:
    region = row.get("region")
    number = row.get("gpcrdb_number")
    if region in {"ICL1", "ICL2", "ICL3", "H8"}:
        return True
    if pd.isna(number):
        return False
    text = str(number)
    return text.startswith(("3x", "5x", "6x", "7x")) and region in {"TM3", "TM5", "TM6", "TM7"}


def is_protein_sequence(seq: str) -> bool:
    return bool(seq) and set(seq).issubset(AA20)


def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    return df


def fix_columns_to_target(df: pd.DataFrame, target: str, aliases: list[str]) -> pd.DataFrame:
    df = df.copy()
    if target not in df.columns:
        for alias in aliases:
            if alias in df.columns:
                df[target] = df[alias]
                break
    return df


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")


def write_parquet(df: pd.DataFrame, path: str | Path | None) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

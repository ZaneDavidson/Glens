from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import yaml

from .build_core_datasets import AA20, normalize_columns

CANONICAL_FAMILIES = ("Gs", "Gi/o", "Gq/11", "G12/13")

# Column positions in the GPCR common coupling map.  The source file has a
# two-level-ish header, so this parser deliberately uses stable positional
# columns after reading as strings.
COMMON_MAP_FAMILY_COLUMNS: dict[str, dict[str, int]] = {
    "Gs": {"support": 14, "rank": 20, "percent": 24, "quant": 29},
    "Gi/o": {"support": 15, "rank": 21, "percent": 25, "quant": 30},
    "Gq/11": {"support": 16, "rank": 22, "percent": 26, "quant": 31},
    "G12/13": {"support": 17, "rank": 23, "percent": 27, "quant": 32},
}

COMMON_MAP_SUBTYPES = (
    "GsL", "GsS", "Golf", "Gi1", "Gi2", "Gi3", "Ggust", "Gz",
    "GoA", "GoB", "Gq", "G11", "G14", "G15", "G12", "G13",
)

SUBTYPE_TO_FAMILY = {
    "GsL": "Gs", "GsS": "Gs", "Golf": "Gs",
    "Gi1": "Gi/o", "Gi2": "Gi/o", "Gi3": "Gi/o", "Ggust": "Gi/o",
    "Gz": "Gi/o", "GoA": "Gi/o", "GoB": "Gi/o",
    "Gq": "Gq/11", "G11": "Gq/11", "G14": "Gq/11", "G15": "Gq/11",
    "G12": "G12/13", "G13": "G12/13",
}

UNIPROT_API = "https://rest.uniprot.org/uniprotkb"
UNIPROT_FIELDS = "accession,id,protein_name,gene_names,organism_name,sequence"


@dataclass(frozen=True)
class UniProtConfig:
    enabled: bool = False
    cache_dir: Path | None = None
    reviewed_only: bool = True
    organism_id: int = 9606
    timeout_seconds: int = 20


@dataclass(frozen=True)
class SourceIngestionConfig:
    common_coupling_map_csv: Path | None
    receptor_seed_csv: Path | None

    receptors_csv: Path
    labels_csv: Path
    receptor_seeds_csv: Path
    unresolved_receptors_csv: Path
    common_coupling_assays_csv: Path
    common_coupling_labels_csv: Path

    g_alpha_family_map_csv: Path
    amino_acid_properties_csv: Path

    require_sequences_for_core: bool
    uniprot: UniProtConfig


@dataclass(frozen=True)
class CommonCouplingFrames:
    receptor_seeds: pd.DataFrame
    assay_observations: pd.DataFrame
    labels: pd.DataFrame


# -----------------------------------------------------------------------------
# Public entry points
# -----------------------------------------------------------------------------


def build_source_artifacts(config_path: str | Path) -> dict[str, str]:
    """Build stable local CSV inputs from raw source files.

    This command writes only two kinds of outputs:
    1. source-derived artifacts from the common coupling map / receptor seeds;
    2. small reference tables that are directly used by normalization or later
       mutation-feature construction.
    """
    cfg = load_source_ingestion_config(config_path)

    common = parse_configured_common_map(cfg.common_coupling_map_csv)
    extra_seeds = read_optional_seed_table(cfg.receptor_seed_csv)
    seeds = combine_receptor_seeds([common.receptor_seeds, extra_seeds])

    if seeds.empty:
        raise ValueError(
            "No receptor source data were ingested. Provide inputs.common_coupling_map_csv "
            "or inputs.receptor_seed_csv in the source-ingestion config."
        )

    if cfg.uniprot.enabled:
        seeds = resolve_receptor_seeds_with_uniprot(seeds, cfg.uniprot)

    core_receptors, unresolved = split_core_and_unresolved_receptors(
        seeds,
        require_sequences=cfg.require_sequences_for_core,
    )
    core_labels = labels_for_core(common.labels)

    write_csv(core_receptors, cfg.receptors_csv)
    write_csv(core_labels, cfg.labels_csv)
    write_csv(seeds, cfg.receptor_seeds_csv)
    write_csv(unresolved, cfg.unresolved_receptors_csv)
    write_csv(common.assay_observations, cfg.common_coupling_assays_csv)
    write_csv(common.labels, cfg.common_coupling_labels_csv)
    write_csv(g_alpha_family_map_frame(), cfg.g_alpha_family_map_csv)
    write_csv(amino_acid_properties_frame(), cfg.amino_acid_properties_csv)

    if cfg.require_sequences_for_core and core_receptors.empty:
        hint = (
            "UniProt resolution is disabled; set uniprot.enabled: true."
            if not cfg.uniprot.enabled
            else "UniProt resolution is enabled; check network access, cache contents, or lookup strategy."
        )
        raise ValueError(
            f"No sequence-resolved receptors were produced. Ingested {len(seeds)} receptor seeds, "
            f"but {len(unresolved)} remain unresolved. {hint}"
        )

    return {
        "receptors_csv": str(cfg.receptors_csv),
        "labels_csv": str(cfg.labels_csv),
        "receptor_seeds_csv": str(cfg.receptor_seeds_csv),
        "unresolved_receptors_csv": str(cfg.unresolved_receptors_csv),
        "common_coupling_assays_csv": str(cfg.common_coupling_assays_csv),
        "common_coupling_labels_csv": str(cfg.common_coupling_labels_csv),
        "g_alpha_family_map_csv": str(cfg.g_alpha_family_map_csv),
        "amino_acid_properties_csv": str(cfg.amino_acid_properties_csv),
    }


def load_source_ingestion_config(path: str | Path) -> SourceIngestionConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    base = path.parent.parent if path.parent.name == "configs" else Path.cwd()
    inputs = raw.get("inputs", {}) or {}
    outputs = raw.get("outputs", {}) or {}
    uniprot_raw = raw.get("uniprot", {}) or {}

    def p(value: object) -> Path | None:
        if value is None or value == "":
            return None
        candidate = Path(str(value))
        return candidate if candidate.is_absolute() else base / candidate

    def output_path(key: str, default: str) -> Path:
        resolved = p(outputs.get(key, default))
        if resolved is None:
            raise ValueError(f"Config must define outputs.{key}")
        return resolved

    cache_dir = p(uniprot_raw.get("cache_dir", "data/raw/cache/uniprot"))
    if cache_dir is None:
        cache_dir = base / "data/raw/cache/uniprot"

    uniprot = UniProtConfig(
        enabled=bool(uniprot_raw.get("enabled", False)),
        cache_dir=cache_dir,
        reviewed_only=bool(uniprot_raw.get("reviewed_only", True)),
        organism_id=int(uniprot_raw.get("organism_id", 9606)),
        timeout_seconds=int(uniprot_raw.get("timeout_seconds", 20)),
    )

    return SourceIngestionConfig(
        common_coupling_map_csv=p(inputs.get("common_coupling_map_csv")),
        receptor_seed_csv=p(inputs.get("receptor_seed_csv")),
        receptors_csv=output_path("receptors_csv", "data/raw/curated/receptors.csv"),
        labels_csv=output_path("labels_csv", "data/raw/curated/labels.csv"),
        receptor_seeds_csv=output_path("receptor_seeds_csv", "data/raw/curated/receptor_seeds.csv"),
        unresolved_receptors_csv=output_path(
            "unresolved_receptors_csv",
            "data/raw/curated/unresolved_receptors.csv",
        ),
        common_coupling_assays_csv=output_path(
            "common_coupling_assays_csv",
            "data/raw/curated/common_coupling_assays.csv",
        ),
        common_coupling_labels_csv=output_path(
            "common_coupling_labels_csv",
            "data/raw/curated/common_coupling_labels.csv",
        ),
        g_alpha_family_map_csv=output_path(
            "g_alpha_family_map_csv",
            "data/raw/curated/g_alpha_family_map.csv",
        ),
        amino_acid_properties_csv=output_path(
            "amino_acid_properties_csv",
            "data/raw/curated/amino_acid_properties.csv",
        ),
        require_sequences_for_core=bool(raw.get("require_sequences_for_core", True)),
        uniprot=uniprot,
    )


# -----------------------------------------------------------------------------
# Common coupling map parsing
# -----------------------------------------------------------------------------


def parse_configured_common_map(path: Path | None) -> CommonCouplingFrames:
    if path is None:
        return empty_common_coupling_frames()
    if not path.exists():
        raise FileNotFoundError(f"Configured common coupling map does not exist: {path}")
    return parse_common_coupling_map(path)


def parse_common_coupling_map(path: str | Path) -> CommonCouplingFrames:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    if raw.shape[1] < 67:
        raise ValueError(
            f"Common coupling map at {path} has {raw.shape[1]} columns; expected at least 67."
        )

    rows = raw.iloc[1:].copy() if looks_like_common_map_header_row(raw) else raw.copy()
    rows = rows[rows.iloc[:, 1].astype(str).str.strip().ne("")].copy()
    rows = rows[~rows.iloc[:, 1].astype(str).str.strip().str.lower().isin({"uniprot", "receptor"})].copy()

    receptor_rows: list[dict[str, Any]] = []
    assay_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []

    for row_number, row in enumerate(rows.itertuples(index=False, name=None), start=1):
        values = [clean_cell(row[idx]) if idx < len(row) else "" for idx in range(67)]
        receptor_id = receptor_id_from_gpcrdb_url(values[3]) or safe_receptor_id(values[1])
        if not receptor_id:
            continue

        source_lab = values[7] or "unknown"
        assay_id = make_assay_observation_id(
            row_number=row_number,
            receptor_id=receptor_id,
            source_lab=source_lab,
            biosensor=values[8],
            reference=values[11],
            ligand=values[12],
        )

        receptor_rows.append(common_map_receptor_seed(values, receptor_id))
        assay_rows.append(common_map_assay_observation(values, assay_id, receptor_id, source_lab))
        label_rows.extend(common_map_label_rows(values, assay_id, receptor_id, source_lab))

    receptor_seeds = finalize_frame(
        receptor_rows,
        empty_receptor_seed_frame(),
        sort_cols=["receptor_id"],
        dedupe_cols=["receptor_id"],
    )
    assay_observations = finalize_frame(
        assay_rows,
        empty_assay_observation_frame(),
        sort_cols=["assay_observation_id"],
        dedupe_cols=["assay_observation_id"],
    )
    labels = finalize_frame(
        label_rows,
        empty_label_frame(),
        sort_cols=["receptor_id", "assay_observation_id", "evidence_kind", "g_alpha_family", "g_alpha_subtype"],
    )

    return CommonCouplingFrames(
        receptor_seeds=receptor_seeds,
        assay_observations=assay_observations,
        labels=labels,
    )


def looks_like_common_map_header_row(raw: pd.DataFrame) -> bool:
    if raw.empty or raw.shape[1] < 20:
        return False
    return (
        clean_cell(raw.iloc[0, 1]).lower() in {"uniprot", "receptor"}
        and clean_cell(raw.iloc[0, 3]).lower() == "gpcrdb"
    )


def common_map_receptor_seed(values: list[str], receptor_id: str) -> dict[str, Any]:
    return {
        "receptor_id": receptor_id,
        "source_receptor_name": values[1],
        "iuphar_name": values[2],
        "gpcrdb_entry_name": receptor_id,
        "gpcrdb_url": values[3],
        "receptor_family": values[4],
        "gpcr_class": values[5],
        "species": "human" if receptor_id.endswith("_human") else "unknown",
        "other_protein": values[6],
        "uniprot_id": "",
        "sequence": "",
        "sequence_source": "unresolved",
    }


def common_map_assay_observation(
    values: list[str],
    assay_id: str,
    receptor_id: str,
    source_lab: str,
) -> dict[str, Any]:
    return {
        "assay_observation_id": assay_id,
        "receptor_id": receptor_id,
        "dataset": source_lab,
        "biosensor": values[8] or "unknown",
        "downstream_steps": values[9],
        "gproteins_tested": values[10],
        "reference": normalize_reference(values[11]),
        "ligand_name": values[12],
        "ligand_context": values[13] or "unknown",
        "primary_family": values[18],
        "family_count": values[19],
        "primary_subtype": values[33],
    }


def common_map_label_rows(
    values: list[str],
    assay_id: str,
    receptor_id: str,
    source_lab: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    biosensor = values[8] or "unknown"
    ligand_context = values[13] or "unknown"

    for family, cols in COMMON_MAP_FAMILY_COLUMNS.items():
        rows.append(family_label_row(
            assay_id=assay_id,
            receptor_id=receptor_id,
            source_lab=source_lab,
            biosensor=biosensor,
            ligand_context=ligand_context,
            family=family,
            support=values[cols["support"]],
            rank=values[cols["rank"]],
            percent=values[cols["percent"]],
            quant=values[cols["quant"]],
            parameter=values[28],
            primary_family=values[18],
            primary_subtype=values[33],
        ))

    for offset, subtype in enumerate(COMMON_MAP_SUBTYPES):
        rows.append(subtype_label_row(
            assay_id=assay_id,
            receptor_id=receptor_id,
            source_lab=source_lab,
            biosensor=biosensor,
            ligand_context=ligand_context,
            subtype=subtype,
            percent=values[34 + offset],
            quant=values[51 + offset],
            parameter=values[50],
            primary_family=values[18],
            primary_subtype=values[33],
        ))

    return rows


def family_label_row(
    *,
    assay_id: str,
    receptor_id: str,
    source_lab: str,
    biosensor: str,
    ligand_context: str,
    family: str,
    support: str,
    rank: str,
    percent: str,
    quant: str,
    parameter: str,
    primary_family: str,
    primary_subtype: str,
) -> dict[str, Any]:
    label_state = infer_common_map_label_state(rank=rank, support=support, percent=percent, quant=quant)
    normalized = normalized_common_map_value(label_state=label_state, percent=percent, quant=quant)
    return {
        "assay_observation_id": assay_id,
        "receptor_id": receptor_id,
        "variant_id": "WT",
        "dataset": f"CommonCouplingMap:{source_lab}",
        "assay_type": biosensor,
        "ligand_context": ligand_context,
        "cell_context": "unknown",
        "g_alpha_subtype": "",
        "g_alpha_family": family,
        "readout_type": normalize_readout_type(parameter),
        "value_raw": first_nonempty(quant, percent, rank, support),
        "value_normalized": normalized,
        "label_state": label_state,
        "evidence_weight": evidence_weight_for_source(source_lab, support),
        "evidence_kind": "family",
        "rank_order": parse_rank(rank),
        "percent_primary": parse_number(percent),
        "supporting_dataset_count": parse_number(support),
        "primary_family": primary_family,
        "primary_subtype": primary_subtype,
    }


def subtype_label_row(
    *,
    assay_id: str,
    receptor_id: str,
    source_lab: str,
    biosensor: str,
    ligand_context: str,
    subtype: str,
    percent: str,
    quant: str,
    parameter: str,
    primary_family: str,
    primary_subtype: str,
) -> dict[str, Any]:
    family = SUBTYPE_TO_FAMILY[subtype]
    label_state = infer_common_map_label_state(rank="", support="", percent=percent, quant=quant)
    normalized = normalized_common_map_value(label_state=label_state, percent=percent, quant=quant)
    return {
        "assay_observation_id": assay_id,
        "receptor_id": receptor_id,
        "variant_id": "WT",
        "dataset": f"CommonCouplingMap:{source_lab}",
        "assay_type": biosensor,
        "ligand_context": ligand_context,
        "cell_context": "unknown",
        "g_alpha_subtype": subtype,
        "g_alpha_family": family,
        "readout_type": normalize_readout_type(parameter),
        "value_raw": first_nonempty(quant, percent),
        "value_normalized": normalized,
        "label_state": label_state,
        "evidence_weight": evidence_weight_for_source(source_lab, ""),
        "evidence_kind": "subtype",
        "rank_order": "",
        "percent_primary": parse_number(percent),
        "supporting_dataset_count": "",
        "primary_family": primary_family,
        "primary_subtype": primary_subtype,
    }


def labels_for_core(labels: pd.DataFrame) -> pd.DataFrame:
    core_cols = [
        "receptor_id", "variant_id", "dataset", "assay_type", "ligand_context", "cell_context",
        "g_alpha_subtype", "g_alpha_family", "readout_type", "value_raw", "value_normalized",
        "label_state", "evidence_weight",
    ]
    if labels.empty:
        return pd.DataFrame(columns=core_cols)
    return labels.loc[:, core_cols].copy()


# -----------------------------------------------------------------------------
# Receptor seed handling and UniProt resolution
# -----------------------------------------------------------------------------


def read_optional_seed_table(path: Path | None) -> pd.DataFrame:
    if path is None:
        return empty_receptor_seed_frame()
    if not path.exists():
        raise FileNotFoundError(f"Configured receptor seed file does not exist: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = normalize_columns(df)
    for col in empty_receptor_seed_frame().columns:
        if col not in df.columns:
            df[col] = ""
    return df.loc[:, list(empty_receptor_seed_frame().columns)]


def combine_receptor_seeds(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if not frame.empty]
    if not nonempty:
        return empty_receptor_seed_frame()

    combined = pd.concat(nonempty, ignore_index=True)
    combined = normalize_columns(combined)
    for col in empty_receptor_seed_frame().columns:
        if col not in combined.columns:
            combined[col] = ""

    combined = combined.loc[:, list(empty_receptor_seed_frame().columns)].copy()
    combined["receptor_id"] = combined["receptor_id"].astype(str).str.strip().str.lower()
    combined = combined[combined["receptor_id"].ne("")].copy()

    combined["_has_sequence"] = combined["sequence"].map(lambda seq: is_protein_sequence(clean_sequence(seq)))
    combined["_has_uniprot"] = combined["uniprot_id"].map(lambda val: clean_cell(val) != "")
    combined = combined.sort_values(
        ["receptor_id", "_has_sequence", "_has_uniprot", "sequence_source"],
        ascending=[True, False, False, True],
    )
    combined = combined.drop_duplicates("receptor_id", keep="first")
    combined = combined.drop(columns=["_has_sequence", "_has_uniprot"])
    return combined.reset_index(drop=True)


def split_core_and_unresolved_receptors(
    seeds: pd.DataFrame,
    require_sequences: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if seeds.empty:
        return empty_core_receptor_frame(), empty_receptor_seed_frame()

    seeds = seeds.copy()
    seeds["sequence"] = seeds["sequence"].map(clean_sequence)
    has_sequence = seeds["sequence"].map(is_protein_sequence)

    core = seeds[has_sequence].copy() if require_sequences else seeds.copy()
    unresolved = seeds[~has_sequence].copy()

    if core.empty:
        return empty_core_receptor_frame(), unresolved.reset_index(drop=True)

    core["has_gpcrdb_numbering"] = False
    core["sequence_length"] = core["sequence"].str.len()
    for col in ["uniprot_id", "gpcrdb_entry_name", "iuphar_name", "species", "gpcr_class", "receptor_family"]:
        if col not in core.columns:
            core[col] = ""

    core_cols = [
        "receptor_id", "uniprot_id", "gpcrdb_entry_name", "iuphar_name", "species",
        "gpcr_class", "receptor_family", "sequence", "sequence_length", "has_gpcrdb_numbering",
    ]
    core = core.loc[:, core_cols].sort_values("receptor_id").reset_index(drop=True)
    return core, unresolved.reset_index(drop=True)


def resolve_receptor_seeds_with_uniprot(seeds: pd.DataFrame, cfg: UniProtConfig) -> pd.DataFrame:
    if seeds.empty:
        return seeds

    resolved_rows: list[dict[str, Any]] = []
    for row in seeds.to_dict(orient="records"):
        sequence = clean_sequence(row.get("sequence", ""))
        if is_protein_sequence(sequence):
            row["sequence"] = sequence
            resolved_rows.append(row) #type: ignore[assignment]
            continue

        record = find_uniprot_record_for_seed(row, cfg) #type: ignore[assignment]
        if record is not None:
            # Do not overwrite stable Glens/GPCRdb identifiers from the seed row.
            for key, value in record.items():
                if key in {"uniprot_id", "uniprot_entry_name", "sequence", "sequence_source", "species"}:
                    row[key] = value
                elif not clean_cell(row.get(key, "")):
                    row[key] = value
        resolved_rows.append(row) #type: ignore[assignment]

    return pd.DataFrame(resolved_rows)


def find_uniprot_record_for_seed(row: dict[str, Any], cfg: UniProtConfig) -> dict[str, str] | None:
    accession = clean_cell(row.get("uniprot_id", ""))
    if accession:
        record = fetch_uniprot_record(accession, cfg)
        if record is not None:
            return record

    entry_name = clean_cell(row.get("gpcrdb_entry_name", "")) or clean_cell(row.get("receptor_id", ""))
    if entry_name:
        record = search_uniprot_by_entry_name(entry_name, cfg)
        if record is not None:
            return record

    for text_key in ("source_receptor_name", "iuphar_name"):
        text = clean_cell(row.get(text_key, ""))
        if text:
            record = search_uniprot_by_text(text, cfg)
            if record is not None:
                return record

    return None


def fetch_uniprot_record(accession: str, cfg: UniProtConfig) -> dict[str, str] | None:
    cache_key = f"accession_{safe_filename(accession)}.json"
    payload = cached_uniprot_json(cache_key, cfg)
    if payload is None:
        payload = safe_request_json(f"{UNIPROT_API}/{accession}.json", cfg.timeout_seconds)
        if payload is None:
            return None
        write_uniprot_cache(cache_key, payload, cfg)
    return parse_uniprot_json(payload)


def search_uniprot_by_entry_name(entry_name: str, cfg: UniProtConfig) -> dict[str, str] | None:
    uniprot_entry_id = gpcrdb_name_to_uniprot_entry_id(entry_name)
    if not uniprot_entry_id:
        return None

    cache_key = f"entry_{safe_filename(uniprot_entry_id)}.json"
    payload = cached_uniprot_json(cache_key, cfg)
    if payload is None:
        query_parts = [f"id:{uniprot_entry_id}", f"organism_id:{cfg.organism_id}"]
        if cfg.reviewed_only:
            query_parts.append("reviewed:true")
        payload = uniprot_search_payload(query_parts, size=1, cfg=cfg)
        if payload is None:
            return None
        write_uniprot_cache(cache_key, payload, cfg)

    return first_parsed_uniprot_result(payload)


def search_uniprot_by_text(text: str, cfg: UniProtConfig) -> dict[str, str] | None:
    query_text = clean_cell(text)
    if not query_text:
        return None

    cache_key = f"text_{safe_filename(query_text)}.json"
    payload = cached_uniprot_json(cache_key, cfg)
    if payload is None:
        query_parts = [query_text, f"organism_id:{cfg.organism_id}"]
        if cfg.reviewed_only:
            query_parts.append("reviewed:true")
        payload = uniprot_search_payload(query_parts, size=5, cfg=cfg)
        if payload is None:
            return None
        write_uniprot_cache(cache_key, payload, cfg)

    return first_parsed_uniprot_result(payload)


def uniprot_search_payload(query_parts: list[str], size: int, cfg: UniProtConfig) -> dict[str, Any] | None:
    query = " AND ".join(query_parts)
    params = urlencode({
        "query": query,
        "format": "json",
        "size": str(size),
        "fields": UNIPROT_FIELDS,
    })
    return safe_request_json(f"{UNIPROT_API}/search?{params}", cfg.timeout_seconds)


def safe_request_json(url: str, timeout_seconds: int) -> dict[str, Any] | None:
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "glens-source-ingestion/0.1"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # nosec: public UniProt API only
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return None


def cached_uniprot_json(cache_key: str, cfg: UniProtConfig) -> dict[str, Any] | None:
    if cfg.cache_dir is None:
        return None
    path = cfg.cache_dir / cache_key
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_uniprot_cache(cache_key: str, payload: dict[str, Any], cfg: UniProtConfig) -> None:
    if cfg.cache_dir is None:
        return
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    with (cfg.cache_dir / cache_key).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def first_parsed_uniprot_result(payload: dict[str, Any]) -> dict[str, str] | None:
    results = payload.get("results", []) if isinstance(payload, dict) else []
    for result in results:
        parsed = parse_uniprot_json(result)
        if parsed is not None:
            return parsed
    return None


def parse_uniprot_json(payload: dict[str, Any]) -> dict[str, str] | None:
    accession = clean_cell(payload.get("primaryAccession", ""))
    entry = clean_cell(payload.get("uniProtkbId", ""))
    sequence = clean_sequence((payload.get("sequence") or {}).get("value", ""))
    if not accession or not is_protein_sequence(sequence):
        return None

    protein_description = payload.get("proteinDescription") or {}
    recommended = protein_description.get("recommendedName") or {}
    full_name = (recommended.get("fullName") or {}).get("value", "")
    organism = clean_cell((payload.get("organism") or {}).get("scientificName", ""))
    species = "human" if organism.lower() == "homo sapiens" else organism

    return {
        "uniprot_id": accession,
        "uniprot_entry_name": entry,
        "sequence": sequence,
        "sequence_source": "uniprot_api",
        "iuphar_name": clean_cell(full_name),
        "species": species,
    }


# -----------------------------------------------------------------------------
# Reference tables
# -----------------------------------------------------------------------------


def g_alpha_family_map_frame() -> pd.DataFrame:
    rows = []
    for subtype, family in SUBTYPE_TO_FAMILY.items():
        rows.append({
            "g_alpha_subtype": subtype,
            "g_alpha_family": family,
            "canonical_family": family,
            "family_rank_group": "G15" if subtype == "G15" else family,
            "special_handling": "flag_G15" if subtype == "G15" else "",
        })
    return pd.DataFrame(rows).sort_values(["g_alpha_family", "g_alpha_subtype"]).reset_index(drop=True)


def amino_acid_properties_frame() -> pd.DataFrame:
    rows = [
        {"aa": "A", "name": "alanine", "charge": 0, "polarity": "nonpolar", "hydropathy": 1.8, "volume": 88.6, "is_aromatic": False, "is_proline": False, "is_glycine": False},
        {"aa": "R", "name": "arginine", "charge": 1, "polarity": "positive", "hydropathy": -4.5, "volume": 173.4, "is_aromatic": False, "is_proline": False, "is_glycine": False},
        {"aa": "N", "name": "asparagine", "charge": 0, "polarity": "polar", "hydropathy": -3.5, "volume": 114.1, "is_aromatic": False, "is_proline": False, "is_glycine": False},
        {"aa": "D", "name": "aspartate", "charge": -1, "polarity": "negative", "hydropathy": -3.5, "volume": 111.1, "is_aromatic": False, "is_proline": False, "is_glycine": False},
        {"aa": "C", "name": "cysteine", "charge": 0, "polarity": "polar", "hydropathy": 2.5, "volume": 108.5, "is_aromatic": False, "is_proline": False, "is_glycine": False},
        {"aa": "Q", "name": "glutamine", "charge": 0, "polarity": "polar", "hydropathy": -3.5, "volume": 143.8, "is_aromatic": False, "is_proline": False, "is_glycine": False},
        {"aa": "E", "name": "glutamate", "charge": -1, "polarity": "negative", "hydropathy": -3.5, "volume": 138.4, "is_aromatic": False, "is_proline": False, "is_glycine": False},
        {"aa": "G", "name": "glycine", "charge": 0, "polarity": "nonpolar", "hydropathy": -0.4, "volume": 60.1, "is_aromatic": False, "is_proline": False, "is_glycine": True},
        {"aa": "H", "name": "histidine", "charge": 0, "polarity": "positive", "hydropathy": -3.2, "volume": 153.2, "is_aromatic": True, "is_proline": False, "is_glycine": False},
        {"aa": "I", "name": "isoleucine", "charge": 0, "polarity": "nonpolar", "hydropathy": 4.5, "volume": 166.7, "is_aromatic": False, "is_proline": False, "is_glycine": False},
        {"aa": "L", "name": "leucine", "charge": 0, "polarity": "nonpolar", "hydropathy": 3.8, "volume": 166.7, "is_aromatic": False, "is_proline": False, "is_glycine": False},
        {"aa": "K", "name": "lysine", "charge": 1, "polarity": "positive", "hydropathy": -3.9, "volume": 168.6, "is_aromatic": False, "is_proline": False, "is_glycine": False},
        {"aa": "M", "name": "methionine", "charge": 0, "polarity": "nonpolar", "hydropathy": 1.9, "volume": 162.9, "is_aromatic": False, "is_proline": False, "is_glycine": False},
        {"aa": "F", "name": "phenylalanine", "charge": 0, "polarity": "nonpolar", "hydropathy": 2.8, "volume": 189.9, "is_aromatic": True, "is_proline": False, "is_glycine": False},
        {"aa": "P", "name": "proline", "charge": 0, "polarity": "nonpolar", "hydropathy": -1.6, "volume": 112.7, "is_aromatic": False, "is_proline": True, "is_glycine": False},
        {"aa": "S", "name": "serine", "charge": 0, "polarity": "polar", "hydropathy": -0.8, "volume": 89.0, "is_aromatic": False, "is_proline": False, "is_glycine": False},
        {"aa": "T", "name": "threonine", "charge": 0, "polarity": "polar", "hydropathy": -0.7, "volume": 116.1, "is_aromatic": False, "is_proline": False, "is_glycine": False},
        {"aa": "W", "name": "tryptophan", "charge": 0, "polarity": "nonpolar", "hydropathy": -0.9, "volume": 227.8, "is_aromatic": True, "is_proline": False, "is_glycine": False},
        {"aa": "Y", "name": "tyrosine", "charge": 0, "polarity": "polar", "hydropathy": -1.3, "volume": 193.6, "is_aromatic": True, "is_proline": False, "is_glycine": False},
        {"aa": "V", "name": "valine", "charge": 0, "polarity": "nonpolar", "hydropathy": 4.2, "volume": 140.0, "is_aromatic": False, "is_proline": False, "is_glycine": False},
    ]
    return pd.DataFrame(rows).sort_values("aa").reset_index(drop=True)


# -----------------------------------------------------------------------------
# Label/value normalization helpers
# -----------------------------------------------------------------------------


def infer_common_map_label_state(*, rank: str, support: str, percent: str, quant: str) -> str:
    tokens = {
        clean_cell(rank).lower(),
        clean_cell(support).lower(),
        clean_cell(percent).lower(),
        clean_cell(quant).lower(),
    }
    if "nc" in tokens:
        return "negative_nc"
    if all(token in {"", "-"} for token in tokens):
        return "missing"

    numeric_values = [parse_number(rank), parse_number(support), parse_number(percent), parse_number(quant)]
    if any(value is not None and value > 0 for value in numeric_values):
        return "positive"
    if any(value == 0 for value in numeric_values):
        return "negative_nc"
    return "unknown"


def normalized_common_map_value(*, label_state: str, percent: str, quant: str) -> float | None:
    quant_number = parse_number(quant)
    if quant_number is not None:
        return float(quant_number)
    percent_number = parse_number(percent)
    if percent_number is not None:
        return float(percent_number) / 100.0
    if label_state == "positive":
        return 1.0
    if label_state == "negative_nc":
        return 0.0
    return None


def evidence_weight_for_source(source_lab: str, support: str) -> float:
    source = source_lab.strip().lower()
    base = 0.70
    if source == "gproteindb":
        base = 1.00
    elif source in {"bouvier", "martemyanov", "inoue"}:
        base = 0.85
    elif source == "gtopdb":
        base = 0.60

    support_number = parse_number(support)
    if support_number is not None and support_number >= 2:
        base += 0.05
    return min(base, 1.0)


# -----------------------------------------------------------------------------
# Small utilities and frame schemas
# -----------------------------------------------------------------------------


def finalize_frame(
    rows: list[dict[str, Any]],
    empty: pd.DataFrame,
    *,
    sort_cols: list[str],
    dedupe_cols: list[str] | None = None,
) -> pd.DataFrame:
    if not rows:
        return empty.copy()
    frame = pd.DataFrame(rows)
    for col in empty.columns:
        if col not in frame.columns:
            frame[col] = ""
    frame = frame.loc[:, list(empty.columns)]
    if dedupe_cols:
        frame = frame.drop_duplicates(dedupe_cols)
    return frame.sort_values(sort_cols).reset_index(drop=True)


def gpcrdb_name_to_uniprot_entry_id(name: str) -> str:
    text = clean_cell(name)
    if not text:
        return ""
    if text.lower().endswith("_human"):
        return f"{text[:-6].upper()}_HUMAN"
    return text.upper()


def is_protein_sequence(seq: Any) -> bool:
    text = clean_sequence(seq)
    return bool(text) and set(text).issubset(AA20)


def clean_sequence(seq: Any) -> str:
    return re.sub(r"\s+", "", clean_cell(seq)).upper()


def receptor_id_from_gpcrdb_url(url: str) -> str:
    match = re.search(r"/protein/([^/\s]+)", url)
    return match.group(1).strip().lower() if match else ""


def safe_receptor_id(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_").lower()
    return f"{cleaned}_human" if cleaned and not cleaned.endswith("_human") else cleaned


def make_assay_observation_id(
    *,
    row_number: int,
    receptor_id: str,
    source_lab: str,
    biosensor: str,
    reference: str,
    ligand: str,
) -> str:
    key = "|".join([str(row_number), receptor_id, source_lab, biosensor, reference, ligand])
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"ccm_{row_number:04d}_{digest}"


def parse_number(value: Any) -> float | None:
    text = clean_cell(value).replace("'", "")
    if text.lower() in {"", "-", "nc", "nan"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_rank(value: Any) -> int | None:
    number = parse_number(value)
    if number is None:
        return None
    return int(number)


def normalize_readout_type(parameter: str) -> str:
    text = clean_cell(parameter).lower()
    if not text or text == "-":
        return "binary"
    if "log" in text and "emax" in text:
        return "log_Emax_EC50"
    if "activation rate" in text:
        return "activation_rate"
    if "constitutive" in text:
        return "constitutive_activity"
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_") or "unknown"


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "<na>"}:
        return ""
    return text


def first_nonempty(*values: Any) -> str:
    for value in values:
        text = clean_cell(value)
        if text and text != "-":
            return text
    return ""


def normalize_reference(value: str) -> str:
    return " ".join(clean_cell(value).split())


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return cleaned[:180]


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def empty_common_coupling_frames() -> CommonCouplingFrames:
    return CommonCouplingFrames(
        receptor_seeds=empty_receptor_seed_frame(),
        assay_observations=empty_assay_observation_frame(),
        labels=empty_label_frame(),
    )


def empty_receptor_seed_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "receptor_id", "source_receptor_name", "iuphar_name", "gpcrdb_entry_name", "gpcrdb_url",
        "receptor_family", "gpcr_class", "species", "other_protein", "uniprot_id", "sequence",
        "sequence_source",
    ])


def empty_core_receptor_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "receptor_id", "uniprot_id", "gpcrdb_entry_name", "iuphar_name", "species",
        "gpcr_class", "receptor_family", "sequence", "sequence_length", "has_gpcrdb_numbering",
    ])


def empty_assay_observation_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "assay_observation_id", "receptor_id", "dataset", "biosensor", "downstream_steps",
        "gproteins_tested", "reference", "ligand_name", "ligand_context", "primary_family",
        "family_count", "primary_subtype",
    ])


def empty_label_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "assay_observation_id", "receptor_id", "variant_id", "dataset", "assay_type", "ligand_context",
        "cell_context", "g_alpha_subtype", "g_alpha_family", "readout_type", "value_raw",
        "value_normalized", "label_state", "evidence_weight", "evidence_kind", "rank_order",
        "percent_primary", "supporting_dataset_count", "primary_family", "primary_subtype",
    ])

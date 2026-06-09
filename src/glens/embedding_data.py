from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import yaml

from .build_core_datasets import AA20, normalize_columns, require_columns, write_parquet
from .mutation_data import read_table

AA_PROPERTY_BACKEND = "aa_properties_v1"
DEFAULT_REGION_POOLS = ("ICL2", "ICL3", "H8", "TM5_CYT", "TM6_CYT", "TM7_H8")
EMBEDDING_COLUMNS = tuple(f"embedding_{i}" for i in range(8))
DELTA_COLUMNS = tuple(f"delta_embedding_{i}" for i in range(8))
WT_WINDOW_COLUMNS = tuple(f"wt_window_embedding_{i}" for i in range(8))
MUT_WINDOW_COLUMNS = tuple(f"mut_window_embedding_{i}" for i in range(8))

# Lightweight, deterministic residue-property embedding used as the smoke-test backend.
# These values are intentionally simple and stable; a later ESM2 backend should preserve
# the same output table contracts, not replace the downstream artifact names.
_HYDROPATHY = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}
_VOLUME = {
    "A": 88.6, "C": 108.5, "D": 111.1, "E": 138.4, "F": 189.9,
    "G": 60.1, "H": 153.2, "I": 166.7, "K": 168.6, "L": 166.7,
    "M": 162.9, "N": 114.1, "P": 112.7, "Q": 143.8, "R": 173.4,
    "S": 89.0, "T": 116.1, "V": 140.0, "W": 227.8, "Y": 193.6,
}
_CHARGE = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.5}
_POLAR = set("CDEHKNQRSTY")
_AROMATIC = set("FHWY")
_SULFUR = set("CM")


@dataclass(frozen=True)
class EmbeddingDataConfig:
    receptor_manifest: Path
    generic_number_maps: Path
    mutation_manifest: Path | None
    residue_embeddings: Path
    region_embeddings: Path
    mutation_embedding_deltas: Path | None
    backend: str = AA_PROPERTY_BACKEND
    window_radius: int = 5
    region_pools: tuple[str, ...] = DEFAULT_REGION_POOLS


def build_embedding_data(config_path: str | Path) -> dict[str, str]:
    """Build residue, region-pool, and mutation-local embedding tables."""
    cfg = load_embedding_data_config(config_path)
    receptors = read_table(cfg.receptor_manifest)
    numbering = read_table(cfg.generic_number_maps)

    residue_embeddings = build_residue_embedding_frame(receptors, numbering, backend=cfg.backend)
    region_embeddings = build_region_embedding_frame(residue_embeddings, cfg.region_pools)

    write_parquet(residue_embeddings, cfg.residue_embeddings)
    write_parquet(region_embeddings, cfg.region_embeddings)

    summary = {
        "residue_embeddings": str(cfg.residue_embeddings),
        "region_embeddings": str(cfg.region_embeddings),
    }

    if cfg.mutation_manifest is not None and cfg.mutation_manifest.exists() and cfg.mutation_embedding_deltas is not None:
        mutation_manifest = read_table(cfg.mutation_manifest)
        deltas = build_mutation_embedding_delta_frame(
            mutation_manifest,
            receptors,
            backend=cfg.backend,
            window_radius=cfg.window_radius,
        )
        write_parquet(deltas, cfg.mutation_embedding_deltas)
        summary["mutation_embedding_deltas"] = str(cfg.mutation_embedding_deltas)

    return summary


def load_embedding_data_config(path: str | Path) -> EmbeddingDataConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    base = path.parent.parent if path.parent.name == "configs" else Path.cwd()
    inputs = raw.get("inputs", {})
    outputs = raw.get("outputs", {})
    embedding = raw.get("embedding", {})

    def p(value: object) -> Path | None:
        if value is None or value == "":
            return None
        candidate = Path(str(value))
        return candidate if candidate.is_absolute() else base / candidate

    receptor_manifest = p(inputs.get("receptor_manifest"))
    generic_number_maps = p(inputs.get("generic_number_maps"))
    if receptor_manifest is None or generic_number_maps is None:
        raise ValueError("Config must define inputs.receptor_manifest and inputs.generic_number_maps")

    residue_embeddings = p(outputs.get("residue_embeddings", "data/embeddings/residue_embeddings.parquet"))
    region_embeddings = p(outputs.get("region_embeddings", "data/embeddings/region_embeddings.parquet"))
    if residue_embeddings is None or region_embeddings is None:
        raise ValueError("Config must define residue and region embedding outputs")

    region_pools = tuple(str(name).strip().upper() for name in embedding.get("region_pools", DEFAULT_REGION_POOLS))
    window_radius = int(embedding.get("window_radius", 5))
    if window_radius < 0:
        raise ValueError("embedding.window_radius must be non-negative")

    return EmbeddingDataConfig(
        receptor_manifest=receptor_manifest,
        generic_number_maps=generic_number_maps,
        mutation_manifest=p(inputs.get("mutation_manifest")),
        residue_embeddings=residue_embeddings,
        region_embeddings=region_embeddings,
        mutation_embedding_deltas=p(outputs.get("mutation_embedding_deltas", "data/embeddings/mutation_embedding_deltas.parquet")),
        backend=str(embedding.get("backend", AA_PROPERTY_BACKEND)),
        window_radius=window_radius,
        region_pools=region_pools,
    )


def build_residue_embedding_frame(
    receptors: pd.DataFrame,
    generic_number_maps: pd.DataFrame,
    *,
    backend: str = AA_PROPERTY_BACKEND,
) -> pd.DataFrame:
    """Return one deterministic embedding row per WT receptor residue."""
    validate_backend(backend)
    recs = normalize_columns(receptors)
    require_columns(recs, ["receptor_id", "sequence"])
    recs["receptor_id"] = recs["receptor_id"].astype(str).str.strip().str.lower()
    recs["sequence"] = recs["sequence"].astype(str).str.replace(r"\s+", "", regex=True).str.upper()

    maps = normalize_columns(generic_number_maps)
    if maps.empty:
        maps = pd.DataFrame(columns=["receptor_id", "seq_pos", "aa", "region", "gpcrdb_number", "is_intracellular_face", "is_candidate_interface"])
    if "receptor_id" in maps.columns:
        maps["receptor_id"] = maps["receptor_id"].astype(str).str.strip().str.lower()
    if "seq_pos" in maps.columns:
        maps["seq_pos"] = pd.to_numeric(maps["seq_pos"], errors="coerce").astype("Int64")

    map_lookup = {
        (str(row.receptor_id), int(row.seq_pos)): row._asdict()
        for row in maps.dropna(subset=["seq_pos"]).itertuples(index=False)
        if hasattr(row, "receptor_id") and hasattr(row, "seq_pos")
    }

    rows: list[dict[str, Any]] = []
    for rec in recs.itertuples(index=False):
        receptor_id = str(rec.receptor_id)
        sequence = str(rec.sequence)
        invalid = sorted(set(sequence) - AA20)
        if invalid:
            raise ValueError(f"Invalid amino acid(s) in {receptor_id}: {', '.join(invalid)}")
        for seq_pos, aa in enumerate(sequence, start=1):
            map_row = map_lookup.get((receptor_id, seq_pos), {})
            values = residue_embedding(aa, backend=backend)
            row = {
                "receptor_id": receptor_id,
                "variant_id": "WT",
                "seq_pos": seq_pos,
                "aa": aa,
                "region": map_row.get("region", "unknown"),
                "gpcrdb_number": map_row.get("gpcrdb_number", pd.NA),
                "is_intracellular_face": bool(map_row.get("is_intracellular_face", False)),
                "is_candidate_interface": bool(map_row.get("is_candidate_interface", False)),
                "embedding_backend": backend,
                "embedding_dim": len(values),
            }
            row.update(dict(zip(EMBEDDING_COLUMNS, values)))
            rows.append(row)

    return pd.DataFrame(rows)


def build_region_embedding_frame(
    residue_embeddings: pd.DataFrame,
    region_pools: Sequence[str] = DEFAULT_REGION_POOLS,
) -> pd.DataFrame:
    """Pool residue embeddings into stable GPCR intracellular/interface regions."""
    require_columns(residue_embeddings, ["receptor_id", "variant_id", "seq_pos", "region", *EMBEDDING_COLUMNS])
    rows: list[dict[str, Any]] = []
    pools = tuple(str(pool).strip().upper() for pool in region_pools)

    for (receptor_id, variant_id), group in residue_embeddings.groupby(["receptor_id", "variant_id"], sort=True):
        for pool in pools:
            selected = select_region_pool(group, pool)
            pooled = mean_vector(selected, EMBEDDING_COLUMNS)
            row = {
                "receptor_id": receptor_id,
                "variant_id": variant_id,
                "region_pool": pool,
                "residue_count": int(len(selected)),
                "embedding_backend": group["embedding_backend"].iloc[0] if "embedding_backend" in group else AA_PROPERTY_BACKEND,
                "embedding_dim": len(EMBEDDING_COLUMNS),
            }
            row.update(dict(zip(EMBEDDING_COLUMNS, pooled)))
            rows.append(row)

    return pd.DataFrame(rows)


def build_mutation_embedding_delta_frame(
    mutation_manifest: pd.DataFrame,
    receptors: pd.DataFrame,
    *,
    backend: str = AA_PROPERTY_BACKEND,
    window_radius: int = 5,
) -> pd.DataFrame:
    """Build local WT-to-mutant window embedding deltas for each substitution row."""
    validate_backend(backend)
    muts = normalize_columns(mutation_manifest)
    require_columns(muts, ["variant_id", "receptor_id", "canonical_mutation", "seq_pos", "wt_aa", "mut_aa", "mutant_sequence"])

    recs = normalize_columns(receptors)
    require_columns(recs, ["receptor_id", "sequence"])
    recs["receptor_id"] = recs["receptor_id"].astype(str).str.strip().str.lower()
    wt_sequence_by_receptor = dict(zip(recs["receptor_id"], recs["sequence"].astype(str)))

    rows: list[dict[str, Any]] = []
    for row in muts.itertuples(index=False):
        row_dict = row._asdict()
        receptor_id = str(row_dict["receptor_id"]).strip().lower()
        if receptor_id not in wt_sequence_by_receptor:
            raise ValueError(f"Unknown receptor_id in mutation manifest: {receptor_id}")

        seq_pos = int(row_dict["seq_pos"])
        wt_sequence = wt_sequence_by_receptor[receptor_id].replace(" ", "").replace("\n", "").upper()
        mutant_sequence = str(row_dict["mutant_sequence"]).replace(" ", "").replace("\n", "").upper()
        window_start, window_end = sequence_window(seq_pos, len(wt_sequence), window_radius)
        wt_values = mean_sequence_embedding(wt_sequence[window_start - 1:window_end], backend=backend)
        mut_values = mean_sequence_embedding(mutant_sequence[window_start - 1:window_end], backend=backend)
        delta_values = tuple(mut - wt for wt, mut in zip(wt_values, mut_values))

        out = {
            "variant_id": str(row_dict["variant_id"]),
            "receptor_id": receptor_id,
            "canonical_mutation": str(row_dict["canonical_mutation"]),
            "substitution_index": int(row_dict.get("substitution_index", 1)),
            "seq_pos": seq_pos,
            "gpcrdb_number": row_dict.get("gpcrdb_number", pd.NA),
            "region": row_dict.get("region", "unknown"),
            "wt_aa": str(row_dict["wt_aa"]),
            "mut_aa": str(row_dict["mut_aa"]),
            "window_start": window_start,
            "window_end": window_end,
            "window_radius": window_radius,
            "embedding_backend": backend,
            "embedding_dim": len(EMBEDDING_COLUMNS),
        }
        out.update(dict(zip(WT_WINDOW_COLUMNS, wt_values)))
        out.update(dict(zip(MUT_WINDOW_COLUMNS, mut_values)))
        out.update(dict(zip(DELTA_COLUMNS, delta_values)))
        rows.append(out)

    return pd.DataFrame(rows).sort_values(["receptor_id", "variant_id", "substitution_index"]).reset_index(drop=True)


def select_region_pool(frame: pd.DataFrame, region_pool: str) -> pd.DataFrame:
    pool = region_pool.upper()
    region = frame["region"].astype(str).str.upper()
    gpcrdb_number = frame["gpcrdb_number"].astype(str)

    if pool in {"ICL1", "ICL2", "ICL3", "H8"}:
        return frame[region == pool]
    if pool == "TM5_CYT":
        return frame[(region == "TM5") & gpcrdb_number.str.startswith("5x", na=False)]
    if pool == "TM6_CYT":
        return frame[(region == "TM6") & gpcrdb_number.str.startswith("6x", na=False)]
    if pool == "TM7_H8":
        return frame[region.isin({"TM7", "H8"})]
    if pool == "INTRACELLULAR_INTERFACE":
        return frame[frame.get("is_candidate_interface", False).astype(bool)]
    return frame[region == pool]


def residue_embedding(aa: str, *, backend: str = AA_PROPERTY_BACKEND) -> tuple[float, ...]:
    validate_backend(backend)
    token = str(aa).strip().upper()
    if token not in AA20:
        raise ValueError(f"Unsupported amino acid for embedding: {aa}")
    hydropathy = _HYDROPATHY[token] / 4.5
    volume = (_VOLUME[token] - 60.1) / (227.8 - 60.1)
    charge = _CHARGE.get(token, 0.0)
    polarity = 1.0 if token in _POLAR else 0.0
    aromatic = 1.0 if token in _AROMATIC else 0.0
    sulfur = 1.0 if token in _SULFUR else 0.0
    proline = 1.0 if token == "P" else 0.0
    glycine = 1.0 if token == "G" else 0.0
    return (hydropathy, volume, charge, polarity, aromatic, sulfur, proline, glycine)


def mean_sequence_embedding(sequence: str, *, backend: str = AA_PROPERTY_BACKEND) -> tuple[float, ...]:
    seq = str(sequence).replace(" ", "").replace("\n", "").upper()
    if not seq:
        return tuple(0.0 for _ in EMBEDDING_COLUMNS)
    vectors = [residue_embedding(aa, backend=backend) for aa in seq]
    return tuple(sum(vector[i] for vector in vectors) / len(vectors) for i in range(len(EMBEDDING_COLUMNS)))


def mean_vector(frame: pd.DataFrame, columns: Iterable[str]) -> tuple[float, ...]:
    cols = tuple(columns)
    if frame.empty:
        return tuple(0.0 for _ in cols)
    return tuple(float(frame[col].mean()) for col in cols)


def sequence_window(seq_pos: int, sequence_length: int, radius: int) -> tuple[int, int]:
    start = max(1, seq_pos - radius)
    end = min(sequence_length, seq_pos + radius)
    return start, end


def validate_backend(backend: str) -> None:
    if backend != AA_PROPERTY_BACKEND:
        raise ValueError(
            f"Unsupported embedding backend '{backend}'. "
            f"Only '{AA_PROPERTY_BACKEND}' is implemented in this lightweight layer."
        )

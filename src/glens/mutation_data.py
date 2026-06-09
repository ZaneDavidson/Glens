from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import pandas as pd
import yaml

from .build_core_datasets import (
    AA20,
    fix_columns_to_target,
    normalize_columns,
    require_columns,
    write_parquet,
)

_MUTATION_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")
_MUTATION_SPLIT_RE = re.compile(r"[+,;\s]+")


@dataclass(frozen=True, order=True)
class MutationSpec:
    seq_pos: int
    wt_aa: str
    mut_aa: str

    @property
    def token(self) -> str:
        return f"{self.wt_aa}{self.seq_pos}{self.mut_aa}"


@dataclass(frozen=True)
class MutationDataConfig:
    receptor_manifest: Path
    generic_number_maps: Path
    mutations_csv: Path
    mutation_manifest: Path


def build_mutation_data(config_path: str | Path) -> dict[str, str]:
    cfg = load_mutation_data_config(config_path)
    receptors = read_table(cfg.receptor_manifest)
    numbering = read_table(cfg.generic_number_maps)
    mutations = read_table(cfg.mutations_csv)
    manifest = build_mutation_manifest_frame(mutations, receptors, numbering)
    write_parquet(manifest, cfg.mutation_manifest)
    return {"mutation_manifest": str(cfg.mutation_manifest)}


def load_mutation_data_config(path: str | Path) -> MutationDataConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    base = path.parent.parent if path.parent.name == "configs" else Path.cwd()
    inputs = raw.get("inputs", {})
    outputs = raw.get("outputs", {})

    def p(value: object) -> Path | None:
        if value is None or value == "":
            return None
        candidate = Path(str(value))
        return candidate if candidate.is_absolute() else base / candidate

    receptor_manifest = p(inputs.get("receptor_manifest"))
    generic_number_maps = p(inputs.get("generic_number_maps"))
    mutations_csv = p(inputs.get("mutations_csv"))
    if receptor_manifest is None or generic_number_maps is None or mutations_csv is None:
        raise ValueError(
            "Config must define inputs.receptor_manifest, inputs.generic_number_maps, and inputs.mutations_csv"
        )

    mutation_manifest = p(outputs.get("mutation_manifest", "data/interim/mutation_manifest.parquet"))
    if mutation_manifest is None:
        raise ValueError("Config must define outputs.mutation_manifest")

    return MutationDataConfig(
        receptor_manifest=receptor_manifest,
        generic_number_maps=generic_number_maps,
        mutations_csv=mutations_csv,
        mutation_manifest=mutation_manifest,
    )


def parse_mutation_string(text: str) -> tuple[MutationSpec, ...]:
    if text is None or not str(text).strip():
        raise ValueError("Mutation string is empty")

    tokens = [tok.strip().upper() for tok in _MUTATION_SPLIT_RE.split(str(text).strip()) if tok.strip()]
    if not tokens:
        raise ValueError("Mutation string is empty")

    specs: list[MutationSpec] = []
    seen_positions: set[int] = set()
    for token in tokens:
        match = _MUTATION_RE.match(token)
        if not match:
            raise ValueError(f"Invalid mutation token '{token}'. Expected format like L147A")
        wt_aa, pos_text, mut_aa = match.groups()
        if wt_aa not in AA20 or mut_aa not in AA20:
            raise ValueError(f"Invalid amino acid in mutation token '{token}'")
        seq_pos = int(pos_text)
        if seq_pos < 1:
            raise ValueError(f"Mutation position must be 1-indexed and positive: '{token}'")
        if seq_pos in seen_positions:
            raise ValueError(f"Multiple substitutions target residue {seq_pos}; split or resolve this variant first")
        seen_positions.add(seq_pos)
        specs.append(MutationSpec(seq_pos=seq_pos, wt_aa=wt_aa, mut_aa=mut_aa))
    return tuple(sorted(specs))


def canonical_mutation_string(specs: tuple[MutationSpec, ...]) -> str:
    return "+".join(spec.token for spec in specs)


def apply_mutation_string(sequence: str, mutation_string: str) -> str:
    return apply_mutations(sequence, parse_mutation_string(mutation_string))


def apply_mutations(sequence: str, specs: tuple[MutationSpec, ...]) -> str:
    seq = str(sequence).replace(" ", "").replace("\n", "").upper()
    if not seq or not set(seq).issubset(AA20):
        raise ValueError("Sequence must contain only the 20 standard amino acids")

    chars = list(seq)
    for spec in specs:
        if spec.seq_pos > len(chars):
            raise ValueError(f"Mutation {spec.token} is outside sequence length {len(chars)}")
        observed = chars[spec.seq_pos - 1]
        if observed != spec.wt_aa:
            raise ValueError(
                f"WT residue mismatch for {spec.token}: sequence has {observed} at position {spec.seq_pos}"
            )
        chars[spec.seq_pos - 1] = spec.mut_aa
    return "".join(chars)


def build_mutation_manifest_frame(
    mutations: pd.DataFrame,
    receptors: pd.DataFrame,
    generic_number_maps: pd.DataFrame,
) -> pd.DataFrame:
    muts = normalize_columns(mutations)
    muts = fix_columns_to_target(muts, "mutation_string", ["mutation", "mutations", "variant", "substitution"])
    require_columns(muts, ["receptor_id", "mutation_string"])

    for col, default in {
        "variant_id": pd.NA,
        "source": "user",
        "has_experimental_label": False,
    }.items():
        if col not in muts.columns:
            muts[col] = default

    recs = normalize_columns(receptors)
    require_columns(recs, ["receptor_id", "sequence"])
    recs["receptor_id"] = recs["receptor_id"].astype(str).str.strip().str.lower()
    sequence_by_receptor = dict(zip(recs["receptor_id"], recs["sequence"].astype(str)))

    maps = normalize_columns(generic_number_maps)
    if not maps.empty:
        maps["receptor_id"] = maps["receptor_id"].astype(str).str.strip().str.lower()
        maps["seq_pos"] = pd.to_numeric(maps["seq_pos"], errors="coerce").astype("Int64")

    rows: list[dict[str, Any]] = []
    for row in muts.itertuples(index=False):
        # row may be a namedtuple from itertuples or another row-like object; try to
        # get a dict representation robustly and avoid static-analysis false-positives
        try:
            row_dict = row._asdict()  # type: ignore
        except Exception:
            row_dict = dict(getattr(row, "__dict__", {}) or {})
        receptor_id = str(row_dict["receptor_id"]).strip().lower()
        if receptor_id not in sequence_by_receptor:
            raise ValueError(f"Unknown receptor_id in mutations table: {receptor_id}")

        mutation_string = str(row_dict["mutation_string"]).strip()
        specs = parse_mutation_string(mutation_string)
        canonical = canonical_mutation_string(specs)
        mutant_sequence = apply_mutations(sequence_by_receptor[receptor_id], specs)
        variant_id = row_dict.get("variant_id")
        if pd.isna(variant_id) or not str(variant_id).strip():
            variant_id = make_variant_id(receptor_id, canonical)

        for index, spec in enumerate(specs, start=1):
            map_row = lookup_numbering(maps, receptor_id, spec.seq_pos)
            rows.append({
                "variant_id": str(variant_id),
                "receptor_id": receptor_id,
                "mutation_string": mutation_string,
                "canonical_mutation": canonical,
                "mutation_count": len(specs),
                "substitution_index": index,
                "seq_pos": spec.seq_pos,
                "gpcrdb_number": map_row.get("gpcrdb_number", pd.NA),
                "wt_aa": spec.wt_aa,
                "mut_aa": spec.mut_aa,
                "region": map_row.get("region", "unknown"),
                "is_intracellular_face": bool(map_row.get("is_intracellular_face", False)),
                "is_candidate_interface": bool(map_row.get("is_candidate_interface", False)),
                "mutant_sequence": mutant_sequence,
                "source": row_dict.get("source", "user"),
                "has_experimental_label": coerce_bool(row_dict.get("has_experimental_label", False)),
            })

    columns = [
        "variant_id", "receptor_id", "mutation_string", "canonical_mutation", "mutation_count",
        "substitution_index", "seq_pos", "gpcrdb_number", "wt_aa", "mut_aa", "region",
        "is_intracellular_face", "is_candidate_interface", "mutant_sequence", "source",
        "has_experimental_label",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["receptor_id", "variant_id", "substitution_index"]
    ).reset_index(drop=True)


def lookup_numbering(maps: pd.DataFrame, receptor_id: str, seq_pos: int) -> dict[str, Any]:
    if maps.empty or "receptor_id" not in maps.columns or "seq_pos" not in maps.columns:
        return {}
    hit = maps[(maps["receptor_id"] == receptor_id) & (maps["seq_pos"] == seq_pos)]
    if hit.empty:
        return {}
    return {str(key): value for key, value in hit.iloc[0].to_dict().items()}


def make_variant_id(receptor_id: str, canonical_mutation: str) -> str:
    safe = canonical_mutation.replace("+", "_")
    return f"{receptor_id}__{safe}"


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    raise ValueError(f"Unsupported table format for {path}. Use .csv, .tsv, or .parquet")
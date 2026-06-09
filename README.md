# Glens

Glens is a lean GPCR G-alpha coupling data and mutation-effect pipeline.

The current build focuses on durable data artifacts rather than package sprawl:

- `receptor_manifest.parquet`
- `generic_number_maps.parquet`
- `label_fact_table.parquet`
- `mutation_manifest.parquet`
- `residue_embeddings.parquet`
- `region_embeddings.parquet`
- `mutation_embedding_deltas.parquet`

The embedding layer currently uses a deterministic amino-acid property backend so tests stay fast and dependency-light. The table contracts are designed so an ESM2 backend can replace the smoke backend later without changing downstream feature builders.

## Install

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Build the included sample artifacts

```bash
glens build-core-datasets --config configs/pass1.yaml
glens build-mutation-manifest --config configs/mutation_data.yaml
glens build-sequence-embeddings --config configs/embedding_data.yaml
```

Useful aliases:

```bash
glens bcd --config configs/pass1.yaml
glens bmm --config configs/mutation_data.yaml
glens bse --config configs/embedding_data.yaml
```

## One-off mutant FASTA inspection

```bash
glens print-mutant-fasta --receptor-id adrb2_human --mutation D130A+R131Q
```

This prints a FASTA record and does not write files.

## Core raw input files

### receptors CSV

Minimum useful columns:

```text
receptor_id,uniprot_id,gpcrdb_entry_name,iuphar_name,species,gpcr_class,receptor_family,sequence
```

### generic numbers CSV

Minimum useful columns:

```text
receptor_id,seq_pos,aa,region,gpcrdb_number
```

If this file is absent, Glens writes a fallback map with one row per sequence residue and unknown generic numbering.

### labels CSV

Minimum useful columns:

```text
receptor_id,variant_id,dataset,assay_type,ligand_context,cell_context,g_alpha_subtype,g_alpha_family,readout_type,value_raw,label_state,evidence_weight
```

`label_state` is normalized into one of:

```text
positive, negative_nc, missing, weak, unknown
```

The important rule is preserved: missing is not the same thing as tested non-coupling (`nc`).

### mutations CSV

Minimum useful columns:

```text
receptor_id,mutation_string,variant_id,source,has_experimental_label
```

Mutation strings are intentionally strict simple substitutions such as `D130A` or `D130A+R131Q`.

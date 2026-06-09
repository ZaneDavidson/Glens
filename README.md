# Glens pass 1

Pass 1 builds the three core data artifacts for the GPCR G-alpha coupling project:

- `receptor_manifest.parquet`
- `generic_number_maps.parquet`
- `label_fact_table.parquet`

This is intentionally small. It does not include embeddings, structure features, modeling, or Streamlit.

## Install

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Run the included sample

```bash
glens build-pass1 --config configs/pass1.yaml
```

Outputs are written to `data/interim/` by default.

## Expected raw input files

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

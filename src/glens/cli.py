from pathlib import Path

import typer

from .build_core_datasets import build_core_datasets
from .embedding_data import build_embedding_data
from .mutation_data import apply_mutation_string, build_mutation_data, read_table
from .source_ingestion import build_source_artifacts

app = typer.Typer(help="Glens: GPCR G-alpha coupling dataset and mutation utilities.")


@app.callback()
def main() -> None:
    """Build GPCR coupling data artifacts and inspect WT-to-mutant sequence changes."""
    pass


@app.command("pull-source-data")
@app.command("pull")
def ingest_source_data_command(
    config: Path = typer.Option(..., "--config", "-c", help="Path to source-data YAML config."),
) -> None:
    """Normalize upstream GPCR/G-protein sources into stable local inputs.

    This command ingests messy external or local source files, including the GPCRdb
    common coupling map and optional receptor seed tables, then writes
    inputs: receptors.csv, labels.csv, assay catalogs, source
    registries, amino-acid property tables, and unresolved receptor reports.
    """
    summary = build_source_artifacts(config)
    typer.echo("Built:")
    for name, path in summary.items():
        typer.echo(f"  {name}: {path}")


@app.command("build-core-datasets")
@app.command("bcd")
def build_core_datasets_command(
    config: Path = typer.Option(..., "--config", "-c", help="Path to core dataset YAML config."),
) -> None:
    """Build receptor, label, and GPCRdb-numbering parquet artifacts from raw tables."""
    summary = build_core_datasets(config)
    typer.echo("Built:")
    for name, path in summary.items():
        typer.echo(f"  {name}: {path}")


@app.command("build-mutation-manifest")
@app.command("bmm")
def build_mutation_manifest_command(
    config: Path = typer.Option(..., "--config", "-c", help="Path to mutation manifest YAML config."),
) -> None:
    """Validate mutation table entries and write mutation_manifest.parquet.

    This checks mutation syntax, validates WT residues against receptor sequences,
    annotates substitutions with GPCRdb numbering when available, and writes one
    manifest row per substitution.
    """
    summary = build_mutation_data(config)
    typer.echo("Built:")
    for name, path in summary.items():
        typer.echo(f"  {name}: {path}")


@app.command("build-sequence-embeddings")
@app.command("bse")
def build_sequence_embeddings_command(
    config: Path = typer.Option(..., "--config", "-c", help="Path to sequence embedding YAML config."),
) -> None:
    """Build residue embedding, region-pool, and mutation-local delta tables.

    The default backend is a deterministic amino-acid property embedding for fast
    smoke tests. Future ESM2 backends should preserve these output artifact
    contracts.
    """
    summary = build_embedding_data(config)
    typer.echo("Built:")
    for name, path in summary.items():
        typer.echo(f"  {name}: {path}")


@app.command("print-mutant-fasta")
@app.command("pmf")
def make_mutant_fasta_command(
    receptor_id: str = typer.Option(..., "--receptor-id", help="Receptor ID in receptor_manifest."),
    mutation: str = typer.Option(..., "--mutation", "-m", help="Mutation string, e.g. D130A or D130A+R131Q."),
    receptor_manifest: Path = typer.Option(
        Path("data/interim/receptor_manifest.parquet"),
        "--receptor-manifest",
        help="Path to receptor manifest parquet/csv.",
    ),
) -> None:
    """Apply a mutation string to one receptor and print the mutant FASTA sequence.
    This is a utility, will not write files.
    """
    receptors = read_table(receptor_manifest)
    receptors["receptor_id"] = receptors["receptor_id"].astype(str).str.strip().str.lower()

    query_id = receptor_id.strip().lower()
    hit = receptors[receptors["receptor_id"] == query_id]

    if hit.empty:
        raise typer.BadParameter(f"Unknown receptor_id: {receptor_id}")

    mutant = apply_mutation_string(str(hit.iloc[0]["sequence"]), mutation)
    typer.echo(f">{query_id}|{mutation.upper()}")
    typer.echo(mutant)
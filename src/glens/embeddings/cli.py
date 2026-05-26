"""CLI for sequence embedding."""

from collections.abc import Sequence
from pathlib import Path

import requests
import typer

from glens.data.gpcrdb import iter_gpcrdb_entry_names
from glens.data.sequences import SequenceRecord, read_sequence_table_csv
from glens.data.uniprot import fetch_uniprot_records
from glens.embeddings.artifact import EmbeddingRunConfig, write_sequence_embedding_artifact
from glens.embeddings.model import DEFAULT_MAX_RESIDUES, DEFAULT_MODEL_ID, load_plm

app = typer.Typer(no_args_is_help=True)


def _run_embedding(
    *,
    records: Sequence[SequenceRecord],
    output_npz: Path,
    model_id: str,
    batch_size: int,
    device: str,
    max_residues: int,
    stride: int | None,
    region_mask_json: Path | None,
    source_name: str,
    source_uri: str | None,
    write_audit_csv: bool,
    write_metadata_json: bool,
) -> None:
    typer.echo(f"Loading embedding model: {model_id}", err=True)
    tokenizer, model, torch_device = load_plm(model_id, device=device)
    typer.echo(f"Loaded embedding model on {torch_device}.", err=True)

    typer.echo(f"Embedding {len(records)} sequences on {torch_device}.")
    config = EmbeddingRunConfig(
        model_id=model_id,
        max_residues=max_residues,
        stride=stride,
        batch_size=batch_size,
        source_name=source_name,
        source_uri=source_uri,
        region_mask_json=region_mask_json,
    )
    write_sequence_embedding_artifact(
        records,
        output_npz,
        tokenizer=tokenizer,
        model=model,
        device=torch_device,
        config=config,
        write_audit_csv=write_audit_csv,
        write_metadata_json=write_metadata_json,
    )
    typer.echo(f"Wrote embeddings: {output_npz}")
    if write_audit_csv:
        typer.echo(f"Wrote audit CSV: {output_npz.with_suffix('.audit.csv')}")
    if write_metadata_json:
        typer.echo(f"Wrote metadata JSON: {output_npz.with_suffix('.metadata.json')}")


@app.command("sequence-table")
def sequence_table(
    input_csv: Path = typer.Argument(..., help="CSV with sequence_id and sequence columns."),
    output_npz: Path = typer.Argument(..., help="Output compressed NPZ embedding artifact."),
    id_column: str = typer.Option("sequence_id", help="Column containing stable row ids."),
    sequence_column: str = typer.Option("sequence", help="Column containing amino-acid sequences."),
    default_gpcrdb_entry_name: str | None = typer.Option(
        None,
        "--default-gpcrdb-entry-name",
        help=(
            "Optional GPCRdb entry name to attach to all rows that lack one. "
            "Useful for WT+mutant batches needing region masks, e.g. opn4_human."
        ),
    ),
    model_id: str = typer.Option(DEFAULT_MODEL_ID, help="Hugging Face ESM-2 model id."),
    batch_size: int = typer.Option(8, min=1, help="Sequences per embedding batch."),
    device: str = typer.Option("auto", help="auto, cuda, mps, or cpu."),
    max_residues: int = typer.Option(DEFAULT_MAX_RESIDUES, min=1, help="Maximum residues accepted by the model."),
    stride: int | None = typer.Option(None, min=1, help="Window stride for residue reconstruction."),
    region_mask_json: Path | None = typer.Option(None, "--region-mask-json", help="Optional region-mask cache JSON."),
    write_audit_csv: bool = typer.Option(True, "--audit/--no-audit", help="Write output audit CSV."),
    write_metadata_json: bool = typer.Option(True, "--md/--no-md", help="Write output metadata JSON."),
) -> None:
    """Embed arbitrary sequence rows from a source-agnostic CSV table."""
    typer.echo(f"Reading sequence table: {input_csv}", err=True)
    records = read_sequence_table_csv(
        input_csv,
        id_column=id_column,
        sequence_column=sequence_column,
        default_gpcrdb_entry_name=default_gpcrdb_entry_name,
    )
    typer.echo(f"Found {len(records)} sequence rows.", err=True)
    _run_embedding(
        records=records,
        output_npz=output_npz,
        model_id=model_id,
        batch_size=batch_size,
        device=device,
        max_residues=max_residues,
        stride=stride,
        region_mask_json=region_mask_json,
        source_name="sequence_table",
        source_uri=str(input_csv),
        write_audit_csv=write_audit_csv,
        write_metadata_json=write_metadata_json,
    )


@app.command("coupling-map")
def coupling_map(
    input_csv: Path = typer.Argument(..., help="GPCR common coupling map CSV."),
    output_npz: Path = typer.Argument(..., help="Output compressed NPZ embedding artifact."),
    gpcrdb_column: str = typer.Option("GPCRdb", help="Column containing GPCRdb URLs/entry names."),
    model_id: str = typer.Option(DEFAULT_MODEL_ID, help="Hugging Face ESM-2 model id."),
    batch_size: int = typer.Option(8, min=1, help="Sequences per embedding batch."),
    device: str = typer.Option("auto", help="auto, cuda, mps, or cpu."),
    max_residues: int = typer.Option(DEFAULT_MAX_RESIDUES, min=1, help="Maximum residues accepted by the model."),
    stride: int | None = typer.Option(None, min=1, help="Window stride for residue reconstruction."),
    region_mask_json: Path | None = typer.Option(None, "--region-mask-json", help="Optional region-mask cache JSON."),
    write_audit_csv: bool = typer.Option(True, "--audit/--no-audit", help="Write output audit CSV."),
    write_metadata_json: bool = typer.Option(True, "--md/--no-md", help="Write output metadata JSON."),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Parse the coupling map and report receptor count without resolving sequences or loading the model.",
    ),
) -> None:
    """Resolve unique GPCRdb entries through UniProt and embed the sequences."""
    typer.echo(f"Reading coupling map: {input_csv}", err=True)
    entry_names = tuple(iter_gpcrdb_entry_names(input_csv, gpcrdb_column))
    typer.echo(f"Found {len(entry_names)} unique GPCRdb entries.", err=True)

    if not entry_names:
        raise typer.BadParameter("No receptor entries found in the coupling map.")
    if dry_run:
        raise typer.Exit()

    records: list[SequenceRecord] = []
    typer.echo("Resolving UniProt sequences...", err=True)
    with requests.Session() as session:
        total = len(entry_names)
        for idx, entry_name in enumerate(entry_names, start=1):
            typer.echo(f"Resolving {idx}/{total}: {entry_name}", err=True)
            records.extend(fetch_uniprot_records((entry_name,), session))

    typer.echo(f"Resolved {len(records)} sequences.", err=True)
    _run_embedding(
        records=tuple(records),
        output_npz=output_npz,
        model_id=model_id,
        batch_size=batch_size,
        device=device,
        max_residues=max_residues,
        stride=stride,
        region_mask_json=region_mask_json,
        source_name="gpcrdb_common_coupling_map",
        source_uri=str(input_csv),
        write_audit_csv=write_audit_csv,
        write_metadata_json=write_metadata_json,
    )

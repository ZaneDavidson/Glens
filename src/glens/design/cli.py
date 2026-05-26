"""Command-line tools for the mutation design workflow."""

import csv
from enum import Enum
from pathlib import Path
from typing import Any

import typer

from glens.design.candidates import (
    ParsedMutationList,
    parse_point_mutation_csv,
    parse_point_mutation_file,
)
from glens.design.results import (
    FamilyEnsemblePredictionLike,
    build_mutation_design_results,
    mutation_results_to_rows,
    per_model_delta_rows,
)
from glens.design.selectivity import DesignObjective
from glens.design.sequences import (
    build_wt_mutant_sequence_rows,
    read_sequence_file,
    sequence_rows_to_dicts,
    sequence_rows_to_fasta,
    validate_sequence_rows,
)
from glens.inference.ensembles import (
    load_family_model_ensemble,
    predict_family_model_ensemble,
    read_family_model_specs,
)
from glens.inference.prediction_io import (
    load_embedding_npz,
    load_ensemble_prediction_npz,
)

app = typer.Typer(
    help="Design and score GPCR point-mutation candidates against changes in predicted coupling selectivity.",
    no_args_is_help=True,
)


class MutationInputFormat(str, Enum):
    """Supported mutation-list input formats."""

    AUTO = "auto"
    TXT = "txt"
    CSV = "csv"


@app.callback()
def design_main() -> None:
    """Design and score GPCR point-mutation candidates."""


@app.command("sequence-table")
def make_mutant_sequence_table(
    wt_sequence_file: Path = typer.Argument(
        ...,
        help="WT sequence file in FASTA or plain text format.",
    ),
    mutation_list: Path = typer.Argument(
        ...,
        help="Text or CSV file containing point mutations.",
    ),
    sequence_csv: Path = typer.Argument(
        ...,
        help="Output  sequence table CSV.",
    ),
    fasta_out: Path | None = typer.Option(
        None,
        "--fasta-out",
        help="Optional output FASTA.",
    ),
    wt_sequence_id: str = typer.Option(
        "WT",
        "--wt-id",
        help="Sequence id used for the WT row and mutant id prefix.",
    ),
    mutation_format: MutationInputFormat = typer.Option(
        MutationInputFormat.AUTO,
        "--mutation-format",
        help="Mutation input format: auto, txt, or csv.",
    ),
    mutation_column: str = typer.Option(
        "mutation",
        "--mutation-column",
        help="CSV column containing mutation labels when using CSV input.",
    ),
    note_column: str | None = typer.Option(
        None,
        "--note-column",
        help="Optional CSV column containing notes, to be included in output.",
    ),
    parse_errors_csv: Path | None = typer.Option(
        None,
        "--parse-errors-csv",
        help="Optional CSV for mutation parse/validation errors.",
    ),
    allow_parse_errors: bool = typer.Option(
        False,
        "--allow-parse-errors/--fail-on-parse-errors",
        help="Continue with valid mutations when some mutation rows fail to parse.",
    ),
) -> None:
    """Validate mutations against a WT sequence and write WT+mutant rows.

    The output row order is the row contract expected by downstream embedding
    and scoring:

        row 0 = WT
        rows 1..n = mutants in parsed mutation-list order
    """
    wt_sequence = read_sequence_file(wt_sequence_file)
    parsed = _load_mutation_candidates(
        mutation_list,
        mutation_format=mutation_format,
        mutation_column=mutation_column,
        note_column=note_column,
        sequence=wt_sequence,
    )
    _handle_parse_errors(
        parsed,
        parse_errors_csv=parse_errors_csv,
        allow_parse_errors=allow_parse_errors,
    )

    if not parsed.candidates:
        raise ValueError("No valid mutation candidates were parsed.")

    rows = build_wt_mutant_sequence_rows(
        wt_sequence=wt_sequence,
        candidates=parsed.candidates,
        wt_sequence_id=wt_sequence_id,
    )
    validate_sequence_rows(rows)
    _write_rows_csv(sequence_csv, sequence_rows_to_dicts(rows))

    if fasta_out is not None:
        fasta_out.parent.mkdir(parents=True, exist_ok=True)
        fasta_out.write_text(sequence_rows_to_fasta(rows), encoding="utf-8")

    typer.echo(f"WT sequence length: {len(wt_sequence)}")
    typer.echo(f"Parsed valid mutations: {len(parsed.candidates)}")
    if parsed.errors:
        typer.echo(f"Parse errors: {len(parsed.errors)}")
    typer.echo(f"Wrote sequence table: {sequence_csv}")
    if fasta_out is not None:
        typer.echo(f"Wrote FASTA: {fasta_out}")


@app.command("score-mutations")
def score_mutations(
    mutation_list: Path = typer.Argument(
        ...,
        help="Text or CSV file containing point mutations.",
    ),
    results_csv: Path = typer.Argument(
        ...,
        help="Output ranked result CSV.",
    ),
    target_family: str = typer.Option(
        ...,
        "--target-family",
        "-t",
        help="Family whose selectivity margin should increase.",
    ),
    predictions_npz: Path | None = typer.Option(
        None,
        "--predictions-npz",
        help=(
            "Precomputed ensemble predictions with WT at row 0 and mutant rows "
            "1..n. Mutually exclusive with --ensemble-manifest/--embeddings-npz."
        ),
    ),
    ensemble_manifest: Path | None = typer.Option(
        None,
        "--ensemble-manifest",
        help="JSON manifest listing saved family-model artifacts.",
    ),
    embeddings_npz: Path | None = typer.Option(
        None,
        "--embeddings-npz",
        help=(
            "Precomputed embedding NPZ with WT at row 0 and mutant rows 1..n. "
            "Requires --ensemble-manifest."
        ),
    ),
    wt_sequence_file: Path | None = typer.Option(
        None,
        "--wt-sequence",
        help=(
            "Optional WT sequence FASTA/plain text file. If provided, mutation "
            "labels are validated against WT residues before scoring."
        ),
    ),
    avoid_family: list[str] | None = typer.Option(
        None,
        "--avoid-family",
        "-a",
        help="Family whose positive delta should be penalized. Repeatable.",
    ),
    preserve_family: list[str] | None = typer.Option(
        None,
        "--preserve-family",
        "-p",
        help="Family whose absolute delta should be penalized. Repeatable.",
    ),
    mutation_format: MutationInputFormat = typer.Option(
        MutationInputFormat.AUTO,
        "--mutation-format",
        help="Mutation input format: auto, txt, or csv.",
    ),
    mutation_column: str = typer.Option(
        "mutation",
        "--mutation-column",
        help="CSV column containing mutation labels when using CSV input.",
    ),
    note_column: str | None = typer.Option(
        None,
        "--note-column",
        help="Optional CSV column containing notes, to be included in the output.",
    ),
    per_model_csv: Path | None = typer.Option(
        None,
        "--per-model-csv",
        help="Optional per-model audit CSV.",
    ),
    parse_errors_csv: Path | None = typer.Option(
        None,
        "--parse-errors-csv",
        help="Optional CSV for mutation parse/validation errors.",
    ),
    allow_parse_errors: bool = typer.Option(
        False,
        "--allow-parse-errors/--fail-on-parse-errors",
        help="Continue with valid mutations when some mutation rows fail to parse.",
    ),
    clip_predictions: bool = typer.Option(
        True,
        "--clip/--no-clip",
        help="Clip model predictions to [0, 1] when predicting from models.",
    ),
) -> None:
    """Score user-provided point mutations using precomputed inputs.

    This first version assumes the prediction/embedding batch has WT at row 0
    and mutants in rows 1..n in the same order as the parsed mutation list.
    """
    wt_sequence = (
        read_sequence_file(wt_sequence_file)
        if wt_sequence_file is not None
        else None
    )
    parsed = _load_mutation_candidates(
        mutation_list,
        mutation_format=mutation_format,
        mutation_column=mutation_column,
        note_column=note_column,
        sequence=wt_sequence,
    )
    _handle_parse_errors(
        parsed,
        parse_errors_csv=parse_errors_csv,
        allow_parse_errors=allow_parse_errors,
    )

    candidates = parsed.candidates
    if not candidates:
        raise ValueError("No valid mutation candidates were parsed.")

    prediction = _load_prediction_source(
        predictions_npz=predictions_npz,
        ensemble_manifest=ensemble_manifest,
        embeddings_npz=embeddings_npz,
        clip_predictions=clip_predictions,
    )
    _validate_prediction_rows(prediction, n_candidates=len(candidates))

    objective = DesignObjective(
        target_family=target_family,
        avoid_families=tuple(avoid_family or ()),
        preserve_families=tuple(preserve_family or ()),
    )
    results = build_mutation_design_results(
        candidates=candidates,
        prediction=prediction,
        objective=objective,
    )

    _write_rows_csv(results_csv, mutation_results_to_rows(results))
    if per_model_csv is not None:
        _write_rows_csv(per_model_csv, per_model_delta_rows(results))

    typer.echo(f"Parsed valid mutations: {len(candidates)}")
    if parsed.errors:
        typer.echo(f"Parse errors: {len(parsed.errors)}")
    typer.echo(f"Prediction rows: {prediction.weighted_mean.shape[0]}")
    typer.echo(f"Wrote ranked mutation results: {results_csv}")
    if per_model_csv is not None:
        typer.echo(f"Wrote per-model audit rows: {per_model_csv}")


def _load_mutation_candidates(
    path: Path,
    *,
    mutation_format: MutationInputFormat,
    mutation_column: str,
    note_column: str | None,
    sequence: str | None = None,
) -> ParsedMutationList:
    use_csv = mutation_format == MutationInputFormat.CSV or (
        mutation_format == MutationInputFormat.AUTO and path.suffix.lower() == ".csv"
    )

    if use_csv:
        return parse_point_mutation_csv(
            path,
            sequence=sequence,
            mutation_column=mutation_column,
            note_column=note_column,
            fail_fast=False,
        )

    return parse_point_mutation_file(
        path,
        sequence=sequence,
        fail_fast=False,
    )


def _handle_parse_errors(
    parsed: ParsedMutationList,
    *,
    parse_errors_csv: Path | None,
    allow_parse_errors: bool,
) -> None:
    if parsed.errors and parse_errors_csv is not None:
        _write_rows_csv(parse_errors_csv, [error.as_dict() for error in parsed.errors])

    if parsed.errors and not allow_parse_errors:
        parsed.raise_if_errors()


def _load_prediction_source(
    *,
    predictions_npz: Path | None,
    ensemble_manifest: Path | None,
    embeddings_npz: Path | None,
    clip_predictions: bool,
) -> FamilyEnsemblePredictionLike:
    using_precomputed_predictions = predictions_npz is not None
    using_model_prediction = ensemble_manifest is not None or embeddings_npz is not None

    if using_precomputed_predictions and using_model_prediction:
        raise ValueError(
            "Use either --predictions-npz or "
            "--ensemble-manifest with --embeddings-npz, not both."
        )

    if predictions_npz is not None:
        return load_ensemble_prediction_npz(predictions_npz)

    if ensemble_manifest is None or embeddings_npz is None:
        raise ValueError(
            "Provide either --predictions-npz or both "
            "--ensemble-manifest and --embeddings-npz."
        )

    specs = read_family_model_specs(ensemble_manifest)
    artifacts = load_family_model_ensemble(specs)
    embeddings = load_embedding_npz(embeddings_npz)
    return predict_family_model_ensemble(
        artifacts,
        embeddings,
        clip=clip_predictions,
    )


def _validate_prediction_rows(
    prediction: FamilyEnsemblePredictionLike,
    *,
    n_candidates: int,
) -> None:
    expected_rows = n_candidates + 1
    observed_rows = int(prediction.weighted_mean.shape[0])
    if observed_rows != expected_rows:
        raise ValueError(
            "Prediction/embedding batch row count does not match mutation list: "
            f"expected {expected_rows} rows (WT + {n_candidates} mutants), "
            f"got {observed_rows}."
        )


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

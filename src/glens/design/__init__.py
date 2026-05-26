"""Design-oriented scoring utilities for GPCR selectivity.

This package provides stable primitives for comparing mutant predictions to
wild-type predictions so that ensemble code can share a unit definition of selectivity deltas.
"""

from .candidates import (
    MutationCandidate,
    MutationParseError,
    ParsedMutationList,
    build_single_mutation_scan_candidates,
    candidates_to_rows,
    parse_point_mutation_csv,
    parse_point_mutation_file,
    parse_point_mutation_lines,
    parse_point_mutation_text,
)
from .mutations import (
    CANONICAL_AMINO_ACIDS,
    PointMutation,
    apply_point_mutation,
    generate_single_mutants,
    mutation_label,
    parse_point_mutation,
    positions_from_boolean_mask,
    region_labels_by_position,
    selected_positions_from_region_masks,
)
from .results import (
    FamilyEnsemblePredictionLike,
    MutationDesignResult,
    build_mutation_design_result,
    build_mutation_design_results,
    mutation_results_to_rows,
    per_model_delta_rows,
    rank_mutation_design_results,
)
from .selectivity import (
    DEFAULT_FAMILY_NAMES,
    DesignObjective,
    EnsembleDeltaSummary,
    SelectivityDelta,
    family_index,
    score_selectivity_delta,
    selectivity_margin,
    summarize_ensemble_delta,
    top_family,
)

__all__ = [
    "CANONICAL_AMINO_ACIDS",
    "DEFAULT_FAMILY_NAMES",
    "DesignObjective",
    "EnsembleDeltaSummary",
    "FamilyEnsemblePredictionLike",
    "MutationCandidate",
    "MutationDesignResult",
    "MutationParseError",
    "ParsedMutationList",
    "PointMutation",
    "SelectivityDelta",
    "apply_point_mutation",
    "build_mutation_design_result",
    "build_mutation_design_results",
    "build_single_mutation_scan_candidates",
    "candidates_to_rows",
    "family_index",
    "generate_single_mutants",
    "mutation_label",
    "mutation_results_to_rows",
    "parse_point_mutation",
    "parse_point_mutation_csv",
    "parse_point_mutation_file",
    "parse_point_mutation_lines",
    "parse_point_mutation_text",
    "per_model_delta_rows",
    "positions_from_boolean_mask",
    "rank_mutation_design_results",
    "region_labels_by_position",
    "score_selectivity_delta",
    "selected_positions_from_region_masks",
    "selectivity_margin",
    "summarize_ensemble_delta",
    "top_family",
]

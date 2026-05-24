"""Design-oriented scoring utilities for GPCR selectivity.

This package provides stable primitives for comparing mutant predictions to
wild-type predictions so that ensemble code can share a unit definition of selectivity deltas.
"""

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
    "DEFAULT_FAMILY_NAMES",
    "DesignObjective",
    "EnsembleDeltaSummary",
    "SelectivityDelta",
    "family_index",
    "score_selectivity_delta",
    "selectivity_margin",
    "summarize_ensemble_delta",
    "top_family",
]

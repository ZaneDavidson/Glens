"""Source-agnostic sequence embedding utilities."""

from glens.embeddings.model import (
    DEFAULT_MAX_RESIDUES,
    DEFAULT_MODEL_ID,
    ResidueEmbeddingResult,
    embed_residue_sequences,
    embed_sequences,
    load_plm,
    resolve_device,
)
from glens.embeddings.views import EmbeddingViews

__all__ = [
    "DEFAULT_MAX_RESIDUES",
    "DEFAULT_MODEL_ID",
    "EmbeddingViews",
    "ResidueEmbeddingResult",
    "embed_residue_sequences",
    "embed_sequences",
    "load_plm",
    "resolve_device",
]

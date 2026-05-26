"""Lightweight data-ingestion utilities for Glens.

This package intentionally contains source parsing / sequence resolution only.
Embedding, model fitting, and design scoring live in their own packages.
"""

from glens.data.sequences import SequenceRecord, read_sequence_table_csv

__all__ = ["SequenceRecord", "read_sequence_table_csv"]

from __future__ import annotations

from pathlib import Path

import pytest

from glens.data.gpcrdb import entry_name_from_gpcrdb_value, iter_gpcrdb_entry_names
from glens.data.sequences import SequenceRecord, read_sequence_table_csv


def test_read_sequence_table_preserves_row_order_and_default_gpcrdb(tmp_path: Path) -> None:
    path = tmp_path / "seqs.csv"
    path.write_text(
        "sequence_id,sequence,note\n"
        "WT, AC D ,wild type\n"
        "WT|A2G, AGD ,mutant\n",
        encoding="utf-8",
    )

    records = read_sequence_table_csv(path, default_gpcrdb_entry_name="opn4_human")

    assert [row.sequence_id for row in records] == ["WT", "WT|A2G"]
    assert [row.sequence for row in records] == ["ACD", "AGD"]
    assert all(row.gpcrdb_entry_name == "opn4_human" for row in records)
    assert records[1].metadata["note"] == "mutant"


def test_entry_name_from_gpcrdb_url_or_bare_value() -> None:
    assert entry_name_from_gpcrdb_value("https://gproteindb.org/protein/opn4_human") == "opn4_human"
    assert entry_name_from_gpcrdb_value("OPN4_HUMAN") == "opn4_human"
    assert entry_name_from_gpcrdb_value("not a receptor") is None


def test_iter_gpcrdb_entry_names_handles_two_row_header(tmp_path: Path) -> None:
    path = tmp_path / "map.csv"
    path.write_text(
        ",Receptor,Receptor\n"
        ",Uniprot,GPCRdb\n"
        ",OPN4,https://gproteindb.org/protein/opn4_human\n"
        ",OPN4,https://gproteindb.org/protein/opn4_human\n"
        ",BAD,not-a-valid-entry\n",
        encoding="utf-8",
    )

    assert list(iter_gpcrdb_entry_names(path)) == ["opn4_human"]


def test_read_sequence_table_rejects_missing_sequence_column(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("sequence_id,not_sequence\nA,ACD\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing column"):
        read_sequence_table_csv(path)

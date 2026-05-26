from pathlib import Path

import numpy as np
import pytest

from glens.design.candidates import (
    build_single_mutation_scan_candidates,
    candidates_to_rows,
    parse_point_mutation_csv,
    parse_point_mutation_lines,
    parse_point_mutation_text,
)


def test_parse_point_mutation_lines_supports_user_uploaded_list() -> None:
    sequence = "ACDRFG"
    parsed = parse_point_mutation_lines(
        [
            "# suspected beneficial mutations",
            "D3W # candidate from paper",
            "",
            "R4A",
        ],
        sequence=sequence,
    )

    assert parsed.ok
    assert [candidate.label for candidate in parsed.candidates] == ["D3W", "R4A"]
    assert parsed.candidates[0].note == "candidate from paper"


def test_parse_point_mutation_lines_collects_errors_without_losing_valid_rows() -> None:
    parsed = parse_point_mutation_text(
        """
        A1C
        bad
        C2D
        """,
        sequence="ACD",
    )

    assert not parsed.ok
    assert [candidate.label for candidate in parsed.candidates] == ["A1C", "C2D"]
    assert len(parsed.errors) == 1
    assert parsed.errors[0].line_number == 3
    assert "Invalid mutation label" in parsed.errors[0].message


def test_parse_point_mutation_lines_fail_fast() -> None:
    with pytest.raises(ValueError, match="line 1"):
        parse_point_mutation_lines(["bad"], fail_fast=True)


def test_parse_point_mutation_csv_reads_mutation_and_note_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "mutations.csv"
    csv_path.write_text(
        "mutation,note\n"
        "A1C,first\n"
        "C2D,second\n",
        encoding="utf-8",
    )

    parsed = parse_point_mutation_csv(
        csv_path,
        sequence="ACD",
        note_column="note",
    )

    assert parsed.ok
    assert [candidate.label for candidate in parsed.candidates] == ["A1C", "C2D"]
    assert [candidate.note for candidate in parsed.candidates] == ["first", "second"]


def test_parse_point_mutation_csv_reports_missing_mutation_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("variant\nA1C\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing mutation column"):
        parse_point_mutation_csv(csv_path)


def test_build_single_mutation_scan_candidates_scans_regions() -> None:
    sequence = "ACD"
    masks = {
        "tm3": np.array([True, False, False]),
        "icl2": np.array([False, False, True]),
    }

    candidates = build_single_mutation_scan_candidates(
        sequence,
        region_masks=masks,
        selected_regions=("tm3",),
        amino_acids=("A", "C", "D"),
    )

    assert [candidate.label for candidate in candidates] == ["A1C", "A1D"]
    assert [candidate.mutation.region for candidate in candidates] == ["tm3", "tm3"]
    assert all(candidate.source == "single_mutant_scan" for candidate in candidates)


def test_build_single_mutation_scan_candidates_intersects_positions_and_regions() -> None:
    sequence = "ACD"
    masks = {
        "tm3": np.array([True, False, False]),
        "icl2": np.array([False, False, True]),
    }

    candidates = build_single_mutation_scan_candidates(
        sequence,
        positions=[0, 2],
        region_masks=masks,
        selected_regions=("icl2",),
        amino_acids=("A", "C", "D"),
    )

    assert [candidate.label for candidate in candidates] == ["D3A", "D3C"]


def test_candidates_to_rows_returns_flat_records() -> None:
    parsed = parse_point_mutation_text("A1C", sequence="ACD")
    rows = candidates_to_rows(parsed.candidates)

    assert rows == [
        {
            "mutation": "A1C",
            "source": "user_list",
            "note": "",
            "sequence_index": 0,
            "position": 1,
            "wt_aa": "A",
            "mutant_aa": "C",
            "region": "",
        }
    ]

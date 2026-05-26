"""UniProt sequence resolution helpers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import requests

from glens.data.sequences import SequenceRecord

UNIPROT_ENTRY_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{entry_id}.fasta"
UNIPROT_ENTRY_JSON_URL = "https://rest.uniprot.org/uniprotkb/{entry_id}.json"


def fetch_uniprot_records(
    entry_names: Iterable[str],
    session: requests.Session,
    *,
    timeout: float = 30.0,
) -> Iterator[SequenceRecord]:
    """Resolve entry names to UniProt-backed ``SequenceRecord`` rows."""
    for entry_name in entry_names:
        yield fetch_uniprot_record(entry_name, session, timeout=timeout)


def fetch_uniprot_record(
    entry_name: str,
    session: requests.Session,
    *,
    timeout: float = 30.0,
) -> SequenceRecord:
    """Resolve one GPCRdb/UniProt-style entry name to a protein sequence.

    Examples of supported input include ``opn4_human`` and ``adrb2_human``.
    The JSON endpoint is tried first because it provides accession and UniProt
    ID metadata. FASTA is used only as a sequence fallback.
    """
    normalized_entry = entry_name.strip().lower()
    entry_id = normalized_entry.upper()

    response = session.get(
        UNIPROT_ENTRY_JSON_URL.format(entry_id=entry_id),
        timeout=timeout,
    )
    response.raise_for_status()
    record = response.json()

    accession = record.get("primaryAccession")
    uniprot_id = record.get("uniProtkbId", entry_id)
    sequence = record.get("sequence", {}).get("value")

    if not accession:
        raise LookupError(
            f"UniProt record for {normalized_entry!r} did not include an accession."
        )

    if not sequence:
        fasta_response = session.get(
            UNIPROT_ENTRY_FASTA_URL.format(entry_id=accession),
            timeout=timeout,
        )
        fasta_response.raise_for_status()
        sequence = parse_fasta_sequence(fasta_response.text)

    if not sequence:
        raise LookupError(
            f"UniProt record for {normalized_entry!r} resolved to "
            f"{uniprot_id!r} / {accession!r}, but no sequence was found."
        )

    organism = record.get("organism", {}) if isinstance(record.get("organism"), dict) else {}
    taxon_id = organism.get("taxonId")
    scientific_name = organism.get("scientificName")

    metadata = {}
    if taxon_id is not None:
        metadata["taxon_id"] = str(taxon_id)
    if scientific_name:
        metadata["organism"] = str(scientific_name)

    return SequenceRecord(
        sequence_id=normalized_entry,
        sequence=str(sequence).strip().upper(),
        source="uniprot",
        gpcrdb_entry_name=normalized_entry,
        uniprot_accession=str(accession),
        uniprot_id=str(uniprot_id),
        metadata=metadata,
    )


def parse_fasta_sequence(text: str) -> str:
    """Return the sequence body from a FASTA string."""
    return "".join(
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.startswith(">")
    ).upper()

"""
Core ESM-2 embedding and lightweight regression utilities.
"""
from collections.abc import Iterable, Iterator
from typing import Any, NamedTuple

import numpy as np
import torch
from dataclasses import dataclass

DEFAULT_MODEL_ID = "facebook/esm2_t33_650M_UR50D"
DEFAULT_MAX_RESIDUES = 1022


class TokenIds(NamedTuple):
    cls_id: int | None
    eos_id: int | None
    pad_id: int | None


class SequenceWindow(NamedTuple):
    sequence_index: int
    start: int
    end: int
    sequence: str


@dataclass(frozen=True)
class ResidueEmbeddingResult:
    sequence: str
    residue_embeddings: np.ndarray  # shape: (L, D)
    coverage: np.ndarray            # shape: (L,)


def resolve_device(device: str = "auto") -> torch.device:
    """
    Resolve available hardware acceleration into a torch device.
    """
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_plm(
    model_id: str = DEFAULT_MODEL_ID,
    device: str | torch.device = "auto",
) -> tuple[Any, torch.nn.Module, torch.device]:
    """
    Load tokenizer/model pair in eval mode.
    """
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "Missing dependency! The 'transformers' package is required for ESM-2 embeddings."
        ) from err

    torch_device = resolve_device(device) if isinstance(device, str) else device
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModel.from_pretrained(model_id)
    model.eval()
    model.to(torch_device)
    return tokenizer, model, torch_device


def _token_ids(tokenizer: Any) -> TokenIds:
    return TokenIds(
        cls_id=getattr(tokenizer, "cls_token_id", None),
        eos_id=getattr(tokenizer, "eos_token_id", None),
        pad_id=getattr(tokenizer, "pad_token_id", None),
    )


def _valid_residue_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_ids: TokenIds,
) -> torch.Tensor:
    valid = attention_mask.to(torch.bool)

    if token_ids.cls_id is not None:
        valid &= input_ids.ne(token_ids.cls_id)
    if token_ids.eos_id is not None:
        valid &= input_ids.ne(token_ids.eos_id)
    if token_ids.pad_id is not None:
        valid &= input_ids.ne(token_ids.pad_id)

    return valid


def _residue_token_arrays(
    last_hidden: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_ids: TokenIds,
) -> list[np.ndarray]:
    valid = _valid_residue_mask(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_ids=token_ids,
    )

    arrays: list[np.ndarray] = []

    for row_idx in range(last_hidden.shape[0]):
        residue_reps = last_hidden[row_idx, valid[row_idx]]
        arrays.append(residue_reps.detach().cpu().float().numpy())

    return arrays


def _batched(items: Iterable[Any], batch_size: int) -> Iterator[list[Any]]:
    batch: list[str] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _clean_sequence(sequence: str) -> str:
    return "".join(str(sequence).split()).upper()

def _iter_windows(
    sequence: str,
    sequence_index: int,
    max_residues: int,
    stride: int | None = None,
) -> Iterator[SequenceWindow]:
    if max_residues < 1:
        raise ValueError("max_residues must be >= 1.")

    if stride is None:
        stride = max_residues

    if stride < 1:
        raise ValueError("stride must be >= 1.")

    if stride > max_residues:
        raise ValueError("stride must be <= max_residues to avoid coverage gaps.")

    length = len(sequence)

    if length <= max_residues:
        yield SequenceWindow(
            sequence_index=sequence_index,
            start=0,
            end=length,
            sequence=sequence,
        )
        return

    # non-overlap mode.
    if stride == max_residues:
        for start in range(0, length, max_residues):
            end = min(start + max_residues, length)
            yield SequenceWindow(
                sequence_index=sequence_index,
                start=start,
                end=end,
                sequence=sequence[start:end],
            )
        return

    # Overlap mode with final backfill so C-terminal end is covered.
    starts = list(range(0, max(1, length - max_residues + 1), stride))
    final_start = max(0, length - max_residues)

    if starts[-1] != final_start:
        starts.append(final_start)

    for start in starts:
        end = min(start + max_residues, length)
        yield SequenceWindow(
            sequence_index=sequence_index,
            start=start,
            end=end,
            sequence=sequence[start:end],
        )


def embed_residue_sequences(
    sequences: str | Iterable[str],
    tokenizer: Any,
    model: torch.nn.Module,
    *,
    batch_size: int = 8,
    max_residues: int = DEFAULT_MAX_RESIDUES,
    stride: int | None = None,
    layer: int = -1,
    device: str | torch.device | None = None,
    mixed_precision: bool = True,
) -> ResidueEmbeddingResult | list[ResidueEmbeddingResult]:
    """
    Reconstruct a residue-wise ESM embedding array per sequence.

    If stride is None, chunk windows are non-overlapping. If stride < max_residues, 
    overlapping windows are averaged at the residue level to mitigate boundary artifacts.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1!")

    single_input = isinstance(sequences, str)
    sequence_iterable: Iterable[str] = [sequences] if single_input else sequences

    torch_device = (
        next(model.parameters()).device
        if device is None
        else resolve_device(device)
        if isinstance(device, str)
        else device
    )
    model.to(torch_device)

    token_ids = _token_ids(tokenizer)
    use_amp = mixed_precision and torch_device.type == "cuda"
    results: list[ResidueEmbeddingResult] = []

    for raw_batch in _batched(sequence_iterable, batch_size):
        seqs = [_clean_sequence(seq) for seq in raw_batch]

        empty = [idx for idx, seq in enumerate(seqs) if not seq]
        if empty:
            raise ValueError(f"Empty sequence at batch positions: {empty}")

        windows = [
            window
            for seq_idx, seq in enumerate(seqs)
            for window in _iter_windows(
                sequence=seq,
                sequence_index=seq_idx,
                max_residues=max_residues,
                stride=stride,
            )
        ]

        residue_sums: list[np.ndarray | None] = [None for _ in seqs]
        coverage = [
            np.zeros(len(seq), dtype=np.float32)
            for seq in seqs
        ]

        for window_batch in _batched(windows, batch_size):
            batch = tokenizer(
                [window.sequence for window in window_batch],
                add_special_tokens=True,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            batch = {key: value.to(torch_device) for key, value in batch.items()}

            with torch.inference_mode(), torch.autocast(
                device_type=torch_device.type,
                enabled=use_amp,
            ):
                out = model(**batch, output_hidden_states=layer != -1)

            if layer == -1:
                token_reps = out.last_hidden_state
            else:
                if out.hidden_states is None:
                    raise RuntimeError("Model did not return hidden states.")
                try:
                    token_reps = out.hidden_states[layer]
                except IndexError as err:
                    raise ValueError(f"Invalid layer index: {layer}") from err

            residue_arrays = _residue_token_arrays(
                last_hidden=token_reps,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_ids=token_ids,
            )

            for window, residue_array in zip(window_batch, residue_arrays, strict=True):
                expected_len = window.end - window.start
                if residue_array.shape[0] != expected_len:
                    raise RuntimeError(
                        "Residue/token length mismatch for window "
                        f"{window.start}:{window.end}: expected {expected_len}, "
                        f"got {residue_array.shape[0]}."
                    )

                seq_idx = window.sequence_index

                if residue_sums[seq_idx] is None:
                    residue_sums[seq_idx] = np.zeros(
                        (len(seqs[seq_idx]), residue_array.shape[1]),
                        dtype=np.float32,
                    )

                residue_sums[seq_idx][window.start:window.end] += residue_array
                coverage[seq_idx][window.start:window.end] += 1.0

        for seq_idx, seq in enumerate(seqs):
            sums = residue_sums[seq_idx]
            counts = coverage[seq_idx]

            if sums is None:
                raise RuntimeError(f"No residue embeddings were produced for sequence {seq_idx}.")

            if np.any(counts <= 0):
                missing = np.flatnonzero(counts <= 0)
                raise RuntimeError(
                    f"Residue coverage has gaps for sequence {seq_idx}; "
                    f"first missing positions: {missing[:10].tolist()}"
                )

            residue_embeddings = sums / counts[:, None]

            results.append(
                ResidueEmbeddingResult(
                    sequence=seq,
                    residue_embeddings=residue_embeddings.astype(np.float32, copy=False),
                    coverage=counts.astype(np.float32, copy=False),
                )
            )

    return results[0] if single_input else results

# Can probably fold this in later
def embed_sequences(
    sequences: str | Iterable[str],
    tokenizer: Any,
    model: torch.nn.Module,
    *,
    batch_size: int = 8,
    max_residues: int = DEFAULT_MAX_RESIDUES,
    stride: int | None = None,
    layer: int = -1,
    device: str | torch.device | None = None,
    mixed_precision: bool = True,
) -> np.ndarray:
    """
    Global mean embedding.
    Function mean-pools reconstructed residue embeddings, for when single X matrix expected.
    """
    single_input = isinstance(sequences, str)

    residue_results = embed_residue_sequences(
        sequences,
        tokenizer,
        model,
        batch_size=batch_size,
        max_residues=max_residues,
        stride=stride,
        layer=layer,
        device=device,
        mixed_precision=mixed_precision,
    )

    if single_input:
        assert isinstance(residue_results, ResidueEmbeddingResult)
        return residue_results.residue_embeddings.mean(axis=0)

    assert isinstance(residue_results, list)

    if not residue_results:
        return np.empty((0, 0), dtype=np.float32)

    return np.vstack([
        result.residue_embeddings.mean(axis=0)
        for result in residue_results
    ]).astype(np.float32, copy=False)
"""
Core ESM-2 embedding and lightweight regression utilities.
"""
from collections.abc import Iterable, Iterator
from typing import Any, NamedTuple

import numpy as np
import torch
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

DEFAULT_MODEL_ID = "facebook/esm2_t33_650M_UR50D"
DEFAULT_MAX_RESIDUES = 1022


class TokenIds(NamedTuple):
    cls_id: int | None
    eos_id: int | None
    pad_id: int | None


class SequenceChunk(NamedTuple):
    sequence_index: int
    sequence: str
    length: int


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


def _mean_pool(
    last_hidden: torch.Tensor,
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

    mask = valid.unsqueeze(-1)
    sums = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return sums / counts


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

def _iter_chunks(
    sequence: str,
    sequence_index: int,
    max_residues: int,
) -> Iterator[SequenceChunk]:
    for start in range(0, len(sequence), max_residues):
        chunk = sequence[start : start + max_residues]
        yield SequenceChunk(
            sequence_index=sequence_index,
            sequence=chunk,
            length=len(chunk),
        )


def embed_sequences(
    sequences: str | Iterable[str],
    tokenizer: Any,
    model: torch.nn.Module,
    *,
    batch_size: int = 8,
    max_residues: int = DEFAULT_MAX_RESIDUES,
    layer: int = -1,
    device: str | torch.device | None = None,
    mixed_precision: bool = True,
) -> np.ndarray:
    """
    Embed a sequence or an iterable of sequences.
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

    # use this if I add a truncation strategy later:
    # special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
    token_ids = _token_ids(tokenizer)
    use_amp = mixed_precision and torch_device.type == "cuda"
    embeddings: list[np.ndarray] = []

    for raw_batch in _batched(sequence_iterable, batch_size):
        seqs = [_clean_sequence(seq) for seq in raw_batch]
        empty = [idx for idx, seq in enumerate(seqs) if not seq]
        if empty:
            raise ValueError(f"Empty sequence at batch positions: {empty}")

        chunks = [
            chunk
            for seq_idx, seq in enumerate(seqs)
            for chunk in _iter_chunks(seq, seq_idx, max_residues)
        ]

        residue_counts = np.array([len(seq) for seq in seqs], dtype=np.float32)
        weighted_sums: np.ndarray | None = None

        for chunk_batch in _batched(chunks, batch_size):
            batch = tokenizer(
                [chunk.sequence for chunk in chunk_batch],
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

            pooled = _mean_pool(
                last_hidden=token_reps,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_ids=token_ids,
            )

            arr = pooled.detach().cpu().float().numpy()

            if weighted_sums is None:
                weighted_sums = np.zeros((len(seqs), arr.shape[1]), dtype=np.float32)

            for chunk, chunk_embedding in zip(chunk_batch, arr, strict=True):
                weighted_sums[chunk.sequence_index] += chunk_embedding * chunk.length

        if weighted_sums is None:
            continue

        batch_embeddings = weighted_sums / residue_counts[:, None]
        embeddings.extend(batch_embeddings)

    if not embeddings:
        return (
            np.empty((0,), dtype=np.float32)
            if single_input
            else np.empty((0, 0), dtype=np.float32)
        )

    stacked = np.vstack(embeddings)
    return stacked[0] if single_input else stacked


class RegularizedRegression:
    """
    Regularize embeddings, then fit ElasticNet to continuous targets.
    """
    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5) -> None:
        self.scaler = StandardScaler()
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
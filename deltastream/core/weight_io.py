"""
weight_io.py — Unified HuggingFace model weight loader.

Supports:
  - Single safetensors file   (model.safetensors)
  - Sharded safetensors       (model-00001-of-00005.safetensors + index.json)
  - Single PyTorch bin        (pytorch_model.bin)
  - Sharded PyTorch bin       (pytorch_model-00001-of-00005.bin + index.json)
  - Remote HuggingFace repos  (auto-downloaded via huggingface_hub)

All tensors returned on CPU. Tied weights are detected and noted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file as st_load_file

from deltastream.utils.logging import log_info, log_step, log_warning

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def load_model_weights(
    model_path: str,
    *,
    cache_dir: str | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """
    Load all model weights from a HuggingFace model (local or remote).

    Returns
    -------
    weights : dict[str, torch.Tensor]
        Mapping tensor_name → CPU tensor (original dtype preserved).
    metadata : dict
        Source info: format, num_shards, tied_weight_pairs, local_dir.
    """
    local_dir = _resolve_local_dir(model_path, cache_dir=cache_dir)
    log_step("Loading weights", f"from {local_dir}")

    fmt, index_path = _detect_format(local_dir)
    log_info(f"Detected format: [highlight]{fmt}[/highlight]")

    if fmt == "safetensors_sharded":
        weights, num_shards = _load_sharded_safetensors(local_dir, index_path)
    elif fmt == "safetensors_single":
        weights, num_shards = _load_single_safetensors(local_dir)
    elif fmt == "bin_sharded":
        weights, num_shards = _load_sharded_bin(local_dir, index_path)
    elif fmt == "bin_single":
        weights, num_shards = _load_single_bin(local_dir)
    else:
        raise ValueError(f"Cannot detect weight format in: {local_dir}")

    tied_pairs = _detect_tied_weights(weights)
    if tied_pairs:
        log_warning(f"Detected {len(tied_pairs)} tied weight pair(s): {tied_pairs}")

    log_info(
        f"Loaded [highlight]{len(weights)}[/highlight] tensors "
        f"from [highlight]{num_shards}[/highlight] shard(s)"
    )

    metadata: dict[str, Any] = {
        "format": fmt,
        "num_shards": num_shards,
        "tied_weight_pairs": tied_pairs,
        "local_dir": str(local_dir),
        "source": model_path,
    }
    return weights, metadata


# ──────────────────────────────────────────────────────────────────────────────
# Format detection
# ──────────────────────────────────────────────────────────────────────────────


def _detect_format(local_dir: Path) -> tuple[str, Path | None]:
    """Return (format_name, index_json_path_or_None)."""
    st_index = local_dir / "model.safetensors.index.json"
    bin_index = local_dir / "pytorch_model.bin.index.json"
    single_st = local_dir / "model.safetensors"

    if st_index.exists():
        return "safetensors_sharded", st_index
    if single_st.exists():
        return "safetensors_single", None
    if bin_index.exists():
        return "bin_sharded", bin_index

    # Fallback: any .bin file
    bin_files = sorted(local_dir.glob("*.bin"))
    if bin_files:
        return "bin_single", None

    # Try any .safetensors
    st_files = sorted(local_dir.glob("*.safetensors"))
    if st_files:
        return "safetensors_single", None

    return "unknown", None


# ──────────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────────


def _load_single_safetensors(
    local_dir: Path,
) -> tuple[dict[str, torch.Tensor], int]:
    st_files = sorted(local_dir.glob("*.safetensors"))
    weights: dict[str, torch.Tensor] = {}
    for f in st_files:
        chunk = st_load_file(str(f), device="cpu")
        weights.update(chunk)
    return weights, len(st_files)


def _load_sharded_safetensors(
    local_dir: Path,
    index_path: Path,
) -> tuple[dict[str, torch.Tensor], int]:
    with open(index_path, "r", encoding="utf-8") as fh:
        index = json.load(fh)

    weight_map: dict[str, str] = index["weight_map"]
    # Unique shard filenames in order
    shard_files = sorted(set(weight_map.values()))
    weights: dict[str, torch.Tensor] = {}
    for shard_name in shard_files:
        shard_path = local_dir / shard_name
        chunk = st_load_file(str(shard_path), device="cpu")
        weights.update(chunk)
    return weights, len(shard_files)


def _load_single_bin(
    local_dir: Path,
) -> tuple[dict[str, torch.Tensor], int]:
    bin_files = sorted(local_dir.glob("*.bin"))
    weights: dict[str, torch.Tensor] = {}
    for f in bin_files:
        chunk: dict[str, torch.Tensor] = torch.load(
            str(f), map_location="cpu", weights_only=True
        )
        weights.update(chunk)
    return weights, len(bin_files)


def _load_sharded_bin(
    local_dir: Path,
    index_path: Path,
) -> tuple[dict[str, torch.Tensor], int]:
    with open(index_path, "r", encoding="utf-8") as fh:
        index = json.load(fh)

    weight_map: dict[str, str] = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))
    weights: dict[str, torch.Tensor] = {}
    for shard_name in shard_files:
        shard_path = local_dir / shard_name
        chunk: dict[str, torch.Tensor] = torch.load(
            str(shard_path), map_location="cpu", weights_only=True
        )
        weights.update(chunk)
    return weights, len(shard_files)


# ──────────────────────────────────────────────────────────────────────────────
# Tied weight detection
# ──────────────────────────────────────────────────────────────────────────────


def _detect_tied_weights(
    weights: dict[str, torch.Tensor],
) -> list[tuple[str, str]]:
    """
    Detect tensor pairs that share the same storage (tied weights).
    Returns list of (key_a, key_b) tuples where key_a < key_b alphabetically.
    """
    seen: dict[int, str] = {}  # data_ptr → first key seen
    tied: list[tuple[str, str]] = []
    for key, tensor in weights.items():
        ptr = tensor.data_ptr()
        if ptr in seen:
            tied.append((seen[ptr], key))
        else:
            seen[ptr] = key
    return tied


# ──────────────────────────────────────────────────────────────────────────────
# Remote → local resolution
# ──────────────────────────────────────────────────────────────────────────────


def _resolve_local_dir(
    model_path: str,
    *,
    cache_dir: str | None = None,
) -> Path:
    """
    If model_path is a local directory, return it directly.
    Otherwise treat it as a HuggingFace repo ID and download via snapshot.
    """
    local = Path(model_path)
    if local.exists() and local.is_dir():
        return local

    # Remote repo — requires huggingface_hub
    from huggingface_hub import snapshot_download  # lazy import

    log_info(f"Downloading model [highlight]{model_path}[/highlight] from HuggingFace Hub …")
    downloaded = snapshot_download(
        repo_id=model_path,
        cache_dir=cache_dir,
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )
    return Path(downloaded)


# ──────────────────────────────────────────────────────────────────────────────
# Per-layer tensor dict helpers (used by encoder/decoder)
# ──────────────────────────────────────────────────────────────────────────────


def tensors_for_layer(
    weights: dict[str, torch.Tensor],
    prefix: str,
    layer_idx: int,
) -> dict[str, torch.Tensor]:
    """
    Extract tensors belonging to a single transformer layer.

    Returns a dict with *short* keys (prefix and layer index stripped), e.g.
      "self_attn.q_proj.weight"  instead of "model.layers.0.self_attn.q_proj.weight"
    """
    full_prefix = f"{prefix}.{layer_idx}."
    result: dict[str, torch.Tensor] = {}
    for key, tensor in weights.items():
        if key.startswith(full_prefix):
            short_key = key[len(full_prefix):]
            result[short_key] = tensor
    return result


def non_layer_tensors(
    weights: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    """Return all tensors whose key does NOT start with the given layer prefix."""
    return {k: v for k, v in weights.items() if not k.startswith(f"{prefix}.")}

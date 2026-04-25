"""
delta_decoder.py — Reconstruct original weights from base + delta files.

Reconstruction algorithm
------------------------
W(0)  = load verbatim from base/layer_00.safetensors
W(N)  = W(N-1).float() + delta_N.float()  →  cast back to original dtype

All arithmetic in float32 (same rule as encoder) to guarantee exact inversion.
Non-layer tensors loaded verbatim from base/non_layer.safetensors.

This module is the Phase 2 hook point:
  Phase 2 will wrap `reconstruct_layer()` with an LRU cache keyed by layer_idx.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterator

import torch
from safetensors.torch import load_file as st_load_file
from safetensors.torch import load as st_load_bytes

from deltastream.core.manifest import DeltaManifest, read_manifest
from deltastream.utils.logging import log_info, log_step, log_warning, make_progress

# ──────────────────────────────────────────────────────────────────────────────
# Decompression helper
# ──────────────────────────────────────────────────────────────────────────────


def _load_with_decompression(
    path: Path,
    compression: str,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """
    Load a safetensors file, transparently decompressing if needed.

    Parameters
    ----------
    path        : path to the .safetensors (or zstd-compressed .safetensors) file
    compression : manifest compression string, e.g. 'none', 'zstd:1'
    device      : target device
    """
    if compression == "none" or not compression.startswith("zstd"):
        return st_load_file(str(path), device=device)

    try:
        import zstandard as zstd
    except ImportError:
        raise ImportError("zstandard required for compressed delta loading: pip install zstandard")

    raw_compressed = path.read_bytes()
    dctx = zstd.ZstdDecompressor()
    raw_bytes = dctx.decompress(raw_compressed)
    tensors = st_load_bytes(raw_bytes)
    if device != "cpu":
        return {k: v.to(device) for k, v in tensors.items()}
    return tensors



# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def reconstruct_all_layers(
    delta_model_dir: Path | str,
    *,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """
    Full reconstruction: returns a flat weight dict identical to the original model.

    Parameters
    ----------
    delta_model_dir : path to the delta_model/ directory
    device          : target device for tensors ('cpu' recommended for verification)

    Returns
    -------
    weights : dict {full_tensor_name: tensor}
              bit-identical to the weight dict that was encoded.
    """
    delta_model_dir = Path(delta_model_dir)
    manifest = read_manifest(delta_model_dir)

    log_step(
        "Reconstructing weights",
        f"from {delta_model_dir}  "
        f"({manifest.num_layers} layers, source: {manifest.source_model})",
    )

    # ── 1. Non-layer tensors (verbatim) ──────────────────────────────────────
    non_layer = _load_non_layer(delta_model_dir, device=device)

    # ── 2. Layer-by-layer reconstruction ─────────────────────────────────────
    layer_weights: dict[str, torch.Tensor] = {}

    with make_progress() as progress:
        task = progress.add_task(
            "Reconstructing layers", total=manifest.num_layers
        )

        prev_layer: dict[str, torch.Tensor] | None = None
        for layer_idx, layer_tensors in _iter_layers(
            delta_model_dir, manifest, device=device
        ):
            # Expand short keys → full keys
            prefix = manifest.layer_prefix
            for short_key, tensor in layer_tensors.items():
                full_key = f"{prefix}.{layer_idx}.{short_key}"
                layer_weights[full_key] = tensor

            progress.advance(task)

    # ── 3. Merge everything ───────────────────────────────────────────────────
    reconstructed: dict[str, torch.Tensor] = {}
    reconstructed.update(non_layer)
    reconstructed.update(layer_weights)

    log_info(
        f"Reconstruction complete: [highlight]{len(reconstructed)}[/highlight] tensors"
    )
    return reconstructed


def reconstruct_layer(
    delta_model_dir: Path | str,
    layer_idx: int,
    manifest: DeltaManifest | None = None,
    *,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """
    Reconstruct a SINGLE layer's tensors (short-key dict).

    Phase 2 hook: this function signature is stable across phases.
    The LRU cache in Phase 2 will wrap this call.

    Parameters
    ----------
    delta_model_dir : path to delta_model/
    layer_idx       : 0-based transformer layer index
    manifest        : optional pre-loaded manifest (avoids repeated disk reads)
    device          : 'cpu' or 'cuda'

    Returns
    -------
    dict {short_tensor_key: tensor}
    """
    delta_model_dir = Path(delta_model_dir)
    if manifest is None:
        manifest = read_manifest(delta_model_dir)

    # Reconstruct from scratch up to the requested layer (naive; Phase 2 caches)
    prev: dict[str, torch.Tensor] | None = None
    for idx, layer_tensors in _iter_layers(
        delta_model_dir, manifest, device=device, stop_at=layer_idx
    ):
        prev = layer_tensors

    if prev is None:
        raise ValueError(f"Layer {layer_idx} not found in delta_model at {delta_model_dir}")
    return prev


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _iter_layers(
    delta_model_dir: Path,
    manifest: DeltaManifest,
    *,
    device: str = "cpu",
    stop_at: int | None = None,
) -> Iterator[tuple[int, dict[str, torch.Tensor]]]:
    """
    Yield (layer_idx, short_key_tensor_dict) for layers 0 … num_layers-1.
    Reconstruction is sequential; each step depends only on the previous layer.
    """
    # Layer 0 — verbatim base (never compressed)
    base_path = delta_model_dir / "base" / "layer_00.safetensors"
    prev = st_load_file(str(base_path), device=device)
    yield 0, dict(prev)

    if stop_at == 0:
        return

    # Layers 1 … N — apply deltas cumulatively
    compression = manifest.compression  # e.g. 'none' or 'zstd:1'
    for layer_idx in range(1, manifest.num_layers):
        delta_path = (
            delta_model_dir / "deltas" / f"layer_{layer_idx:02d}.delta.safetensors"
        )
        if not delta_path.exists():
            raise FileNotFoundError(
                f"Delta file missing: {delta_path}\n"
                f"Re-run conversion to regenerate the delta_model/ directory."
            )

        delta = _load_with_decompression(delta_path, compression, device=device)
        current = _apply_delta(prev, delta, layer_idx, manifest)

        yield layer_idx, current
        prev = current

        if stop_at is not None and layer_idx >= stop_at:
            return



def _float_to_int_dtype(dtype: torch.dtype) -> torch.dtype:
    """Map a floating-point dtype to its same-width integer dtype."""
    if dtype == torch.float32:
        return torch.int32
    if dtype == torch.float16 or dtype == torch.bfloat16:
        return torch.int16
    if dtype == torch.float64:
        return torch.int64
    raise ValueError(f"Unsupported floating point dtype for delta: {dtype}")

def _apply_delta(
    prev_layer: dict[str, torch.Tensor],
    delta: dict[str, torch.Tensor],
    layer_idx: int,
    manifest: DeltaManifest,
) -> dict[str, torch.Tensor]:
    """
    Apply a delta dict to the previous layer's tensors using integer arithmetic.

        W_curr[k] = (W_prev[k].view(int) + delta_int[k]).view(original_dtype)

    This is the exact inverse of _compute_delta() in delta_encoder.py.
    Deltas are stored as integers (e.g. int16 for bfloat16 models).
    The original dtype is recovered from manifest.tensor_dtypes.
    """
    from deltastream.core.manifest import dtype_from_str

    current: dict[str, torch.Tensor] = {}
    prefix = manifest.layer_prefix

    for key in delta:
        delta_t = delta[key]

        # Determine the original dtype from manifest
        full_key = f"{prefix}.{layer_idx}.{key}"
        dtype_str = manifest.tensor_dtypes.get(full_key)
        if dtype_str is None:
            # Fallback: use the delta tensor's own dtype
            orig_dtype = delta_t.dtype
            log_warning(f"Layer {layer_idx}: '{key}' not in manifest dtype map — using {orig_dtype}")
        else:
            orig_dtype = dtype_from_str(dtype_str)

        if key not in prev_layer:
            # Raw tensor stored for missing-prev edge case
            log_warning(
                f"Layer {layer_idx}: key '{key}' has no previous layer entry — using raw."
            )
            current[key] = delta_t.to(orig_dtype) if delta_t.dtype != orig_dtype else delta_t
            continue

        prev_t = prev_layer[key]

        # If it wasn't a float to begin with, it was stored raw
        if not orig_dtype.is_floating_point:
            current[key] = delta_t
            continue

        # Reconstruct via integer view addition
        int_dtype = _float_to_int_dtype(orig_dtype)
        prev_int = prev_t.view(int_dtype)
        
        recon_int = prev_int + delta_t
        current[key] = recon_int.view(orig_dtype)

    return current


def _load_non_layer(
    delta_model_dir: Path,
    *,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Load non-layer tensors from base/non_layer.safetensors."""
    path = delta_model_dir / "base" / "non_layer.safetensors"
    if not path.exists():
        log_warning("base/non_layer.safetensors not found — skipping non-layer tensors.")
        return {}
    try:
        return st_load_file(str(path), device=device)
    except Exception as exc:
        log_warning(f"Could not load non_layer.safetensors: {exc}")
        return {}

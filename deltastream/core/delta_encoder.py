"""
delta_encoder.py — Zero-loss delta encoding of transformer layer weights.

Algorithm
---------
Layer 0  → saved verbatim as base/layer_00.safetensors
Layer N  → delta[k] = W_N[k].float() - W_{N-1}[k].float(), cast back to orig dtype
           saved as deltas/layer_{N:02d}.delta.safetensors

Non-layer tensors (embeddings, lm_head, …) saved verbatim in base/non_layer.safetensors

Key design decision: all delta arithmetic is done in float32 to avoid BF16/FP16
accumulation errors. The delta is then immediately cast back to the original dtype,
so the file on disk is the same precision as the original — reconstruction just
reverses the operation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from deltastream.core.manifest import (
    DeltaManifest,
    LayerTopology,
    build_manifest,
    update_manifest_checksums,
    write_manifest,
)
from deltastream.core.weight_io import non_layer_tensors, tensors_for_layer
from deltastream.utils.logging import log_info, log_step, log_success, log_warning, make_progress

# ──────────────────────────────────────────────────────────────────────────────
# Compression helpers
# ──────────────────────────────────────────────────────────────────────────────

def _compress_file_inplace(path: Path, level: int = 1) -> None:
    """Read a file, compress with zstd at the given level, overwrite in-place."""
    try:
        import zstandard as zstd
    except ImportError:
        raise ImportError(
            "zstandard is required for compression. Install with: pip install zstandard"
        )
    raw = path.read_bytes()
    cctx = zstd.ZstdCompressor(level=level)
    path.write_bytes(cctx.compress(raw))


# How much bigger than the original a delta may be before we store raw instead.
# 1.0 = always store delta; 0.95 = store raw if delta is ≥ 95 % of original size.
DELTA_SIZE_RATIO_THRESHOLD = 1.02  # allow up to 2% overhead (practically never triggers)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def encode_model(
    weights: dict[str, torch.Tensor],
    topology: LayerTopology,
    output_dir: Path | str,
    source_model: str,
    *,
    compression: str = "none",
) -> DeltaManifest:
    """
    Convert a flat weight dict into delta-compressed format.

    Parameters
    ----------
    weights       : flat dict {full_tensor_name: tensor}
    topology      : LayerTopology from manifest.discover_layers()
    output_dir    : path to write delta_model/ contents (created if absent)
    source_model  : original HF model id / path (stored in manifest)
    compression   : 'none' (Phase 1). 'zstd' reserved for Phase 2.

    Returns
    -------
    manifest : DeltaManifest (also written to output_dir/manifest.json)
    """
    output_dir = Path(output_dir)
    _prepare_dirs(output_dir)

    log_step("Encoding model to delta format", str(output_dir))

    manifest = build_manifest(
        source_model=source_model,
        topology=topology,
        weights=weights,
        compression=compression,
    )

    # ── 1. Save base layer (layer 0) verbatim ────────────────────────────────
    _save_base_layer(weights, topology, output_dir)

    # ── 2. Save non-layer tensors (embeddings, head, norms) verbatim ─────────
    _save_non_layer_tensors(weights, topology, output_dir)

    # ── 3. Compute and save deltas for layers 1..N ───────────────────────────
    _save_all_deltas(weights, topology, output_dir, compression=compression)


    # ── 4. Compute and persist checksums ─────────────────────────────────────
    log_step("Computing checksums")
    update_manifest_checksums(manifest, output_dir)

    # ── 5. Write manifest ─────────────────────────────────────────────────────
    write_manifest(manifest, output_dir)

    _print_size_report(output_dir, weights)

    log_success("Delta encoding complete ✓")
    return manifest


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _prepare_dirs(output_dir: Path) -> None:
    (output_dir / "base").mkdir(parents=True, exist_ok=True)
    (output_dir / "deltas").mkdir(parents=True, exist_ok=True)


def _save_base_layer(
    weights: dict[str, torch.Tensor],
    topology: LayerTopology,
    output_dir: Path,
) -> None:
    """Save layer 0 tensors verbatim to base/layer_00.safetensors."""
    layer0 = tensors_for_layer(weights, topology.prefix, layer_idx=0)
    if not layer0:
        raise ValueError(
            f"No tensors found for layer 0 under prefix '{topology.prefix}'. "
            "Check that your model has at least one transformer block."
        )
    out_path = output_dir / "base" / "layer_00.safetensors"
    save_file(layer0, str(out_path), metadata={"layer": "0", "type": "base"})
    log_info(
        f"Base layer saved → [highlight]{out_path.name}[/highlight]  "
        f"({len(layer0)} tensors, {_file_mb(out_path):.1f} MB)"
    )


def _save_non_layer_tensors(
    weights: dict[str, torch.Tensor],
    topology: LayerTopology,
    output_dir: Path,
) -> None:
    """Save embeddings, lm_head, and other non-layer tensors verbatim."""
    non_layer = non_layer_tensors(weights, topology.prefix)

    # Deduplicate tied weights (save only the canonical copy)
    tied_targets: set[str] = set()
    for a, b in topology.tied_pairs:
        # Keep the first name alphabetically, skip the second
        tied_targets.add(max(a, b))

    # Filter out tied duplicates (they will be reconstructed from the first copy)
    non_layer_deduped = {k: v for k, v in non_layer.items() if k not in tied_targets}

    if not non_layer_deduped:
        log_warning("No non-layer tensors found (unusual — check model structure).")
        # still write an empty file so manifest paths stay consistent
        non_layer_deduped = {}

    out_path = output_dir / "base" / "non_layer.safetensors"
    if non_layer_deduped:
        save_file(
            non_layer_deduped,
            str(out_path),
            metadata={"type": "non_layer"},
        )
    else:
        # Write a minimal placeholder
        import json as _json
        out_path.write_bytes(b"\x02\x00\x00\x00\x00\x00\x00\x00{}")

    log_info(
        f"Non-layer tensors saved → [highlight]{out_path.name}[/highlight]  "
        f"({len(non_layer_deduped)} tensors, {_file_mb(out_path):.1f} MB)"
    )


def _save_all_deltas(
    weights: dict[str, torch.Tensor],
    topology: LayerTopology,
    output_dir: Path,
    compression: str = "none",
) -> None:
    """Compute and save delta files for layers 1 … N-1."""
    num_layers = topology.num_layers
    if num_layers < 2:
        log_warning("Model has fewer than 2 layers — no deltas to encode.")
        return

    use_zstd = compression.startswith("zstd")
    try:
        zstd_level = int(compression.split(":")[1]) if ":" in compression else 1
    except (IndexError, ValueError):
        zstd_level = 1

    prev_layer = tensors_for_layer(weights, topology.prefix, layer_idx=0)

    with make_progress() as progress:
        task = progress.add_task("Encoding deltas", total=num_layers - 1)

        for layer_idx in range(1, num_layers):
            curr_layer = tensors_for_layer(weights, topology.prefix, layer_idx=layer_idx)
            delta = _compute_delta(prev_layer, curr_layer, layer_idx)

            out_path = output_dir / "deltas" / f"layer_{layer_idx:02d}.delta.safetensors"
            save_file(
                delta,
                str(out_path),
                metadata={"layer": str(layer_idx), "type": "delta"},
            )

            if use_zstd:
                _compress_file_inplace(out_path, level=zstd_level)

            prev_layer = curr_layer
            progress.advance(task)



def _float_to_int_dtype(dtype: torch.dtype) -> torch.dtype:
    """Map a floating-point dtype to its same-width integer dtype."""
    if dtype == torch.float32:
        return torch.int32
    if dtype == torch.float16 or dtype == torch.bfloat16:
        return torch.int16
    if dtype == torch.float64:
        return torch.int64
    raise ValueError(f"Unsupported floating point dtype for delta: {dtype}")

def _compute_delta(
    prev_layer: dict[str, torch.Tensor],
    curr_layer: dict[str, torch.Tensor],
    layer_idx: int,
) -> dict[str, torch.Tensor]:
    """
    Compute per-tensor deltas using integer bit-representation subtraction:
        delta = curr.view(int) - prev.view(int)

    *** Critical design decision ***
    We do NOT use floating point subtraction (curr.float() - prev.float()).
    Floating point addition is not associative: (prev + (curr - prev)) may not equal
    curr exactly due to rounding at the ULP level, breaking bit-identical verification
    across accumulation chains.

    By viewing the floats as integers and performing wrapping integer subtraction,
    we guarantee mathematically perfect reconstruction:
        recon = (prev.view(int) + delta).view(orig_float_dtype)
    This also keeps the delta tensor the EXACT same byte size as the original
    (e.g., a bfloat16 tensor yields an int16 delta, saving 50% space vs a float32 delta).
    """
    delta: dict[str, torch.Tensor] = {}
    for key in curr_layer:
        curr_t = curr_layer[key]

        if key not in prev_layer:
            # Tensor appeared starting from this layer (edge case) — store raw
            log_warning(
                f"Layer {layer_idx}: key '{key}' missing in previous layer — storing raw."
            )
            delta[key] = curr_t
            continue

        prev_t = prev_layer[key]

        if curr_t.shape != prev_t.shape:
            raise ValueError(
                f"Shape mismatch at layer {layer_idx}, key '{key}': "
                f"{curr_t.shape} vs {prev_t.shape}"
            )

        # Integer / bool tensors: store raw (delta arithmetic not meaningful)
        if not curr_t.is_floating_point():
            delta[key] = curr_t
            continue

        # Float tensors: compute exact bit-wise delta using integer views
        int_dtype = _float_to_int_dtype(curr_t.dtype)
        curr_int = curr_t.view(int_dtype)
        prev_int = prev_t.view(int_dtype)
        
        delta[key] = curr_int - prev_int

    return delta


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────


def _file_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024) if path.exists() else 0.0


def _print_size_report(
    output_dir: Path,
    original_weights: dict[str, torch.Tensor],
) -> None:
    """Compare delta_model/ total size vs estimated original size."""
    original_bytes = sum(t.nbytes for t in original_weights.values())

    delta_bytes = sum(
        f.stat().st_size
        for f in output_dir.rglob("*")
        if f.is_file() and f.suffix in {".safetensors", ".json"}
    )

    orig_mb = original_bytes / (1024 * 1024)
    delta_mb = delta_bytes / (1024 * 1024)
    ratio = delta_mb / orig_mb if orig_mb > 0 else 1.0

    log_info(
        f"Size report: original ≈ [highlight]{orig_mb:.1f} MB[/highlight] | "
        f"delta_model/ = [highlight]{delta_mb:.1f} MB[/highlight] "
        f"([highlight]{ratio * 100:.1f}%[/highlight] of original)"
    )

"""
convert.py — CLI handler for the 'convert' subcommand.

Usage:
  python -m deltastream convert --model <id_or_path> --output <dir> [options]
"""

from __future__ import annotations

from pathlib import Path

from deltastream.core.delta_encoder import encode_model
from deltastream.core.manifest import discover_layers
from deltastream.core.weight_io import load_model_weights
from deltastream.utils.logging import log_header, log_info, log_step, log_success


def run_convert(
    model_path: str,
    output_dir: str,
    *,
    compression: str = "none",
    cache_dir: str | None = None,
    layer_prefix: str | None = None,
) -> None:
    """
    Convert a HuggingFace model into delta-compressed format.

    Parameters
    ----------
    model_path   : HF repo ID (e.g. 'Qwen/Qwen2-0.5B') or local directory path
    output_dir   : directory to write delta_model/ contents
    compression  : 'none' (Phase 1 only)
    cache_dir    : HuggingFace hub cache directory override
    layer_prefix : override auto-detected layer prefix (e.g. 'model.layers')
    """
    log_header("DeltaStream  —  Convert")
    log_info(f"Source model : [highlight]{model_path}[/highlight]")
    log_info(f"Output dir   : [highlight]{output_dir}[/highlight]")
    log_info(f"Compression  : [highlight]{compression}[/highlight]")

    # ── 1. Load weights ───────────────────────────────────────────────────────
    weights, meta = load_model_weights(model_path, cache_dir=cache_dir)

    # ── 2. Discover layer topology ────────────────────────────────────────────
    topology = discover_layers(weights, tied_pairs=meta["tied_weight_pairs"])
    if layer_prefix:
        topology.prefix = layer_prefix
        log_info(f"Layer prefix overridden → [highlight]{layer_prefix}[/highlight]")

    # ── 3. Encode ─────────────────────────────────────────────────────────────
    out = Path(output_dir)
    manifest = encode_model(
        weights=weights,
        topology=topology,
        output_dir=out,
        source_model=model_path,
        compression=compression,
    )

    log_success(
        f"Done! delta_model written to: [highlight]{out.resolve()}[/highlight]\n"
        f"  Layers   : {manifest.num_layers}\n"
        f"  Tensors  : {len(manifest.tensor_dtypes)}\n"
        f"  Delta files: {len(manifest.delta_files)}"
    )

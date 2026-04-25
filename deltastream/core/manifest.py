"""
manifest.py — Layer topology discovery and manifest read/write for DeltaStream.

The manifest.json file is the single source of truth for a delta_model/ directory.
It records:
  - Source model metadata
  - Layer prefix and count
  - Per-tensor original dtypes
  - File list (base + delta shards)
  - Tied weight info
  - Integrity checksums
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from deltastream.utils.logging import log_info, log_step, log_warning

# ──────────────────────────────────────────────────────────────────────────────
# Known layer prefixes for popular model families
# ──────────────────────────────────────────────────────────────────────────────

KNOWN_LAYER_PREFIXES: list[str] = [
    "model.layers",          # LLaMA, Mistral, Qwen2, Gemma, Falcon, Phi-3
    "transformer.h",         # GPT-2, GPT-J, BLOOM (partial)
    "transformer.blocks",    # MPT
    "model.decoder.layers",  # OPT
    "gpt_neox.layers",       # GPT-NeoX, Pythia
    "model.transformer.h",   # some BLOOM variants
    "language_model.model.layers",  # LLaVA-style
]

MANIFEST_VERSION = "1.0.0"

# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class LayerTopology:
    """Describes how transformer layers are organised in a weight dict."""

    prefix: str
    """Key prefix shared by all layer tensors, e.g. 'model.layers'."""

    num_layers: int
    """Total number of transformer blocks (0-indexed)."""

    layer_tensor_keys: dict[int, list[str]]
    """layer_idx → list of short tensor keys within that layer."""

    non_layer_keys: list[str]
    """Tensor keys that do NOT belong to any numbered transformer block."""

    tied_pairs: list[tuple[str, str]] = field(default_factory=list)
    """Tied weight pairs detected in the original weight dict."""


@dataclass
class DeltaManifest:
    """All metadata needed to reconstruct a delta_model/."""

    deltastreamx_version: str
    source_model: str
    num_layers: int
    layer_prefix: str
    base_layer_idx: int
    tensor_dtypes: dict[str, str]
    """Full tensor key → dtype string, e.g. 'BF16', 'F32'."""
    compression: str
    """'none' | 'zstd' (future)."""
    base_files: list[str]
    """Relative paths of verbatim-copied base/non-layer files."""
    delta_files: list[str]
    """Relative paths of delta shard files in layer order."""
    tied_pairs: list[list[str]]
    """Serialisable version of tied weight pairs."""
    checksums: dict[str, str] = field(default_factory=dict)
    """Relative path → sha256 hex digest (populated after encoding)."""


# ──────────────────────────────────────────────────────────────────────────────
# Topology discovery
# ──────────────────────────────────────────────────────────────────────────────


def discover_layers(
    weights: dict[str, torch.Tensor],
    tied_pairs: list[tuple[str, str]] | None = None,
) -> LayerTopology:
    """
    Automatically discover the layer topology from a flat weight dict.

    Tries each known prefix in turn; falls back to a regex scan if none match.

    Returns a LayerTopology describing the prefix, layer count, and grouping.
    """
    log_step("Discovering layer topology")

    prefix, indices = _find_prefix_and_indices(weights)
    log_info(
        f"Layer prefix: [highlight]{prefix}[/highlight]  "
        f"— [highlight]{len(indices)}[/highlight] layer(s)"
    )

    layer_tensor_keys: dict[int, list[str]] = {}
    for idx in indices:
        full_prefix = f"{prefix}.{idx}."
        short_keys = [
            k[len(full_prefix):]
            for k in weights
            if k.startswith(full_prefix)
        ]
        layer_tensor_keys[idx] = sorted(short_keys)

    non_layer = [k for k in weights if not k.startswith(f"{prefix}.")]

    topology = LayerTopology(
        prefix=prefix,
        num_layers=len(indices),
        layer_tensor_keys=layer_tensor_keys,
        non_layer_keys=non_layer,
        tied_pairs=tied_pairs or [],
    )
    log_info(
        f"[highlight]{len(non_layer)}[/highlight] non-layer tensor(s) "
        f"(embeddings, head, norms, …)"
    )
    return topology


def _find_prefix_and_indices(
    weights: dict[str, torch.Tensor],
) -> tuple[str, list[int]]:
    """Return (best_prefix, sorted_layer_indices)."""
    best_prefix: str | None = None
    best_indices: list[int] = []

    for prefix in KNOWN_LAYER_PREFIXES:
        indices = _extract_indices(weights, prefix)
        if len(indices) > len(best_indices):
            best_indices = indices
            best_prefix = prefix

    if best_prefix is None or not best_indices:
        # Generic regex fallback: find prefix ending in .N.
        best_prefix, best_indices = _regex_fallback(weights)

    if not best_indices:
        raise ValueError(
            "Could not determine layer prefix from weight keys. "
            "Please check the model or manually specify --layer-prefix."
        )

    return best_prefix, sorted(best_indices)


def _extract_indices(
    weights: dict[str, torch.Tensor],
    prefix: str,
) -> list[int]:
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.")
    indices: set[int] = set()
    for key in weights:
        m = pattern.match(key)
        if m:
            indices.add(int(m.group(1)))
    return sorted(indices)


def _regex_fallback(
    weights: dict[str, torch.Tensor],
) -> tuple[str, list[int]]:
    """Scan all keys for pattern prefix.N.suffix and pick the most common prefix."""
    pattern = re.compile(r"^(.+?)\.(\d+)\.")
    prefix_counts: dict[str, set[int]] = {}
    for key in weights:
        m = pattern.match(key)
        if m:
            p, idx = m.group(1), int(m.group(2))
            prefix_counts.setdefault(p, set()).add(idx)

    if not prefix_counts:
        return "", []

    best = max(prefix_counts, key=lambda p: len(prefix_counts[p]))
    log_warning(f"Using auto-detected prefix via fallback: [highlight]{best}[/highlight]")
    return best, sorted(prefix_counts[best])


# ──────────────────────────────────────────────────────────────────────────────
# Manifest read / write
# ──────────────────────────────────────────────────────────────────────────────


def build_manifest(
    *,
    source_model: str,
    topology: LayerTopology,
    weights: dict[str, torch.Tensor],
    compression: str = "none",
) -> DeltaManifest:
    """Construct a DeltaManifest from topology and weights (before saving)."""
    tensor_dtypes = {k: _dtype_str(v) for k, v in weights.items()}
    num_layers = topology.num_layers

    base_files = ["base/layer_00.safetensors", "base/non_layer.safetensors"]
    delta_files = [
        f"deltas/layer_{i:02d}.delta.safetensors"
        for i in range(1, num_layers)
    ]

    return DeltaManifest(
        deltastreamx_version=MANIFEST_VERSION,
        source_model=source_model,
        num_layers=num_layers,
        layer_prefix=topology.prefix,
        base_layer_idx=0,
        tensor_dtypes=tensor_dtypes,
        compression=compression,
        base_files=base_files,
        delta_files=delta_files,
        tied_pairs=[[a, b] for a, b in topology.tied_pairs],
    )


def write_manifest(manifest: DeltaManifest, output_dir: Path) -> Path:
    """Write manifest.json to output_dir. Returns the file path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    data = asdict(manifest)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    log_info(f"Manifest written → [highlight]{path}[/highlight]")
    return path


def read_manifest(delta_model_dir: Path) -> DeltaManifest:
    """Load and parse manifest.json from a delta_model/ directory."""
    path = delta_model_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"No manifest.json found in: {delta_model_dir}")
    with open(path, "r", encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    # Re-hydrate
    data["tied_pairs"] = [tuple(pair) for pair in data.get("tied_pairs", [])]
    return DeltaManifest(**data)


# ──────────────────────────────────────────────────────────────────────────────
# Checksum helpers
# ──────────────────────────────────────────────────────────────────────────────


def compute_checksum(file_path: Path) -> str:
    """Return SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def update_manifest_checksums(manifest: DeltaManifest, output_dir: Path) -> None:
    """Compute and store checksums for all files listed in the manifest."""
    all_files = manifest.base_files + manifest.delta_files
    for rel_path in all_files:
        abs_path = output_dir / rel_path
        if abs_path.exists():
            manifest.checksums[rel_path] = compute_checksum(abs_path)


def verify_checksums(manifest: DeltaManifest, delta_model_dir: Path) -> bool:
    """
    Verify all stored checksums match the files on disk.
    Returns True if all match, False otherwise.
    """
    if not manifest.checksums:
        log_warning("No checksums stored in manifest — skipping integrity check.")
        return True
    all_pass = True
    for rel_path, expected in manifest.checksums.items():
        abs_path = delta_model_dir / rel_path
        if not abs_path.exists():
            log_warning(f"Missing file: {rel_path}")
            all_pass = False
            continue
        actual = compute_checksum(abs_path)
        if actual != expected:
            log_warning(f"Checksum MISMATCH: {rel_path}")
            all_pass = False
    return all_pass


# ──────────────────────────────────────────────────────────────────────────────
# Dtype helpers
# ──────────────────────────────────────────────────────────────────────────────

_DTYPE_MAP: dict[torch.dtype, str] = {
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int8: "I8",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}

_DTYPE_INV: dict[str, torch.dtype] = {v: k for k, v in _DTYPE_MAP.items()}


def _dtype_str(tensor: torch.Tensor) -> str:
    return _DTYPE_MAP.get(tensor.dtype, str(tensor.dtype))


def dtype_from_str(s: str) -> torch.dtype:
    if s not in _DTYPE_INV:
        raise ValueError(f"Unknown dtype string: {s!r}")
    return _DTYPE_INV[s]

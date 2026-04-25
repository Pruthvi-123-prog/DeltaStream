"""
test_compression.py — Round-trip test: encode with zstd, decode, verify bit-identical.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from deltastream.core.delta_encoder import encode_model
from deltastream.core.delta_decoder import reconstruct_layer, reconstruct_all_layers
from deltastream.core.manifest import discover_layers, read_manifest


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_fake_model(num_layers: int = 3, hidden: int = 16) -> dict[str, torch.Tensor]:
    """Create a minimal fake weight dict with GPT2-style keys."""
    weights = {}
    for i in range(num_layers):
        prefix = f"transformer.h.{i}"
        weights[f"{prefix}.attn.c_attn.weight"] = torch.randn(hidden, hidden * 3)
        weights[f"{prefix}.attn.c_attn.bias"] = torch.randn(hidden * 3)
        weights[f"{prefix}.mlp.c_fc.weight"] = torch.randn(hidden, hidden * 4)
        weights[f"{prefix}.mlp.c_fc.bias"] = torch.randn(hidden * 4)
    weights["transformer.wte.weight"] = torch.randn(100, hidden)
    weights["transformer.wpe.weight"] = torch.randn(50, hidden)
    weights["lm_head.weight"] = weights["transformer.wte.weight"].clone()  # avoid shared memory error in safetensors
    return weights


# ─── Tests ───────────────────────────────────────────────────────────────────

def test_compression_round_trip_bit_identical():
    """Compressed encode → decode must be bit-identical to uncompressed."""
    weights = _make_fake_model(num_layers=3, hidden=32)
    topology = discover_layers(weights)

    with tempfile.TemporaryDirectory() as tmp:
        out_uncompressed = Path(tmp) / "delta_uncompressed"
        out_compressed = Path(tmp) / "delta_compressed"

        # Encode without compression
        encode_model(weights, topology, out_uncompressed, source_model="test", compression="none")
        # Encode with zstd:1
        encode_model(weights, topology, out_compressed, source_model="test", compression="zstd:1")

        # Reconstruct all layers from both
        recon_u = reconstruct_all_layers(out_uncompressed, device="cpu")
        recon_c = reconstruct_all_layers(out_compressed, device="cpu")

        # Compare every reconstructed tensor
        assert set(recon_u.keys()) == set(recon_c.keys()), "Key mismatch between compressed/uncompressed"
        for key in recon_u:
            t_u = recon_u[key]
            t_c = recon_c[key]
            assert t_u.shape == t_c.shape, f"Shape mismatch for {key}"
            assert torch.equal(t_u, t_c), f"Tensor mismatch for {key} (not bit-identical)"


def test_compression_manifest_field():
    """Manifest should record compression='zstd:1' when compress is used."""
    weights = _make_fake_model(num_layers=2, hidden=16)
    topology = discover_layers(weights)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "delta_zstd"
        encode_model(weights, topology, out, source_model="test", compression="zstd:1")
        manifest = read_manifest(out)
        assert manifest.compression == "zstd:1", f"Expected 'zstd:1', got '{manifest.compression}'"


def test_compressed_deltas_are_smaller():
    """Delta files should be smaller after zstd compression."""
    weights = _make_fake_model(num_layers=3, hidden=64)
    topology = discover_layers(weights)

    with tempfile.TemporaryDirectory() as tmp:
        out_u = Path(tmp) / "delta_uncompressed"
        out_c = Path(tmp) / "delta_compressed"

        encode_model(weights, topology, out_u, source_model="test", compression="none")
        encode_model(weights, topology, out_c, source_model="test", compression="zstd:1")

        def delta_size(d: Path) -> int:
            return sum(f.stat().st_size for f in (d / "deltas").iterdir() if f.is_file())

        size_u = delta_size(out_u)
        size_c = delta_size(out_c)
        assert size_c < size_u, (
            f"Compressed deltas ({size_c} bytes) are not smaller than uncompressed ({size_u} bytes)"
        )


def test_uncompressed_still_works():
    """Encoding with compression='none' still produces correct output."""
    weights = _make_fake_model(num_layers=2, hidden=16)
    topology = discover_layers(weights)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "delta_none"
        encode_model(weights, topology, out, source_model="test", compression="none")
        recon = reconstruct_all_layers(out, device="cpu")

        # Every original layer tensor should be reconstructable
        prefix = topology.prefix
        for layer_idx in range(topology.num_layers):
            for short_key in topology.layer_tensor_keys[layer_idx]:
                full_key = f"{prefix}.{layer_idx}.{short_key}"
                assert full_key in recon, f"Missing key {full_key}"
                orig = weights[full_key]
                assert torch.equal(orig, recon[full_key]), f"Not bit-identical: {full_key}"

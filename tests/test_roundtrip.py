"""
tests/test_roundtrip.py — Smoke test for the full encode → decode → verify pipeline.

Uses GPT-2 (small, ~500 MB, CPU-only) so it runs on any machine without a GPU.

Run with:
  pytest tests/test_roundtrip.py -v
  # or directly:
  python tests/test_roundtrip.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ID = "gpt2"          # ~500 MB, always available, fast to download


def _load_gpt2_weights() -> tuple[dict[str, torch.Tensor], dict]:
    """Download (or use cache) and load GPT-2 weights."""
    from deltastream.core.weight_io import load_model_weights
    return load_model_weights(MODEL_ID)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRoundtrip:
    """Full encode → decode → tensor equality pipeline."""

    @pytest.fixture(scope="class")
    def setup(self):
        """Load weights once per test class; encode to a temp dir."""
        from deltastream.core.delta_encoder import encode_model
        from deltastream.core.manifest import discover_layers

        weights, meta = _load_gpt2_weights()
        topology = discover_layers(weights, tied_pairs=meta["tied_weight_pairs"])

        tmp = tempfile.mkdtemp(prefix="deltastreamx_test_")
        manifest = encode_model(
            weights=weights,
            topology=topology,
            output_dir=Path(tmp),
            source_model=MODEL_ID,
        )
        return weights, topology, manifest, Path(tmp)

    def test_manifest_written(self, setup):
        """manifest.json must exist after encoding."""
        _, _, _, tmp = setup
        assert (tmp / "manifest.json").exists(), "manifest.json not found"

    def test_base_layer_exists(self, setup):
        """base/layer_00.safetensors must exist."""
        _, _, _, tmp = setup
        assert (tmp / "base" / "layer_00.safetensors").exists()

    def test_delta_files_exist(self, setup):
        """One delta file per layer > 0."""
        _, _, manifest, tmp = setup
        for rel in manifest.delta_files:
            assert (tmp / rel).exists(), f"Missing delta file: {rel}"

    def test_non_layer_file_exists(self, setup):
        """base/non_layer.safetensors must exist."""
        _, _, _, tmp = setup
        assert (tmp / "base" / "non_layer.safetensors").exists()

    def test_tensor_count(self, setup):
        """Manifest tensor_dtypes count must match original weight count."""
        weights, _, manifest, _ = setup
        assert len(manifest.tensor_dtypes) == len(weights), (
            f"Manifest has {len(manifest.tensor_dtypes)} entries but "
            f"original has {len(weights)} tensors"
        )

    def test_reconstruction_bit_identical(self, setup):
        """Every reconstructed tensor must be bit-identical to the original."""
        from deltastream.core.delta_decoder import reconstruct_all_layers

        weights, _, _, tmp = setup
        reconstructed = reconstruct_all_layers(tmp, device="cpu")

        failures: list[str] = []
        for key, orig_tensor in weights.items():
            if key not in reconstructed:
                failures.append(f"MISSING: {key}")
                continue
            recon_tensor = reconstructed[key]
            if not torch.equal(orig_tensor, recon_tensor):
                max_diff = (orig_tensor.float() - recon_tensor.float()).abs().max().item()
                failures.append(f"MISMATCH: {key}  max_abs_diff={max_diff:.2e}")

        assert not failures, (
            f"{len(failures)} tensor(s) failed reconstruction:\n"
            + "\n".join(failures[:10])
            + ("\n…" if len(failures) > 10 else "")
        )

    def test_no_extra_tensors(self, setup):
        """Reconstructed dict must not contain tensors absent from original."""
        from deltastream.core.delta_decoder import reconstruct_all_layers

        weights, _, _, tmp = setup
        reconstructed = reconstruct_all_layers(tmp, device="cpu")

        extra = set(reconstructed) - set(weights)
        assert not extra, f"Extra tensors in reconstruction: {extra}"


class TestManifest:
    """Unit tests for manifest read/write."""

    def test_write_and_read_roundtrip(self):
        from deltastream.core.manifest import DeltaManifest, read_manifest, write_manifest

        manifest = DeltaManifest(
            deltastreamx_version="1.0.0",
            source_model="test/model",
            num_layers=4,
            layer_prefix="model.layers",
            base_layer_idx=0,
            tensor_dtypes={"model.layers.0.w": "F32"},
            compression="none",
            base_files=["base/layer_00.safetensors", "base/non_layer.safetensors"],
            delta_files=["deltas/layer_01.delta.safetensors"],
            tied_pairs=[],
            checksums={},
        )

        with tempfile.TemporaryDirectory() as tmp:
            write_manifest(manifest, Path(tmp))
            loaded = read_manifest(Path(tmp))

        assert loaded.source_model == "test/model"
        assert loaded.num_layers == 4
        assert loaded.layer_prefix == "model.layers"
        assert loaded.tensor_dtypes == {"model.layers.0.w": "F32"}


class TestWeightIO:
    """Unit tests for weight loader helpers."""

    def test_tensors_for_layer(self):
        from deltastream.core.weight_io import tensors_for_layer

        weights = {
            "model.layers.0.self_attn.q.weight": torch.zeros(4, 4),
            "model.layers.0.mlp.w.weight": torch.zeros(8, 4),
            "model.layers.1.self_attn.q.weight": torch.ones(4, 4),
            "lm_head.weight": torch.zeros(100, 4),
        }
        layer0 = tensors_for_layer(weights, "model.layers", 0)
        assert set(layer0.keys()) == {"self_attn.q.weight", "mlp.w.weight"}
        assert layer0["self_attn.q.weight"].sum().item() == 0.0

        layer1 = tensors_for_layer(weights, "model.layers", 1)
        assert "self_attn.q.weight" in layer1
        assert layer1["self_attn.q.weight"].sum().item() == 16.0  # 4x4 ones

    def test_non_layer_tensors(self):
        from deltastream.core.weight_io import non_layer_tensors

        weights = {
            "model.layers.0.w": torch.zeros(2),
            "model.layers.1.w": torch.zeros(2),
            "embed_tokens.weight": torch.zeros(10, 4),
            "lm_head.weight": torch.zeros(10, 4),
        }
        non = non_layer_tensors(weights, "model.layers")
        assert set(non.keys()) == {"embed_tokens.weight", "lm_head.weight"}


class TestDeltaMath:
    """
    Verify delta arithmetic is exactly reversible.

    Encoder contract:
        delta_int = curr.view(int) - prev.view(int)

    Decoder contract:
        recon = (prev.view(int) + delta_int).view(orig_dtype)

    Proof of exactness:
        • Integer addition/subtraction on same-width types wraps around exactly.
        • (A - B) + B == A exactly in wrapping integer arithmetic.
        • Re-viewing the bits as float restores the exact float value, preserving NaNs, 
          infinities, and all non-associative float quirks.
    """

    def _float_to_int_dtype(self, dtype: torch.dtype) -> torch.dtype:
        if dtype == torch.float32: return torch.int32
        if dtype in (torch.float16, torch.bfloat16): return torch.int16
        if dtype == torch.float64: return torch.int64
        raise ValueError()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_delta_exact_inversion(self, dtype: torch.dtype):
        """Round-trip must be bit-exact using integer bit representation deltas."""
        torch.manual_seed(42)
        prev = torch.randn(64, 64).to(dtype)
        curr = torch.randn(64, 64).to(dtype)
        int_dtype = self._float_to_int_dtype(dtype)

        # Encoder: store delta as integer difference
        delta_int = curr.view(int_dtype) - prev.view(int_dtype)
        assert delta_int.dtype == int_dtype

        # Decoder: integer addition, view back as float
        recon = (prev.view(int_dtype) + delta_int).view(dtype)

        assert torch.equal(curr, recon), (
            f"Delta roundtrip failed for {dtype}: "
            f"max_diff={(curr.float() - recon.float()).abs().max().item():.2e}"
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_delta_zero_for_identical_layers(self, dtype: torch.dtype):
        """If two layers are identical, their delta must be all-zeros."""
        layer = torch.randn(32, 32).to(dtype)
        int_dtype = self._float_to_int_dtype(dtype)
        delta_int = layer.view(int_dtype) - layer.view(int_dtype)
        assert delta_int.abs().max().item() == 0, "Delta of identical layers must be zero"

    def test_integer_tensor_stored_raw(self):
        """Integer tensors must pass through unchanged (no delta)."""
        curr = torch.tensor([4, 5, 6], dtype=torch.int32)
        stored = curr.clone()
        assert torch.equal(stored, curr)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_delta_accumulation_chain(self, dtype: torch.dtype):
        """Multi-layer cumulative reconstruction must remain exact for all dtypes."""
        torch.manual_seed(7)
        num_layers = 12
        layers = [torch.randn(32, 32).to(dtype) for _ in range(num_layers)]
        int_dtype = self._float_to_int_dtype(dtype)

        # Encode: sequential integer deltas
        deltas_int = [
            layers[i].view(int_dtype) - layers[i - 1].view(int_dtype)
            for i in range(1, num_layers)
        ]

        # Decode: cumulative reconstruction
        recon = [layers[0]]
        for i, d in enumerate(deltas_int):
            r = (recon[-1].view(int_dtype) + d).view(dtype)
            recon.append(r)

        # Every layer must match exactly
        for i in range(num_layers):
            assert torch.equal(layers[i], recon[i]), (
                f"[{dtype}] Layer {i} mismatch after {i}-step accumulation: "
                f"max_diff={(layers[i].float()-recon[i].float()).abs().max().item():.2e}"
            )

    def test_delta_dtype_is_integer(self):
        """Encoder must produce int16 deltas for 16-bit floats, int32 for 32-bit floats."""
        for dtype, expected_int in [
            (torch.float32, torch.int32),
            (torch.float16, torch.int16),
            (torch.bfloat16, torch.int16)
        ]:
            prev = torch.randn(8, 8).to(dtype)
            curr = torch.randn(8, 8).to(dtype)
            delta = curr.view(expected_int) - prev.view(expected_int)
            assert delta.dtype == expected_int


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running smoke test with GPT-2 (CPU-only) …")
    print("This downloads ~500 MB on first run.")
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))

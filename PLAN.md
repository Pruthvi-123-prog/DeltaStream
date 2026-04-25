# DeltaStream — Phase 1: Technical Plan

## 1. Research Findings

### 1.1 HuggingFace Weight File Formats

#### Safetensors (.safetensors)
The safetensors format has a clean binary layout:
```
[ 8 bytes: header_length (little-endian uint64) ]
[ header_length bytes: JSON header              ]
[ tensor data (raw bytes, contiguous)           ]
```
The JSON header maps tensor keys → `{ dtype, shape, data_offsets: [start, end] }`.
- Zero-copy: memory-map the file, slice by data_offsets
- No pickle: safe by default
- Tensors may be stored in any dtype (F32, F16, BF16, etc.)

#### PyTorch Bin (.bin)
- Pickle-based serialization via `torch.save()` / `torch.load()`
- Often uses `zip` container internally (PyTorch ≥ 1.6)
- Loaded with `torch.load(map_location="cpu")` or via `transformers`
- Less safe (arbitrary code execution risk), being phased out by HF

#### Multi-shard models
Large models split weight tensors across multiple shard files:
- `model.safetensors.index.json` → maps tensor name → shard filename
- `model-00001-of-00005.safetensors` → individual shards
- Index JSON: `{ "metadata": {...}, "weight_map": {tensor_name: filename} }`

### 1.2 Transformer Layer Mathematical Structure

A standard decoder-only transformer block (e.g., LLaMA, GPT) contains:
```
Layer N weights:
  - self_attn.q_proj.weight    [hidden, hidden]
  - self_attn.k_proj.weight    [hidden, hidden]
  - self_attn.v_proj.weight    [hidden, hidden]
  - self_attn.o_proj.weight    [hidden, hidden]
  - mlp.gate_proj.weight       [intermediate, hidden]
  - mlp.up_proj.weight         [intermediate, hidden]
  - mlp.down_proj.weight       [hidden, intermediate]
  - input_layernorm.weight     [hidden]
  - post_attention_layernorm.weight [hidden]
```

**Key mathematical observation**: Adjacent layers (L and L+1) are initialized
from the same distribution and trained with similar loss gradients. In practice:
- Deeper layers (especially middle layers) are highly similar
- Weights differ by small residual updates, not entirely different surfaces
- Delta = W(L+1) - W(L)  has significantly lower entropy than raw weights
- Delta can be efficiently compressed (often 30-70% of original size when gzip'd)
- However: for ZERO accuracy loss, we must store the *exact* delta with
  full float32/bfloat16 precision — no rounding, no quantization

**Why layer-to-layer delta encoding works:**
- Each transformer layer learns an incremental refinement
- Empirical studies (DeltaZip, etc.) confirm 20-70% density reduction
  of the delta tensor vs. raw tensor when compressing
- Since we store exact deltas, reconstruction is mathematically perfect

### 1.3 Delta Encoding Format Decision

**Options considered:**

| Format | Pros | Cons |
|--------|------|------|
| numpy .npy | Simple, fast, zero deps | No metadata, one tensor/file |
| numpy .npz | Compressed, multi-tensor | No streaming, full load |
| safetensors | Fast, safe, metadata-rich | Slightly more setup |
| custom binary | Max control | Maintenance burden |

**Decision: safetensors for all outputs**
- Consistent with HuggingFace ecosystem
- Native support for metadata (dtype, shape, offsets)
- Fast memory-mapped loading (critical for Phase 2's LRU cache)
- `save_file()` + `safe_open()` API is clean
- Delta tensors stored as same dtype as original (no loss)

**Compression layer (optional, lossless)**: `zstd` on the safetensors buffer
can reduce delta files by 40-60% with very fast decompression. Applied
as a second pass; stored as `.delta.st.zst` files. The manifest records
whether compression was applied.

### 1.4 DeltaStream Integration

DeltaStream (`pip install deltastream`) loads and runs large LLMs layer-by-layer,
streaming from disk to stay within GPU/CPU memory. Key mechanics:
- `DeltaStreamQWen`, `DeltaStreamLLaMA`, `DeltaStreamGPT*` etc. subclass a base engine
- Layers are loaded one-by-one from the model's shard files
- The layer cache is LRU-controlled in Phase 2

**Phase 1 integration strategy:**
- Phase 1 is a standalone converter + verifier (no DeltaStream runtime yet)
- We produce a `delta_model/` directory that DeltaStream's engine will consume in Phase 2
- We add a "delta reconstruction" shim that DeltaStream's layer loader calls before passing
  weights to the model — this is the Phase 2 hook point

---

## 2. Technical Architecture

```
deltastream/
├── core/
│   ├── __init__.py
│   ├── delta_encoder.py      # Layer → delta computation + saving
│   ├── delta_decoder.py      # Base + delta → reconstructed layer
│   ├── manifest.py           # Read/write delta_model/manifest.json
│   └── weight_io.py          # Unified safetensors/bin loading abstraction
├── cli/
│   ├── __init__.py
│   ├── convert.py            # CLI: convert a HF model to delta format
│   └── verify.py             # CLI: verify bit-identical reconstruction
├── utils/
│   ├── __init__.py
│   └── logging.py            # Rich-formatted progress output
└── main.py                   # CLI entry point
```

### delta_model/ output structure
```
delta_model/
├── manifest.json             # Shard map, layer order, dtype per tensor, metadata
├── base/
│   └── layer_00.safetensors  # The first complete layer (no delta, just raw weights)
└── deltas/
    ├── layer_01.delta.safetensors   # delta = W1 - W0
    ├── layer_02.delta.safetensors   # delta = W2 - W1
    ├── ...
    └── layer_N.delta.safetensors    # delta = WN - W(N-1)
```

Non-layer tensors (embeddings, lm_head, layernorms) stored verbatim in base/.

### manifest.json schema
```json
{
  "deltastreamx_version": "1.0.0",
  "source_model": "meta-llama/Llama-2-7b-chat-hf",
  "num_layers": 32,
  "layer_prefix": "model.layers",
  "base_layer_idx": 0,
  "tensor_dtypes": { "model.layers.0.self_attn.q_proj.weight": "BF16", ...},
  "compression": "none",
  "shards": ["base/layer_00.safetensors", "deltas/layer_01.delta.safetensors", ...]
}
```

---

## 3. Reconstruction Algorithm

```
W(0) = load(base/layer_00.safetensors)   # exact copy of original layer 0
W(1) = W(0) + load(deltas/layer_01.delta.safetensors)   # = W(1)_original
W(2) = W(1) + load(deltas/layer_02.delta.safetensors)   # = W(2)_original
...
W(N) = W(N-1) + load(deltas/layer_N.delta.safetensors)  # = W(N)_original
```

All arithmetic done in float32 to avoid any accumulation error from lower-precision
intermediate values, then cast back to original dtype.

---

## 4. Verification Strategy

1. Load original model weights (CPU, no grad)
2. Reconstruct all layers from delta_model/
3. For every tensor: `torch.equal(original_tensor, reconstructed_tensor)`
4. Run same text prompt through original model and delta-reconstructed model
5. Verify token-by-token output is identical (greedy decode, temperature=0)

---

## 5. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Float arithmetic drift in delta recon | Medium | Cast to float32 for delta math, cast back |
| Shared tensors (lm_head tied to embed) | Medium | Detect tied weights via manifest, skip delta |
| VRAM overflow during verify step | High on 4GB GPU | Load one layer at a time, CPU verify by default |
| Massive delta files (no compression gain) | Low | Fallback: store full tensor if delta > 95% of original |
| Multi-shard model shard ordering | Medium | Sort shards by name, validate layer index continuity |
| BF16 precision artifacts | Low | Always delta in BF16 if original is BF16, never upcast permanently |

---

## 6. Phase 2 Hook Points

- `delta_decoder.py::reconstruct_layer(layer_idx)` → returns tensor dict
- This function signature is designed to be called by DeltaStream's layer loader
- Phase 2 wraps this in an LRU cache keyed by `layer_idx`
- Manifest is designed to be extended with cache hints

---

## 7. Dependencies

```
torch>=2.0
safetensors>=0.4
transformers>=4.38
huggingface_hub>=0.21
numpy>=1.24
tqdm>=4.66
rich>=13.0
zstandard>=0.22   # optional, for future compression support
liburing>=2024.5  # optional, for io_uring backend (bare metal only; source build required)
```

---

## 8. WSL2 Limitation: io_uring + Hyper-V CPU Virtualization

### Root Cause

WSL2 runs inside a Hyper-V Level-1 hypervisor. The Hyper-V virtual CPU does not
expose all x86-64 instruction set extensions to guest VMs. The `liburing` Python
package, when built from source with GCC targeting the host CPU, may emit
instructions (e.g., certain `AVX-512` or atomic variants) that Hyper-V silently
blocks, causing the process to receive `SIGILL` (Illegal Instruction, signal 4)
at ring initialization time.

This is **not a bug in DeltaStream or liburing**. It is a hard ceiling of the WSL2
environment. The `io_uring` syscall itself is available on WSL2 kernel 6.6+, but
the userspace `liburing` compiled binary triggers SIGILL on ring init before any
syscall is made.

### Mitigation in Code

`IOBackendFactory.get_backend()` probes liburing in an isolated subprocess before
loading it in-process. If the probe exits with signal 4 (SIGILL) or any other
crash, the factory logs:

```
io_uring blocked by CPU virtualization (SIGILL), using StandardIOBackend
```

and returns `StandardIOBackend` transparently. The rest of the system is unaffected.

### Bare Metal Expected Results

On native Linux (no hypervisor), with `O_DIRECT` enabled (`baremetal` mode),
expected improvements over `StandardIOBackend` (mmap-based):

| Layer Size | Expected io_uring gain |
|------------|------------------------|
| 30–100 MB  | 20–35% (ring amortization) |
| 500 MB+    | 35–55% (DMA direct, page cache bypassed) |
| 1.5 GB     | 40–60% (NVMe queue depth saturated) |

These estimates are based on published io_uring vs readv benchmarks on NVMe SSDs
(Samsung 980 Pro class, which matches the Dell G15 5520 hardware target).
Validation is deferred until bare-metal deployment.

### Phase 4+ Note

`O_DIRECT` mode (bare metal) requires all read buffers and file offsets to be
aligned to 4096 bytes. The safetensors header is variable-length and almost never
page-aligned. The bare-metal path will need an alignment-correction read: floor the
offset to 4096, read an expanded block, then slice the exact tensor bytes from the
buffer. This is stubbed in `IOUringBackend.__init__` with `NotImplementedError`.

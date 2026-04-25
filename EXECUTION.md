# DeltaStream Phase 1 — Execution Flow

## Overview
Convert any HuggingFace model into a delta-compressed format and verify bit-identical reconstruction.

---

## Step 1: Environment Setup

**Script:** `setup_env.sh` / `pip install -r requirements.txt`  
**Input:** None  
**Output:** Python environment with all dependencies installed  
**Validation:** `python -c "import torch, safetensors, transformers; print('OK')"`

---

## Step 2: Model Loading & Inspection

**Module:** `deltastream/core/weight_io.py`  
**Function:** `load_model_weights(model_path: str) -> dict[str, Tensor]`

**Input:**
- `model_path`: HuggingFace repo ID (e.g. `"Qwen/Qwen2-0.5B"`) or local path

**Process:**
1. Detect format: check for `model.safetensors.index.json`, `*.safetensors`, or `*.bin`
2. If remote repo: use `huggingface_hub.snapshot_download()` to local cache
3. If sharded: load `index.json`, determine shard file list and tensor-to-shard mapping
4. Load all tensors to CPU (avoid GPU to prevent VRAM overflow)
5. Detect tied weights (e.g. `lm_head.weight` == `embed_tokens.weight`)

**Output:**
- `weights: dict[str, torch.Tensor]` — all model weights on CPU
- `metadata: dict` — source format, dtype, shard count

---

## Step 3: Layer Topology Discovery

**Module:** `deltastream/core/manifest.py`  
**Function:** `discover_layers(weights: dict) -> LayerTopology`

**Input:** Full weight dictionary from Step 2

**Process:**
1. Identify layer prefix (e.g. `model.layers` for LLaMA, `transformer.h` for GPT)
2. Parse layer indices: extract `[0..N]` from key patterns
3. Group tensors by layer index
4. Identify non-layer tensors: embeddings, lm_head, etc.

**Output:**
```python
LayerTopology(
    prefix="model.layers",
    num_layers=32,
    layer_tensors={0: ["self_attn.q_proj.weight", ...], ...},
    non_layer_keys=["model.embed_tokens.weight", "lm_head.weight", ...]
)
```

---

## Step 4: Delta Encoding

**Module:** `deltastream/core/delta_encoder.py`  
**Function:** `encode_model(weights, topology, output_dir) -> manifest`

**Input:**
- `weights`: full weight dict
- `topology`: LayerTopology from Step 3
- `output_dir`: target directory path

**Process (for each layer pair L, L+1):**
1. Extract layer L tensor dict: `{short_key: tensor}`
2. For L=0 (base layer): save verbatim as `base/layer_00.safetensors`
3. For L>0:
   a. Compute delta: for each tensor key `k`:
      - Cast both prev_weight and curr_weight to float32
      - `delta[k] = curr_weight[k].float() - prev_weight[k].float()`
      - Cast delta back to original dtype
   b. Save delta as `deltas/layer_{L:02d}.delta.safetensors`
4. Save non-layer tensors verbatim to `base/non_layer.safetensors`
5. Build and save `manifest.json`

**Output:**
- `delta_model/base/layer_00.safetensors`
- `delta_model/deltas/layer_01.delta.safetensors` ... `layer_N.delta.safetensors`
- `delta_model/base/non_layer.safetensors`
- `delta_model/manifest.json`

---

## Step 5: Manifest Finalization

**Module:** `deltastream/core/manifest.py`  
**Function:** `write_manifest(manifest, output_dir)`

**Input:** Manifest dict built during Step 4  
**Output:** `delta_model/manifest.json`

Manifest contains:
- Source model path/id
- Layer count, prefix, base layer index
- Per-tensor dtype mapping
- Shard file list in order
- Compression flag
- Checksum (SHA256) of base layer file for integrity validation

---

## Step 6: Delta Decoding (Reconstruction)

**Module:** `deltastream/core/delta_decoder.py`  
**Function:** `reconstruct_all_layers(delta_model_dir) -> dict[int, dict[str, Tensor]]`

**Input:** `delta_model_dir` path

**Process:**
1. Read manifest.json
2. Load base layer (layer 0) verbatim
3. For each subsequent layer:
   a. Load delta file
   b. `reconstructed[k] = prev_layer[k].float() + delta[k].float()` → cast back to dtype
4. Load non-layer tensors verbatim
5. Reconstruct full weight dict (layer-indexed tensors + non-layer tensors)

**Output:** Full reconstructed weight dict `{tensor_name: tensor}`

---

## Step 7: Bit-Identical Verification

**Module:** `deltastream/cli/verify.py`  
**Function:** `verify_reconstruction(original_path, delta_model_dir)`

**Input:**
- `original_path`: original HF model path
- `delta_model_dir`: converted delta model directory

**Process:**
1. Load original weights (Step 2)
2. Reconstruct from delta (Step 6)
3. For every tensor name:
   - `assert torch.equal(original[k], reconstructed[k])`
   - Log pass/fail per tensor
4. Run inference verification:
   a. Load original model with transformers
   b. Load delta-reconstructed model (swap weights in-place)
   c. Run same prompt through both (greedy, temperature=0)
   d. Compare output token IDs: `assert original_ids == reconstructed_ids`

**Output:**
- Tensor-level pass/fail report
- Inference comparison result (PASS/FAIL)
- Summary stats: total tensors, pass count, fail count

---

## Step 8: CLI Entry Points

**Module:** `deltastream/main.py`

```
python -m deltastream convert --model <hf_model_id_or_path> --output <dir>
python -m deltastream verify  --model <hf_model_id_or_path> --delta  <dir>
python -m deltastream info    --delta <dir>
```

---

## Data Flow Diagram

```
[HF Hub / Local Path]
        ↓
  weight_io.py: load_model_weights()
        ↓
  manifest.py: discover_layers()
        ↓
  delta_encoder.py: encode_model()
        ↓
  [delta_model/ directory]
        ↓
  delta_decoder.py: reconstruct_all_layers()
        ↓
  verify.py: verify_reconstruction()
        ↓
  [PASS: bit-identical reconstruction confirmed]
```

---

## Error Handling Per Step

| Step | Error | Handling |
|------|-------|----------|
| 2 | Network failure | Retry 3x with backoff, then raise |
| 2 | Local path missing | Raise with clear message |
| 3 | Unknown layer prefix | Fallback heuristics → prompt user |
| 4 | Tensor shape mismatch | Raise immediately, do not save partial |
| 4 | Disk full | Check free space before, raise if < 2x model size |
| 7 | Verification fails | Print mismatch tensor names, exit code 1 |

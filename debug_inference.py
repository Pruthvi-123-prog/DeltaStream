"""
Debug script to trace why DeltaStream produces empty responses on Windows.
Run with: python debug_inference.py
"""
import torch
import traceback

print("[1] Loading runtime...")
from deltastream.runtime import DeltaStreamRuntime
model = DeltaStreamRuntime("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "delta_tinyllama_1.1b_chat_v1.0")
print(f"[1] Device: {model.device}")

print("\n[2] Testing tokenizer...")
tokenizer = model.tokenizer
test_input = "<|system|>\nYou are helpful.\n<|user|>\nhello\n<|assistant|>\n"
input_ids = tokenizer(test_input, return_tensors="pt")["input_ids"]
print(f"[2] Input shape: {input_ids.shape}, EOS id: {tokenizer.eos_token_id}")

print("\n[3] Testing single layer forward...")
try:
    model._load_layer_weights_to_device(0)
    block = model._layer_module[0]

    # Compute rotary embeddings
    hidden = model.model.get_input_embeddings()(input_ids)
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
    
    rotary_emb = (
        getattr(model.model, "rotary_emb", None)
        or getattr(getattr(model.model, "model", None), "rotary_emb", None)
    )
    position_embeddings = None
    if rotary_emb:
        try:
            position_embeddings = rotary_emb(hidden, position_ids)
            print(f"[3] rotary_emb OK — cos shape: {position_embeddings[0].shape}")
        except Exception as e:
            print(f"[3] rotary_emb FAILED: {e}")
    else:
        print("[3] No rotary_emb found")

    out = block(hidden, attention_mask=None, position_ids=position_ids,
                position_embeddings=position_embeddings, past_key_value=None, use_cache=True)
    hidden_out = out[0] if isinstance(out, tuple) else out
    print(f"[3] Layer 0 output shape: {hidden_out.shape}, all zeros: {hidden_out.abs().max().item() < 1e-6}")
    model._offload_layer_weights(0)
except Exception as e:
    traceback.print_exc()

print("\n[4] Testing lm_head output...")
try:
    lm_head = model.model.lm_head
    print(f"[4] lm_head device: {next(lm_head.parameters()).device}")
    print(f"[4] lm_head is meta: {next(lm_head.parameters()).is_meta}")
except Exception as e:
    print(f"[4] lm_head check failed: {e}")

print("\n[5] Running full generate with 10 tokens...")
try:
    result = model.generate("Hello", max_new_tokens=10)
    print(f"[5] Result: {result}")
except Exception as e:
    traceback.print_exc()

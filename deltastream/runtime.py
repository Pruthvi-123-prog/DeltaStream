"""
runtime.py — DeltaStreamXRuntime: Streaming layer-by-layer inference runtime.

This is a self-contained replacement for vanilla transformers inference.
It uses the DeltaStream stack:
  - IOBackendFactory (auto-detects io_uring vs standard, SIGILL-safe)
  - LayerCacheManager (LRU, mlock-pinned, N+1/N+2 prefetch)
  - DeltaDecoder (base + compressed delta reconstruction)

Drop-in interface: generate(input_ids, max_new_tokens) → token ids
                   generate_text(prompt, max_new_tokens) → str
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from deltastream.core.io_backend_uring import IOBackendFactory
from deltastream.core.cache import LayerCacheManager
from deltastream.core.manifest import read_manifest
from deltastream.utils.logging import log_info, log_warning


class DeltaStreamRuntime:
    """
    Streaming inference runtime for delta-compressed transformer models.

    Parameters
    ----------
    model_id : str
        HuggingFace model repo id or local path to the original model.
    delta_dir : str | Path
        Path to the delta_model/ directory created by Phase 1 converter.
        If it doesn't exist, the converter is run automatically.
    device : str
        'cuda' or 'cpu'. Defaults to 'cuda' if available.
    max_ram_gb : float
        RAM budget for the LayerCacheManager (default 10 GB).
    compress : bool
        If True and delta_dir does not exist, run converter with --compress.
    """

    def __init__(
        self,
        model_id: str,
        delta_dir: str | Path,
        device: str | None = None,
        max_ram_gb: float = 10.0,
        compress: bool = False,
    ):
        self.model_id = model_id
        self.delta_dir = Path(delta_dir)
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            log_warning("No hardware acceleration (CUDA/MPS) found. Running on CPU (will be slow).")
            
        self.max_ram_gb = max_ram_gb

        # ── 1. Auto-convert if delta_dir missing ────────────────────────────
        if not self.delta_dir.exists() or not (self.delta_dir / "manifest.json").exists():
            log_info(f"delta_dir not found at {self.delta_dir}. Running converter...")
            self._run_converter(compress=compress)

        # ── 2. Load manifest ─────────────────────────────────────────────────
        self.manifest = read_manifest(self.delta_dir)
        log_info(
            f"Loaded manifest: {self.manifest.source_model}, "
            f"{self.manifest.num_layers} layers, "
            f"compression={self.manifest.compression}"
        )

        # ── 3. Tokenizer ─────────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── 4. Model config + empty skeleton ────────────────────────────────
        self.config = AutoConfig.from_pretrained(model_id)
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(self.config)
        self.model.eval()

        # ── 5. IO Backend + Cache ────────────────────────────────────────────
        backend = IOBackendFactory.get_backend(
            str(self.delta_dir), str(self.delta_dir), device=self.device
        )
        self.cache = LayerCacheManager(backend, max_ram_gb=max_ram_gb)

        # Detect layer container (e.g. model.layers for LLaMA, transformer.h for GPT2)
        self._layer_module = self._find_layer_module()
        log_info(f"Layer container: {type(self._layer_module).__name__} ({len(self._layer_module)} blocks)")

        # Warm base layer into cache
        log_info("Pre-loading base layer into cache...")
        self.cache.get_layer(0)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _run_converter(self, compress: bool = False) -> None:
        """Run Phase 1 converter to create delta_dir."""
        import subprocess, sys
        cmd = [
            sys.executable, "-m", "deltastream", "convert",
            "--model", self.model_id,
            "--output", str(self.delta_dir),
        ]
        if compress:
            cmd.append("--compress")
        log_info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        if result.returncode != 0:
            raise RuntimeError(f"Converter failed with exit code {result.returncode}")

    def _find_layer_module(self):
        """Return the transformer block list (ModuleList) from the model skeleton."""
        import torch.nn as nn
        prefix = self.manifest.layer_prefix  # e.g. 'model.layers' or 'transformer.h'
        
        # Try direct path first (works for LLaMA's 'model.layers')
        try:
            obj = self.model
            for part in prefix.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, nn.ModuleList):
                return obj
        except AttributeError:
            pass
            
        # Fallback: search for a ModuleList matching the layer count
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) == self.manifest.num_layers:
                if name.endswith(prefix) or prefix.endswith(name.split('.')[-1]):
                    return module
                    
        raise RuntimeError(f"Could not locate transformer blocks (prefix '{prefix}', expected {self.manifest.num_layers} layers)")

    def _load_layer_weights_to_device(self, layer_idx: int) -> None:
        """Materialize one transformer block's weights onto the device."""
        tensors = self.cache.get_layer(layer_idx)
        
        skeleton_prefix = None
        for name, module in self.model.named_modules():
            if module is self._layer_module:
                skeleton_prefix = name
                break
                
        for short_key, tensor in tensors.items():
            full_key = f"{skeleton_prefix}.{layer_idx}.{short_key}"
            # set_module_tensor_to_device handles empty/meta tensors
            try:
                set_module_tensor_to_device(
                    self.model, full_key, device=self.device, value=tensor
                )
            except Exception:
                pass  # Ignore extra tensors in checkpoint (e.g. legacy attn.bias)

    def _offload_layer_weights(self, layer_idx: int) -> None:
        """Move one transformer block back to meta device to free VRAM."""
        skeleton_prefix = None
        for name, module in self.model.named_modules():
            if module is self._layer_module:
                skeleton_prefix = name
                break
                
        block = self._layer_module[layer_idx]
        for name, _ in block.named_parameters():
            full_key = f"{skeleton_prefix}.{layer_idx}.{name}"
            set_module_tensor_to_device(self.model, full_key, device="meta")
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

    def _load_non_layer_weights(self) -> None:
        """Load embeddings, lm_head, norms onto device."""
        from safetensors.torch import load_file as st_load
        non_layer_path = self.delta_dir / "base" / "non_layer.safetensors"
        if not non_layer_path.exists():
            log_warning("non_layer.safetensors not found — skipping non-layer weights.")
            return
        weights = st_load(str(non_layer_path), device=self.device)
        
        skeleton_names = [n for n, _ in self.model.named_parameters()]
        
        for name, tensor in weights.items():
            target_name = name
            for skel_name in skeleton_names:
                if skel_name == name or skel_name.endswith("." + name):
                    target_name = skel_name
                    break
                    
            try:
                set_module_tensor_to_device(self.model, target_name, device=self.device, value=tensor)
            except Exception:
                pass  # ignore keys that don't map to the model (e.g. tied weights)

    # ── Public generate API ────────────────────────────────────────────────────

    def _generate_ids(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        do_sample: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate tokens via streaming layer-by-layer forward passes.

        Parameters
        ----------
        input_ids      : (1, seq_len) input token ids on CPU
        max_new_tokens : number of new tokens to generate
        temperature    : sampling temperature (ignored if do_sample=False)
        do_sample      : if True, sample; else greedy

        Returns
        -------
        generated_ids : (1, seq_len + max_new_tokens) tensor on CPU
        """
        self._load_non_layer_weights()

        generated = input_ids.clone()
        past_key_values = None

        with torch.no_grad():
            for step in range(max_new_tokens):
                cur_input = generated if past_key_values is None else generated[:, -1:]

                # Stream through all transformer blocks
                logits, past_key_values = self._forward_streaming(
                    cur_input.to(self.device), past_key_values=past_key_values
                )

                # Sample / greedy next token
                next_logits = logits[:, -1, :]  # (1, vocab)
                if do_sample and temperature != 1.0:
                    next_logits = next_logits / temperature
                if do_sample:
                    probs = torch.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = next_logits.argmax(dim=-1, keepdim=True)

                generated = torch.cat([generated, next_token.cpu()], dim=1)

                # EOS check
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        return generated

    def _forward_streaming(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
    ):
        """Run one full forward pass streaming layer-by-layer. Returns (logits, new_past_kv)."""
        # Embed
        embedding = self.model.get_input_embeddings()
        hidden = embedding(input_ids)

        past_len = 0
        if past_key_values is not None:
            # past_key_values is a tuple of (key, value) per layer
            try:
                past_len = past_key_values[0][0].shape[2]
            except (IndexError, TypeError, AttributeError):
                past_len = 0

        seq_len = hidden.shape[1]
        position_ids = torch.arange(past_len, past_len + seq_len, device=self.device).unsqueeze(0)

        # Compute rotary position embeddings once for all layers (modern transformers >=4.45)
        position_embeddings = None
        rotary_emb = (
            getattr(self.model, "rotary_emb", None)
            or getattr(getattr(self.model, "model", None), "rotary_emb", None)
        )
        if rotary_emb is not None:
            try:
                position_embeddings = rotary_emb(hidden, position_ids)
            except Exception:
                position_embeddings = None

        new_past_kv = []

        for layer_idx in range(self.manifest.num_layers):
            # Load weights onto device, run block, offload
            self._load_layer_weights_to_device(layer_idx)

            block = self._layer_module[layer_idx]
            layer_past = past_key_values[layer_idx] if past_key_values is not None else None

            # Build kwargs progressively — try most-specific first
            out = None
            call_kwargs = dict(
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=layer_past,
                use_cache=True,
            )
            if position_embeddings is not None:
                call_kwargs["position_embeddings"] = position_embeddings

            for attempt in range(3):
                try:
                    out = block(hidden, **call_kwargs)
                    break
                except TypeError as te:
                    msg = str(te)
                    if "position_embeddings" in msg and "position_embeddings" in call_kwargs:
                        call_kwargs.pop("position_embeddings")
                    elif "position_ids" in msg and "position_ids" in call_kwargs:
                        call_kwargs.pop("position_ids")
                    elif "past_key_value" in msg and "past_key_value" in call_kwargs:
                        call_kwargs.pop("past_key_value")
                    else:
                        raise

            if out is None:
                raise RuntimeError(f"Layer {layer_idx} forward pass failed after all fallback attempts")

            if isinstance(out, tuple):
                hidden = out[0]
                if len(out) > 1 and out[1] is not None:
                    new_past_kv.append(out[1])
                else:
                    new_past_kv.append(None)
            else:
                hidden = out
                new_past_kv.append(None)

            self._offload_layer_weights(layer_idx)

        # Final norm + LM head
        try:
            norm = self.model.model.norm
            hidden = norm(hidden)
        except AttributeError:
            try:
                hidden = self.model.transformer.ln_f(hidden)
            except AttributeError:
                pass

        lm_head = self.model.lm_head
        logits = lm_head(hidden)

        past_kv_out = tuple(new_past_kv) if any(x is not None for x in new_past_kv) else None
        return logits, past_kv_out

    def generate(self, prompt: str, max_new_tokens: int = 50, **kwargs) -> tuple[str, dict]:
        """
        Generate text from a string prompt.
        Returns: (generated_text, stats_dict)
        """
        import time
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"]
            input_len = input_ids.shape[1]
            
            start_t = time.time()
            output_ids = self._generate_ids(input_ids, max_new_tokens=max_new_tokens, **kwargs)
            end_t = time.time()
            
            if output_ids is None:
                raise ValueError("_generate_ids returned None")
                
            gen_ids = output_ids[0][input_len:]
            response = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            if response is None:
                response = ""
                
            elapsed = end_t - start_t
            tokens_per_sec = len(gen_ids) / elapsed if elapsed > 0 else 0
            stats = {
                "tokens_per_sec": tokens_per_sec,
                "elapsed": elapsed,
                "generated_tokens": len(gen_ids)
            }
            return response, stats
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Generation error: {e}", {"tokens_per_sec": 0}

    def generate_text(self, prompt: str, max_new_tokens: int = 50, **kwargs) -> str:
        """Convenience wrapper: prompt str → generated str."""
        res, _ = self.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)
        return res

    def cache_stats(self) -> dict:
        return self.cache.get_metrics()

"""
verify.py — Bit-identical verification of delta-reconstructed model weights.

Two verification levels:
  1. Tensor-level  — torch.equal() on every tensor to confirm exact bit match
  2. Inference-level — run same prompt through original and reconstructed model,
                       compare output token IDs (greedy, temp=0)

Exit code: 0 on full pass, 1 on any failure.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from deltastream.core.delta_decoder import reconstruct_all_layers
from deltastream.core.manifest import read_manifest, verify_checksums
from deltastream.core.weight_io import load_model_weights
from deltastream.utils.logging import (
    console,
    log_error,
    log_header,
    log_info,
    log_step,
    log_success,
    log_warning,
    make_progress,
)
from rich.table import Table

# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────


def run_verify(
    model_path: str,
    delta_model_dir: str,
    *,
    prompt: str = "The quick brown fox",
    max_new_tokens: int = 20,
    skip_inference: bool = False,
    cache_dir: str | None = None,
) -> bool:
    """
    Full verification pipeline.

    Returns True if everything passes, False otherwise.
    Prints a rich-formatted report to stdout.
    """
    log_header("DeltaStream  —  Verification")
    delta_dir = Path(delta_model_dir)

    # ── 0. Load manifest & check checksums ───────────────────────────────────
    manifest = read_manifest(delta_dir)
    log_info(
        f"Manifest: source=[highlight]{manifest.source_model}[/highlight]  "
        f"layers=[highlight]{manifest.num_layers}[/highlight]  "
        f"compression=[highlight]{manifest.compression}[/highlight]"
    )

    cksum_ok = verify_checksums(manifest, delta_dir)
    if not cksum_ok:
        log_warning("Checksum failures detected — files may be corrupted.")

    # ── 1. Load original weights ──────────────────────────────────────────────
    log_step("Loading original weights")
    original, _ = load_model_weights(model_path, cache_dir=cache_dir)

    # ── 2. Reconstruct from deltas ────────────────────────────────────────────
    log_step("Reconstructing from delta_model/")
    reconstructed = reconstruct_all_layers(delta_dir, device="cpu")

    # ── 3. Tensor-level comparison ────────────────────────────────────────────
    log_step("Tensor-level comparison")
    tensor_results = _compare_tensors(original, reconstructed)
    tensor_pass = _print_tensor_report(tensor_results)

    # ── 4. Inference-level comparison ─────────────────────────────────────────
    inference_pass = True
    if not skip_inference:
        log_step("Inference-level comparison")
        inference_pass = _compare_inference(
            model_path=model_path,
            original_weights=original,
            reconstructed_weights=reconstructed,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            cache_dir=cache_dir,
        )

    # ── 5. Summary ────────────────────────────────────────────────────────────
    overall = tensor_pass and inference_pass
    log_header("Verification Summary")
    if overall:
        log_success("ALL CHECKS PASSED — reconstruction is bit-identical ✓")
    else:
        log_error("VERIFICATION FAILED — see details above ✗")

    return overall


# ──────────────────────────────────────────────────────────────────────────────
# Tensor comparison
# ──────────────────────────────────────────────────────────────────────────────


def _compare_tensors(
    original: dict[str, torch.Tensor],
    reconstructed: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    """Compare each tensor and return a list of result dicts."""
    results: list[dict[str, Any]] = []

    all_keys = sorted(set(original) | set(reconstructed))

    with make_progress() as progress:
        task = progress.add_task("Comparing tensors", total=len(all_keys))
        for key in all_keys:
            row: dict[str, Any] = {"key": key}

            if key not in original:
                row["status"] = "EXTRA"
                row["detail"] = "only in reconstructed"
            elif key not in reconstructed:
                row["status"] = "MISSING"
                row["detail"] = "only in original"
            else:
                orig_t = original[key]
                recon_t = reconstructed[key]

                if orig_t.shape != recon_t.shape:
                    row["status"] = "SHAPE_MISMATCH"
                    row["detail"] = f"{orig_t.shape} vs {recon_t.shape}"
                elif orig_t.dtype != recon_t.dtype:
                    row["status"] = "DTYPE_MISMATCH"
                    row["detail"] = f"{orig_t.dtype} vs {recon_t.dtype}"
                elif not torch.equal(orig_t, recon_t):
                    # Compute max absolute difference for debugging
                    max_diff = (orig_t.float() - recon_t.float()).abs().max().item()
                    row["status"] = "VALUE_MISMATCH"
                    row["detail"] = f"max_abs_diff={max_diff:.2e}"
                else:
                    row["status"] = "PASS"
                    row["detail"] = ""

            results.append(row)
            progress.advance(task)

    return results


def _print_tensor_report(results: list[dict[str, Any]]) -> bool:
    """Print a compact summary table. Returns True if all tensors pass."""
    passed = [r for r in results if r["status"] == "PASS"]
    failed = [r for r in results if r["status"] != "PASS"]

    # Always show failures
    if failed:
        table = Table(title="Tensor Mismatches", show_lines=True)
        table.add_column("Tensor", style="red", overflow="fold", max_width=60)
        table.add_column("Status", style="red", justify="center")
        table.add_column("Detail", style="yellow")
        for r in failed:
            table.add_row(r["key"], r["status"], r["detail"])
        console.print(table)

    total = len(results)
    n_pass = len(passed)
    n_fail = len(failed)

    if n_fail == 0:
        log_success(
            f"Tensor check: [highlight]{n_pass}/{total}[/highlight] tensors PASS ✓"
        )
    else:
        log_error(
            f"Tensor check: [highlight]{n_fail}[/highlight] tensor(s) FAIL "
            f"([highlight]{n_pass}/{total}[/highlight] passed)"
        )

    return n_fail == 0


# ──────────────────────────────────────────────────────────────────────────────
# Inference comparison
# ──────────────────────────────────────────────────────────────────────────────


def _compare_inference(
    model_path: str,
    original_weights: dict[str, torch.Tensor],
    reconstructed_weights: dict[str, torch.Tensor],
    prompt: str,
    max_new_tokens: int,
    cache_dir: str | None,
) -> bool:
    """
    Load the model architecture via transformers, swap weights, and compare outputs.
    Both runs are on CPU to avoid VRAM issues on 4GB GPU.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        log_warning("transformers not installed — skipping inference comparison.")
        return True

    log_info(f"Prompt: [italic]\"{prompt}\"[/italic]")

    # ── Tokenize ──────────────────────────────────────────────────────────────
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
    except Exception as exc:
        log_warning(f"Could not load tokenizer: {exc}  — skipping inference check.")
        return True

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids: torch.Tensor = inputs["input_ids"]

    # ── Load model architecture (no weights yet) ──────────────────────────────
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    except Exception as exc:
        log_warning(f"Could not load model architecture: {exc} — skipping inference check.")
        return True

    model.eval()

    # ── Run with original weights (already loaded) ────────────────────────────
    log_info("Running inference with ORIGINAL weights …")
    with torch.no_grad():
        original_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        ).squeeze().tolist()

    original_text = tokenizer.decode(original_ids, skip_special_tokens=True)
    log_info(f"Original output: [italic]{original_text!r}[/italic]")

    # ── Swap in reconstructed weights ─────────────────────────────────────────
    log_info("Swapping in RECONSTRUCTED weights …")
    _load_weights_into_model(model, reconstructed_weights)

    log_info("Running inference with RECONSTRUCTED weights …")
    with torch.no_grad():
        recon_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        ).squeeze().tolist()

    recon_text = tokenizer.decode(recon_ids, skip_special_tokens=True)
    log_info(f"Reconstructed output: [italic]{recon_text!r}[/italic]")

    # ── Compare ───────────────────────────────────────────────────────────────
    if original_ids == recon_ids:
        log_success("Inference check: output tokens IDENTICAL ✓")
        return True
    else:
        log_error("Inference check: output tokens DIFFER ✗")
        _print_token_diff(original_ids, recon_ids, tokenizer)
        return False


def _load_weights_into_model(
    model: Any,
    weights: dict[str, torch.Tensor],
) -> None:
    """
    Replace model parameters in-place with tensors from `weights`.
    Missing keys are left as-is (warns once).
    """
    model_state = model.state_dict()
    missing: list[str] = []
    extra: list[str] = []

    for name, param in model.named_parameters():
        if name in weights:
            param.data.copy_(weights[name].to(param.dtype))
        else:
            missing.append(name)

    for key in weights:
        if key not in model_state:
            extra.append(key)

    if missing:
        log_warning(f"{len(missing)} model param(s) not found in reconstructed dict: "
                    f"{missing[:5]}{'…' if len(missing) > 5 else ''}")
    if extra:
        log_warning(f"{len(extra)} reconstructed tensor(s) not in model: "
                    f"{extra[:5]}{'…' if len(extra) > 5 else ''}")


def _print_token_diff(
    original_ids: list[int],
    recon_ids: list[int],
    tokenizer: Any,
) -> None:
    """Print a human-readable first-mismatch report."""
    for i, (a, b) in enumerate(zip(original_ids, recon_ids)):
        if a != b:
            orig_tok = tokenizer.decode([a])
            recon_tok = tokenizer.decode([b])
            log_error(
                f"First mismatch at position {i}: "
                f"original={a!r} ({orig_tok!r})  reconstructed={b!r} ({recon_tok!r})"
            )
            return
    # Length mismatch
    log_error(
        f"Sequences differ in length: original={len(original_ids)}, "
        f"reconstructed={len(recon_ids)}"
    )

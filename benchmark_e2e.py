"""
benchmark_e2e.py — DeltaStream vs Vanilla Transformers: publishable comparison table.

Usage:
    python benchmark_e2e.py --model gpt2
    python benchmark_e2e.py --model gpt2 --compress
    python benchmark_e2e.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --runs 3
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import time
from pathlib import Path

import psutil

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("❌ ERROR: Required packages not found. Did you forget to activate the virtual environment?")
    print("👉 Run: source .venv/bin/activate")
    exit(1)

from deltastream.runtime import DeltaStreamXRuntime
from deltastream.utils.logging import log_info, log_warning

# ─── Prompts ─────────────────────────────────────────────────────────────────

PROMPTS = [
    "The capital of France is",
    "Machine learning is a field of",
    "The best way to learn programming is",
]

# ─── RAM / VRAM measurement ──────────────────────────────────────────────────

def _ram_mb() -> float:
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024 * 1024)


def _vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def _reset_vram():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ─── Vanilla baseline ─────────────────────────────────────────────────────────

def run_vanilla_baseline(model_id: str, prompts: list[str], max_new_tokens: int = 20):
    """Run vanilla transformers (all weights in RAM, no streaming)."""
    log_info("Loading vanilla model (all weights in RAM)...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()
    load_time = time.perf_counter() - t0
    ram_after_load = _ram_mb()
    log_info(f"  Load time: {load_time:.1f}s, RAM: {ram_after_load:.0f} MB")

    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        _reset_vram()
        ram_before = _ram_mb()

        t_start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        t_end = time.perf_counter()

        elapsed = t_end - t_start
        new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
        tok_per_sec = new_tokens / elapsed if elapsed > 0 else 0
        ttft = elapsed / new_tokens if new_tokens > 0 else elapsed
        peak_ram = _ram_mb()
        peak_vram = _vram_mb()
        text = tokenizer.decode(out[0], skip_special_tokens=True)

        results.append({
            "tokens_per_sec": tok_per_sec,
            "ttft_sec": ttft,
            "peak_ram_mb": peak_ram,
            "peak_vram_mb": peak_vram,
            "elapsed_sec": elapsed,
            "output": text[:80],
        })
        log_info(f"  Prompt: '{prompt[:30]}...' → {tok_per_sec:.3f} tok/s | TTFT {ttft:.3f}s")

    # cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results, load_time


# ─── DeltaStream runtime ────────────────────────────────────────────────────────

def run_deltastreamx(
    model_id: str,
    delta_dir: str,
    prompts: list[str],
    max_new_tokens: int = 20,
    compress: bool = False,
    max_ram_gb: float = 10.0,
):
    """Run DeltaStream streaming runtime."""
    log_info(f"Initializing DeltaStreamXRuntime (compress={compress})...")
    t0 = time.perf_counter()
    runtime = DeltaStreamXRuntime(
        model_id=model_id,
        delta_dir=delta_dir,
        max_ram_gb=max_ram_gb,
        compress=compress,
    )
    load_time = time.perf_counter() - t0
    log_info(f"  Init time: {load_time:.1f}s")

    results = []
    for i, prompt in enumerate(prompts):
        inputs = runtime.tokenizer(prompt, return_tensors="pt")
        _reset_vram()

        t_start = time.perf_counter()
        out_ids = runtime.generate(inputs["input_ids"], max_new_tokens=max_new_tokens)
        t_end = time.perf_counter()

        elapsed = t_end - t_start
        new_tokens = out_ids.shape[1] - inputs["input_ids"].shape[1]
        tok_per_sec = new_tokens / elapsed if elapsed > 0 else 0
        ttft = elapsed / new_tokens if new_tokens > 0 else elapsed
        peak_ram = _ram_mb()
        peak_vram = _vram_mb()
        text = runtime.tokenizer.decode(out_ids[0], skip_special_tokens=True)

        cache_stats = runtime.cache_stats()

        results.append({
            "tokens_per_sec": tok_per_sec,
            "ttft_sec": ttft,
            "peak_ram_mb": peak_ram,
            "peak_vram_mb": peak_vram,
            "elapsed_sec": elapsed,
            "cache_hit_rate": cache_stats["hit_rate"],
            "output": text[:80],
        })
        log_info(
            f"  Prompt: '{prompt[:30]}...' → {tok_per_sec:.3f} tok/s | "
            f"TTFT {ttft:.3f}s | Cache hit {cache_stats['hit_rate']:.0%}"
        )

    return results, load_time


# ─── Table printer ───────────────────────────────────────────────────────────

def _median(values: list[float]) -> float:
    return statistics.median(values)


def print_comparison_table(
    vanilla_results: list[dict],
    deltastreamx_results: list[dict],
    vanilla_load: float,
    deltastreamx_load: float,
    model_id: str,
    compress: bool,
):
    v_tps = _median([r["tokens_per_sec"] for r in vanilla_results])
    a_tps = _median([r["tokens_per_sec"] for r in deltastreamx_results])
    v_ttft = _median([r["ttft_sec"] for r in vanilla_results])
    a_ttft = _median([r["ttft_sec"] for r in deltastreamx_results])
    v_ram = _median([r["peak_ram_mb"] for r in vanilla_results])
    a_ram = _median([r["peak_ram_mb"] for r in deltastreamx_results])
    v_vram = _median([r["peak_vram_mb"] for r in vanilla_results])
    a_vram = _median([r["peak_vram_mb"] for r in deltastreamx_results])

    speedup = a_tps / v_tps if v_tps > 0 else float("nan")
    ttft_ratio = v_ttft / a_ttft if a_ttft > 0 else float("nan")

    compress_tag = " + zstd:1" if compress else ""

    lines = [
        "",
        "=" * 72,
        f"  DeltaStream vs Vanilla Transformers — {model_id}",
        "=" * 72,
        f"  {'Metric':<30} {'Vanilla':>12} {'DeltaStream' + compress_tag:>16}",
        "  " + "-" * 60,
        f"  {'Tokens / second':<30} {v_tps:>12.3f} {a_tps:>16.3f}",
        f"  {'Time to first token (s)':<30} {v_ttft:>12.3f} {a_ttft:>16.3f}",
        f"  {'Peak RAM (MB)':<30} {v_ram:>12.0f} {a_ram:>16.0f}",
        f"  {'Peak VRAM (MB)':<30} {v_vram:>12.0f} {a_vram:>16.0f}",
        f"  {'Model load time (s)':<30} {vanilla_load:>12.1f} {deltastreamx_load:>16.1f}",
        "  " + "-" * 60,
        f"  {'Speedup (tok/s)':<30} {'':>12} {speedup:>15.2f}x",
        f"  {'TTFT improvement':<30} {'':>12} {ttft_ratio:>14.2f}x faster",
        "=" * 72,
        "",
    ]
    for line in lines:
        print(line)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DeltaStream vs Vanilla benchmark")
    parser.add_argument("--model", default="gpt2", help="HF model id or local path")
    parser.add_argument("--delta-dir", default=None, help="Delta model directory (auto-derived if not set)")
    parser.add_argument("--compress", action="store_true", help="Use zstd:1 compressed deltas")
    parser.add_argument("--max-new-tokens", type=int, default=20, help="Tokens to generate per prompt")
    parser.add_argument("--runs", type=int, default=3, help="Number of prompts to run (max 3)")
    parser.add_argument("--max-ram-gb", type=float, default=10.0, help="RAM budget for cache (GB)")
    parser.add_argument("--skip-vanilla", action="store_true", help="Skip vanilla baseline (faster)")
    args = parser.parse_args()

    model_id = args.model
    delta_dir = args.delta_dir or f"delta_{model_id.split('/')[-1].lower().replace('-', '_')}"
    prompts = PROMPTS[: args.runs]

    log_info(f"Model: {model_id}")
    log_info(f"Delta dir: {delta_dir}")
    log_info(f"Compress: {args.compress}")
    log_info(f"Prompts: {len(prompts)}, max_new_tokens={args.max_new_tokens}")
    print()

    # Vanilla baseline
    vanilla_results, vanilla_load = [], 0.0
    if not args.skip_vanilla:
        log_info("─── VANILLA BASELINE ─────────────────────────────────────────")
        vanilla_results, vanilla_load = run_vanilla_baseline(
            model_id, prompts, args.max_new_tokens
        )

    print()

    # DeltaStream
    log_info("─── DeltaStream RUNTIME ─────────────────────────────────────────")
    deltastreamx_results, deltastreamx_load = run_deltastreamx(
        model_id, delta_dir, prompts,
        max_new_tokens=args.max_new_tokens,
        compress=args.compress,
        max_ram_gb=args.max_ram_gb,
    )

    # Print table
    if not args.skip_vanilla and vanilla_results:
        print_comparison_table(
            vanilla_results, deltastreamx_results,
            vanilla_load, deltastreamx_load,
            model_id, args.compress,
        )
    else:
        log_info(f"DeltaStream median: {statistics.median([r['tokens_per_sec'] for r in deltastreamx_results]):.3f} tok/s")


if __name__ == "__main__":
    main()

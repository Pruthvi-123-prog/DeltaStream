"""
main.py — DeltaStream CLI entry point.

Usage
-----
  python -m deltastream convert --model <hf_id_or_path> --output <dir>
  python -m deltastream verify  --model <hf_id_or_path> --delta  <dir>
  python -m deltastream info    --delta <dir>

Run any subcommand with --help for full option list.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from deltastream import __version__
from deltastream.utils.logging import console, log_error


# ──────────────────────────────────────────────────────────────────────────────
# Sub-command handlers
# ──────────────────────────────────────────────────────────────────────────────


def _cmd_convert(args: argparse.Namespace) -> int:
    from deltastream.cli.convert import run_convert

    # --compress is shorthand for --compression zstd:1
    compression = args.compression
    if getattr(args, "compress", False) and compression == "none":
        compression = "zstd:1"

    run_convert(
        model_path=args.model,
        output_dir=args.output,
        compression=compression,
        cache_dir=args.cache_dir or None,
        layer_prefix=args.layer_prefix or None,
    )
    return 0



def _cmd_verify(args: argparse.Namespace) -> int:
    from deltastream.cli.verify import run_verify

    passed = run_verify(
        model_path=args.model,
        delta_model_dir=args.delta,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        skip_inference=args.skip_inference,
        cache_dir=args.cache_dir or None,
    )
    return 0 if passed else 1


def _cmd_info(args: argparse.Namespace) -> int:
    """Print manifest info without loading weights."""
    from deltastream.core.manifest import read_manifest
    from deltastream.utils.logging import log_header, log_info
    from rich.table import Table

    delta_dir = Path(args.delta)
    manifest = read_manifest(delta_dir)

    log_header("DeltaStream  —  Model Info")
    log_info(f"Source model : [highlight]{manifest.source_model}[/highlight]")
    log_info(f"Version      : [highlight]{manifest.deltastreamx_version}[/highlight]")
    log_info(f"Layers       : [highlight]{manifest.num_layers}[/highlight]")
    log_info(f"Layer prefix : [highlight]{manifest.layer_prefix}[/highlight]")
    log_info(f"Compression  : [highlight]{manifest.compression}[/highlight]")
    log_info(f"Base files   : [highlight]{len(manifest.base_files)}[/highlight]")
    log_info(f"Delta files  : [highlight]{len(manifest.delta_files)}[/highlight]")
    log_info(f"Tied pairs   : [highlight]{len(manifest.tied_pairs)}[/highlight]")
    log_info(f"Tensors (dtype map): [highlight]{len(manifest.tensor_dtypes)}[/highlight]")

    # Dtype breakdown
    dtype_counts: dict[str, int] = {}
    for dt in manifest.tensor_dtypes.values():
        dtype_counts[dt] = dtype_counts.get(dt, 0) + 1

    table = Table(title="Dtype Distribution", show_header=True)
    table.add_column("Dtype", style="cyan")
    table.add_column("Count", justify="right", style="green")
    for dtype, count in sorted(dtype_counts.items()):
        table.add_row(dtype, str(count))
    console.print(table)

    # File sizes
    size_table = Table(title="File Sizes", show_header=True)
    size_table.add_column("File", style="cyan", overflow="fold")
    size_table.add_column("Size (MB)", justify="right", style="yellow")
    total_bytes = 0
    for rel_path in manifest.base_files + manifest.delta_files:
        fp = delta_dir / rel_path
        size = fp.stat().st_size if fp.exists() else 0
        total_bytes += size
        size_table.add_row(rel_path, f"{size / 1_048_576:.2f}")
    size_table.add_row("[bold]TOTAL[/bold]", f"[bold]{total_bytes / 1_048_576:.2f}[/bold]")
    console.print(size_table)

    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deltastream",
        description=(
            "DeltaStream — Zero-accuracy-loss LLM inference accelerator.\n"
            "Phase 1: Delta-compress any HuggingFace model and verify bit-identical reconstruction."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", "-V", action="version", version=f"deltastream {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ─── convert ───────────────────────────────────────────────────────────────
    p_convert = subparsers.add_parser(
        "convert",
        help="Convert a HuggingFace model to delta-compressed format.",
        description=(
            "Loads all model weights from a HuggingFace repo or local directory,\n"
            "computes per-layer deltas (float32 arithmetic, zero accuracy loss),\n"
            "and writes the result to --output as a delta_model/ directory.\n\n"
            "Example:\n"
            "  python -m deltastream convert --model gpt2 --output ./delta_gpt2\n"
            "  python -m deltastream convert --model /path/to/llama --output ./delta_llama"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_convert.add_argument(
        "--model", "-m",
        required=True,
        help="HuggingFace model repo ID (e.g. 'gpt2') or local directory path.",
    )
    p_convert.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for the delta_model/ (created if absent).",
    )
    p_convert.add_argument(
        "--compression",
        default="none",
        choices=["none", "zstd:1"],
        help="Compression mode for delta files. 'zstd:1' = zstd level 1 (fastest decompress). (default: none)",
    )
    p_convert.add_argument(
        "--compress",
        action="store_true",
        help="Shorthand for --compression zstd:1. Compresses delta files with zstd level 1.",
    )
    p_convert.add_argument(
        "--layer-prefix",
        default=None,
        dest="layer_prefix",
        help=(
            "Override auto-detected layer key prefix "
            "(e.g. 'model.layers', 'transformer.h'). "
            "Auto-detected if not specified."
        ),
    )
    p_convert.add_argument(
        "--cache-dir",
        default=None,
        dest="cache_dir",
        help="HuggingFace Hub cache directory override.",
    )
    p_convert.set_defaults(func=_cmd_convert)


    # ─── verify ────────────────────────────────────────────────────────────────
    p_verify = subparsers.add_parser(
        "verify",
        help="Verify bit-identical reconstruction from delta_model/.",
        description=(
            "Loads the original model and the delta-compressed copy, then:\n"
            "  1. Compares every tensor with torch.equal()\n"
            "  2. Runs the same prompt through both and compares token IDs.\n\n"
            "Exit code 0 = all checks pass. Exit code 1 = any failure.\n\n"
            "Example:\n"
            "  python -m deltastream verify --model gpt2 --delta ./delta_gpt2\n"
            "  python -m deltastream verify --model gpt2 --delta ./delta_gpt2 "
            "--prompt 'Hello world' --max-new-tokens 30"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_verify.add_argument(
        "--model", "-m",
        required=True,
        help="Original HuggingFace model repo ID or local path.",
    )
    p_verify.add_argument(
        "--delta", "-d",
        required=True,
        help="Path to the delta_model/ directory produced by 'convert'.",
    )
    p_verify.add_argument(
        "--prompt",
        default="The quick brown fox jumps over the lazy dog.",
        help="Prompt text for inference comparison. (default: 'The quick brown fox …')",
    )
    p_verify.add_argument(
        "--max-new-tokens",
        type=int,
        default=20,
        dest="max_new_tokens",
        help="Max tokens to generate for inference comparison. (default: 20)",
    )
    p_verify.add_argument(
        "--skip-inference",
        action="store_true",
        dest="skip_inference",
        help="Only run tensor-level comparison; skip inference comparison.",
    )
    p_verify.add_argument(
        "--cache-dir",
        default=None,
        dest="cache_dir",
        help="HuggingFace Hub cache directory override.",
    )
    p_verify.set_defaults(func=_cmd_verify)

    # ─── info ──────────────────────────────────────────────────────────────────
    p_info = subparsers.add_parser(
        "info",
        help="Display manifest metadata for a delta_model/ directory.",
        description=(
            "Reads and pretty-prints the manifest.json from a delta_model/ "
            "directory without loading any weights.\n\n"
            "Example:\n"
            "  python -m deltastream info --delta ./delta_gpt2"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_info.add_argument(
        "--delta", "-d",
        required=True,
        help="Path to the delta_model/ directory.",
    )
    p_info.set_defaults(func=_cmd_info)

    return parser


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        exit_code = args.func(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)
    except Exception as exc:
        log_error(f"{type(exc).__name__}: {exc}")
        # Re-raise so the traceback is visible for debugging
        raise

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

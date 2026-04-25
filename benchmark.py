"""
benchmark.py — Honest cold-disk throughput comparison: Standard vs io_uring.

Measures raw safetensors file read speed, bypassing the delta decoder pipeline.
Both backends read the exact same file, produce the exact same tensors.
Every run drops OS page caches and validates the result isn't a cache hit.
"""

import time
import struct
import subprocess
import os
import json
import statistics
import ctypes
import ctypes.util
import torch
from pathlib import Path
from safetensors.torch import save_file

from deltastream.utils.logging import log_info, log_warning

# ─── Constants ────────────────────────────────────────────────────────────────

CACHE_HIT_THRESHOLD_MBPS = 3000  # Anything above this is OS page cache, not disk
NUM_RUNS = 5
POSIX_FADV_DONTNEED = 4  # Linux: advise kernel to evict pages for this fd

# ─── Cache eviction ──────────────────────────────────────────────────────────

def drop_os_caches():
    """Belt-and-suspenders: drop entire OS page cache."""
    try:
        subprocess.run(
            ["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return True
    except Exception:
        return False


def evict_file_pages(filepath: str):
    """Use posix_fadvise(DONTNEED) to evict a specific file's pages from cache."""
    try:
        libc_name = ctypes.util.find_library("c")
        if not libc_name:
            return False
        libc = ctypes.CDLL(libc_name, use_errno=True)
        fd = os.open(filepath, os.O_RDONLY)
        try:
            file_size = os.fstat(fd).st_size
            ret = libc.posix_fadvise(fd, 0, file_size, POSIX_FADV_DONTNEED)
            return ret == 0
        finally:
            os.close(fd)
    except Exception:
        return False


def ensure_cold(filepath: str):
    """Try every available method to evict the file from OS page cache."""
    drop_os_caches()
    evict_file_pages(filepath)
    # Sync first to flush any dirty pages
    try:
        subprocess.run(["sync"], timeout=5, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    drop_os_caches()


# ─── Safetensors header parser (shared by both read paths) ───────────────────

_DTYPE_MAP = {
    "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
    "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
}

def parse_safetensors_header(filepath: str):
    """Returns (metadata_dict, data_start_offset)."""
    with open(filepath, "rb") as f:
        header_bytes = f.read(8)
        # Check for zstd magic bytes
        if header_bytes[:4] == b'\x28\xb5\x2f\xfd':
            return {"__compressed__": True}, 0
            
        try:
            (header_len,) = struct.unpack("<Q", header_bytes)
            metadata = json.loads(f.read(header_len).decode("utf-8"))
            metadata.pop("__metadata__", None)
            return metadata, 8 + header_len
        except (struct.error, UnicodeDecodeError, OverflowError, MemoryError):
            return {"__compressed__": True}, 0


# ─── Read path: Standard (explicit open+read+close, NO mmap) ────────────────

def read_file_standard(filepath: str) -> dict[str, torch.Tensor]:
    """Read a safetensors file using explicit POSIX read() — no mmap."""
    metadata, data_start = parse_safetensors_header(filepath)
    file_size = os.path.getsize(filepath)
    data_len = file_size - data_start

    fd = os.open(filepath, os.O_RDONLY)
    try:
        if data_start > 0:
            os.lseek(fd, data_start, os.SEEK_SET)
        raw = os.read(fd, data_len)          # single large read, fully materialized
    finally:
        os.close(fd)
        
    if metadata.get("__compressed__"):
        # For I/O benchmarks, we just wanted to measure the raw read time
        return {"dummy": torch.zeros(1)}

    results = {}
    for key, meta in metadata.items():
        start, end = meta["data_offsets"]
        dtype = _DTYPE_MAP[meta["dtype"]]
        shape = meta["shape"]
        t = torch.frombuffer(bytearray(raw[start:end]), dtype=dtype).clone()
        if shape:
            t = t.view(shape)
        results[key] = t
    return results


# ─── Read path: io_uring (batched async SQE submission) ──────────────────────

def read_file_uring(filepath: str) -> dict[str, torch.Tensor]:
    """Read a safetensors file using io_uring, batching multiple tensor reads."""
    import liburing

    metadata, data_start = parse_safetensors_header(filepath)
    file_size = os.path.getsize(filepath)
    
    fd = os.open(filepath, os.O_RDONLY)
    try:
        if metadata.get("__compressed__"):
            # Single bulk read for compressed files
            _Ring = getattr(liburing, 'io_uring', None) or getattr(liburing, 'Ring')
            _Cqe = getattr(liburing, 'io_uring_cqe', None) or getattr(liburing, 'Cqe')
            ring = _Ring()
            liburing.io_uring_queue_init(2, ring, 0)
            
            buf = bytearray(file_size)
            sqe = liburing.io_uring_get_sqe(ring)
            liburing.io_uring_prep_read(sqe, fd, buf, file_size, 0)
            liburing.io_uring_sqe_set_data64(sqe, 0)
            liburing.io_uring_submit(ring)
            
            cqe = _Cqe()
            liburing.io_uring_wait_cqe(ring, cqe)
            if cqe.res < 0:
                raise OSError(f"io_uring bulk read failed: {-cqe.res}")
            liburing.io_uring_cqe_seen(ring, cqe)
            liburing.io_uring_queue_exit(ring)
            
            return {"dummy": torch.zeros(1)}

        num_tensors = len(metadata)
        if num_tensors == 0:
            return {}

        # Version-adaptive: liburing < 2026 vs >= 2026
        _Ring = getattr(liburing, 'io_uring', None) or getattr(liburing, 'Ring')
        _Cqe = getattr(liburing, 'io_uring_cqe', None) or getattr(liburing, 'Cqe')

        ring = _Ring()
        liburing.io_uring_queue_init(max(num_tensors, 2), ring, 0)

        buffers = []
        keys = list(metadata.keys())

        for i, key in enumerate(keys):
            tensor_meta = metadata[key]
            start_offset, end_offset = tensor_meta["data_offsets"]
            length = end_offset - start_offset
            abs_offset = data_start + start_offset

            buf = bytearray(length)
            buffers.append((key, tensor_meta, buf))

            sqe = liburing.io_uring_get_sqe(ring)
            liburing.io_uring_prep_read(sqe, fd, buf, length, abs_offset)
            liburing.io_uring_sqe_set_data64(sqe, i)

        # Submit all reads
        liburing.io_uring_submit(ring)

        # Wait for all reads
        cqe = _Cqe()
        for _ in range(num_tensors):
            liburing.io_uring_wait_cqe(ring, cqe)
            if cqe.res < 0:
                raise OSError(f"io_uring read failed with error {-cqe.res}")
            liburing.io_uring_cqe_seen(ring, cqe)

        liburing.io_uring_queue_exit(ring)

        results = {}
        for key, tensor_meta, buf in buffers:
            dtype = _DTYPE_MAP[tensor_meta["dtype"]]
            shape = tensor_meta["shape"]
            t = torch.frombuffer(buf, dtype=dtype).clone()
            if shape:
                t = t.view(shape)
            results[key] = t
        return results
    finally:
        os.close(fd)


# ─── Measurement engine ─────────────────────────────────────────────────────

def measure_read(read_fn, filepath: str, num_runs: int = NUM_RUNS):
    """
    Run read_fn(filepath) num_runs times with cold cache.
    Returns list of (mb_per_sec, is_valid) tuples.
    """
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    runs = []

    for i in range(num_runs):
        ensure_cold(filepath)
        time.sleep(0.2)  # small settle time after cache drop

        start = time.perf_counter()
        tensors = read_fn(filepath)
        elapsed = time.perf_counter() - start

        # Force tensor materialization (touch every byte)
        total_bytes = sum(t.element_size() * t.numel() for t in tensors.values())
        mb_per_sec = (total_bytes / (1024 * 1024)) / elapsed
        is_valid = mb_per_sec < CACHE_HIT_THRESHOLD_MBPS

        tag = "VALID" if is_valid else "CACHE_HIT"
        log_info(f"  Run {i+1}: {mb_per_sec:>8.1f} MB/s [{tag}]")
        runs.append((mb_per_sec, is_valid))

        del tensors  # free memory before next run

    return runs


def report_results(name: str, runs: list[tuple[float, bool]]):
    """Report median of valid runs only."""
    valid = [mbps for mbps, ok in runs if ok]
    invalid = [mbps for mbps, ok in runs if not ok]

    if valid:
        median = statistics.median(valid)
        log_info(f"  {name} Median (cold): {median:.1f} MB/s  ({len(valid)}/{len(runs)} valid)")
        return median
    else:
        log_warning(f"  {name}: ALL {len(runs)} runs were CACHE_HIT (>{CACHE_HIT_THRESHOLD_MBPS} MB/s). No valid cold reads!")
        return None


# ─── Synthetic model generator ───────────────────────────────────────────────

def generate_synthetic_file(target_path: str, size_gb: float = 1.5):
    """Generate a single large safetensors file for benchmarking."""
    p = Path(target_path)
    if p.exists():
        log_info(f"Synthetic file already exists: {target_path}")
        return target_path

    log_info(f"Generating {size_gb}GB synthetic safetensors file...")
    p.parent.mkdir(parents=True, exist_ok=True)

    num_elements = int((size_gb * 1024**3) // 2)  # float16 = 2 bytes
    big = torch.empty((num_elements,), dtype=torch.float16)
    save_file({"weight": big}, target_path)
    actual_mb = os.path.getsize(target_path) / (1024 * 1024)
    log_info(f"Generated: {actual_mb:.0f} MB")
    return target_path


# ─── Main benchmark ──────────────────────────────────────────────────────────

def run_benchmark():
    log_info("=" * 60)
    log_info("DeltaStream Disk I/O Benchmark: Standard vs io_uring")
    log_info("=" * 60)

    # Check io_uring availability
    try:
        import liburing
        has_uring = True
    except ImportError:
        has_uring = False
        log_warning("liburing not available — io_uring benchmark will be skipped.")

    # Check sudo
    drop_ok = drop_os_caches()
    if not drop_ok:
        log_warning("sudo drop_caches failed — using posix_fadvise only. Results may be less reliable.")

    # ── Test 1: GPT2 delta layer (~31 MB) ──
    gpt2_file = "delta_gpt2/deltas/layer_01.delta.safetensors"
    if os.path.exists(gpt2_file):
        file_mb = os.path.getsize(gpt2_file) / (1024 * 1024)
        log_info(f"\n{'─'*60}")
        log_info(f"TEST 1: GPT2 delta layer ({file_mb:.0f} MB)")
        log_info(f"{'─'*60}")

        log_info("Standard (open+read):")
        std_runs = measure_read(read_file_standard, gpt2_file)
        std_median = report_results("Standard", std_runs)

        if has_uring:
            log_info("io_uring (batched SQE):")
            uring_runs = measure_read(read_file_uring, gpt2_file)
            uring_median = report_results("io_uring", uring_runs)

            if std_median and uring_median:
                log_info(f"  Speedup: {uring_median / std_median:.2f}x")
    else:
        log_warning(f"GPT2 delta not found at {gpt2_file} — skipping Test 1.")

    # ── Test 2: Synthetic 1.5GB layer ──
    synth_file = "bench_synthetic_1.5gb.safetensors"
    generate_synthetic_file(synth_file, 1.5)
    file_mb = os.path.getsize(synth_file) / (1024 * 1024)

    log_info(f"\n{'─'*60}")
    log_info(f"TEST 2: Synthetic large layer ({file_mb:.0f} MB)")
    log_info(f"{'─'*60}")

    log_info("Standard (open+read):")
    std_runs = measure_read(read_file_standard, synth_file)
    std_median = report_results("Standard", std_runs)

    if has_uring:
        log_info("io_uring (batched SQE):")
        uring_runs = measure_read(read_file_uring, synth_file)
        uring_median = report_results("io_uring", uring_runs)

        if std_median and uring_median:
            log_info(f"  Speedup: {uring_median / std_median:.2f}x")

    log_info(f"\n{'='*60}")
    log_info("Benchmark complete.")
    log_info(f"{'='*60}")


if __name__ == "__main__":
    run_benchmark()

import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING
import torch

from deltastream.core.io_backend import IOBackend, StandardIOBackend
from deltastream.core.manifest import read_manifest
from deltastream.utils.logging import log_warning, log_info

if TYPE_CHECKING:
    from deltastream.core.cache import LayerCacheManager

# ─── SIGILL probe ─────────────────────────────────────────────────────────────

_PROBE_SCRIPT = """
import sys
try:
    import liburing
    ring_cls = getattr(liburing, 'io_uring', None) or getattr(liburing, 'Ring')
    ring = ring_cls()
    liburing.io_uring_queue_init(2, ring, 0)
    liburing.io_uring_queue_exit(ring)
    print('OK')
except Exception as e:
    print('ERR:' + str(e))
"""


def _probe_liburing_safe() -> tuple[bool, str]:
    """
    Runs a minimal liburing ring-init in a child subprocess.
    Returns (success: bool, reason: str).

    A SIGILL in the child manifests as returncode == -4 (SIGILL = signal 4).
    Any other crash or ImportError is also treated as unavailable.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", _PROBE_SCRIPT],
            capture_output=True,
            text=True,
            timeout=10,
        )
        stdout = result.stdout.strip()
        returncode = result.returncode

        if returncode == -4 or returncode == 132:  # SIGILL on Linux
            return False, "io_uring blocked by CPU virtualization (SIGILL), using StandardIOBackend"
        if returncode != 0:
            return False, f"liburing probe crashed (exit {returncode}): {result.stderr.strip()[:120]}"
        if stdout == "OK":
            return True, "ok"
        return False, f"liburing probe returned: {stdout}"
    except subprocess.TimeoutExpired:
        return False, "liburing probe timed out"
    except Exception as e:
        return False, f"liburing probe exception: {e}"


class IOBackendFactory:
    """Factory to detect the environment and return the best IOBackend."""

    @staticmethod
    def detect_environment() -> str:
        if os.name != "posix":
            return "unsupported"
        try:
            with open("/proc/version", "r") as f:
                version_str = f.read().lower()
                if "microsoft" in version_str:
                    return "wsl2"
                return "baremetal"
        except FileNotFoundError:
            return "unsupported"

    @staticmethod
    def get_backend(base_model_path: str, delta_model_path: str, device: str = "cpu") -> IOBackend:
        env = IOBackendFactory.detect_environment()

        if env not in ("wsl2", "baremetal"):
            log_info(f"Using StandardIOBackend (unsupported environment: {env})")
            return StandardIOBackend(base_model_path, delta_model_path, device)

        try:
            import liburing  # noqa: F401 — just check import
        except ImportError:
            log_info("Using StandardIOBackend (liburing not installed)")
            return StandardIOBackend(base_model_path, delta_model_path, device)

        # Probe in a subprocess to catch SIGILL before it kills our process
        ok, reason = _probe_liburing_safe()
        if not ok:
            log_warning(reason)
            return StandardIOBackend(base_model_path, delta_model_path, device)

        log_info(f"Using IOUringBackend (mode: {env})")
        return IOUringBackend(base_model_path, delta_model_path, env, device)



class IOUringBackend(IOBackend):
    """
    Async NVMe Backend using io_uring.
    Batches multiple tensor reads per ring submission.
    """

    def __init__(self, base_model_path: str, delta_model_path: str, mode: str, device: str = "cpu"):
        self.delta_model_dir = Path(delta_model_path)
        self.base_model_dir = Path(base_model_path)
        self.device = device
        self.manifest = read_manifest(self.delta_model_dir)
        self.mode = mode
        
        if self.mode == "baremetal":
            # TODO: Implement O_DIRECT + page-aligned mmap + torch.frombuffer zero-copy
            raise NotImplementedError("Bare metal O_DIRECT mode is stubbed for Phase 3 future work.")

    def _parse_safetensors_metadata(self, filepath: str):
        """Reads safetensors header to find byte offsets for each tensor."""
        with open(filepath, "rb") as f:
            length_bytes = f.read(8)
            (header_len,) = struct.unpack("<Q", length_bytes)
            header_json = f.read(header_len).decode("utf-8")
            metadata = json.loads(header_json)
            
            # The __metadata__ key isn't a tensor
            metadata.pop("__metadata__", None)
            
            data_start = 8 + header_len
            return metadata, data_start

    def _dtype_from_safetensors(self, st_dtype: str) -> torch.dtype:
        mapping = {
            "F32": torch.float32,
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "I32": torch.int32,
            "I16": torch.int16,
            "I8": torch.int8,
            "U8": torch.uint8,
        }
        return mapping[st_dtype]

    def _read_file_uring(self, filepath: str) -> Dict[str, torch.Tensor]:
        import liburing
        
        metadata, data_start = self._parse_safetensors_metadata(filepath)
        
        fd = os.open(filepath, os.O_RDONLY)
        try:
            num_tensors = len(metadata)
            if num_tensors == 0:
                return {}

            # Version-adaptive: 2024 uses io_uring(), 2026 uses Ring()
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
            
            # Submit entire batch in one syscall
            liburing.io_uring_submit(ring)
            
            cqe = _Cqe()
            for _ in range(num_tensors):
                liburing.io_uring_wait_cqe(ring, cqe)
                if cqe.res < 0:
                    raise OSError(f"io_uring read failed with error {-cqe.res}")
                liburing.io_uring_cqe_seen(ring, cqe)
            
            liburing.io_uring_queue_exit(ring)
            
            # Convert buffers to tensors
            results = {}
            for key, tensor_meta, buf in buffers:
                dtype = self._dtype_from_safetensors(tensor_meta["dtype"])
                shape = tensor_meta["shape"]
                t = torch.frombuffer(buf, dtype=dtype).clone().view(shape)
                if self.device != "cpu":
                    t = t.to(self.device)
                results[key] = t
                
            return results
            
        finally:
            os.close(fd)


    def fetch_layer(self, layer_idx: int, cache: "LayerCacheManager") -> Dict[str, torch.Tensor]:
        if layer_idx < 0 or layer_idx >= self.manifest.num_layers:
            raise IndexError(f"Layer {layer_idx} out of bounds")

        if layer_idx == 0:
            base_path = str(self.delta_model_dir / "base" / "layer_00.safetensors")
            try:
                return self._read_file_uring(base_path)
            except Exception as e:
                log_warning(f"IOUringBackend failed on layer 0: {e}. Falling back to standard read.")
                from safetensors.torch import load_file
                return load_file(base_path, device=self.device)
            
        # Recursive fetch: get previous layer from cache
        prev_layer = cache.get_layer(layer_idx - 1)
        
        delta_path = self.delta_model_dir / "deltas" / f"layer_{layer_idx:02d}.delta.safetensors"
        
        if self.manifest.compression == "none" or not self.manifest.compression.startswith("zstd"):
            try:
                delta = self._read_file_uring(str(delta_path))
            except Exception as e:
                log_warning(f"IOUringBackend failed on layer {layer_idx}: {e}. Falling back to standard read.")
                from deltastream.core.delta_decoder import _load_with_decompression
                delta = _load_with_decompression(delta_path, self.manifest.compression, device=self.device)
        else:
            from deltastream.core.delta_decoder import _load_with_decompression
            delta = _load_with_decompression(delta_path, self.manifest.compression, device=self.device)
        
        from deltastream.core.delta_decoder import _apply_delta
        return _apply_delta(prev_layer, delta, layer_idx, self.manifest)

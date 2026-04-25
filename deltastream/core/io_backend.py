import abc
from pathlib import Path
from typing import Dict, TYPE_CHECKING
import torch

from safetensors.torch import load_file as st_load_file
from deltastream.core.manifest import read_manifest
from deltastream.core.delta_decoder import _load_with_decompression


if TYPE_CHECKING:
    from deltastream.core.cache import LayerCacheManager

class IOBackend(abc.ABC):
    """
    Abstract interface for disk I/O when a cache miss occurs.
    Phase 3 will introduce an io_uring backend here.
    """
    
    @abc.abstractmethod
    def fetch_layer(self, layer_idx: int, cache: "LayerCacheManager") -> Dict[str, torch.Tensor]:
        """
        Fetch all tensors for the given layer. 
        Uses `cache` to fetch the previous layer if needed (N-1).
        """
        pass

class StandardIOBackend(IOBackend):
    """
    Default IO Backend using standard blocking file reads via safetensors.
    Reconstructs layer N using layer N-1 from the cache.
    """
    
    def __init__(self, base_model_path: str, delta_model_path: str, device: str = "cpu"):
        self.delta_model_dir = Path(delta_model_path)
        self.device = device
        self.manifest = read_manifest(self.delta_model_dir)
        
    def fetch_layer(self, layer_idx: int, cache: "LayerCacheManager") -> Dict[str, torch.Tensor]:
        if layer_idx < 0 or layer_idx >= self.manifest.num_layers:
            raise IndexError(f"Layer {layer_idx} out of bounds")

        if layer_idx == 0:
            base_path = self.delta_model_dir / "base" / "layer_00.safetensors"
            return st_load_file(str(base_path), device=self.device)  # base never compressed

        # Recursive fetch: get previous layer from cache
        prev_layer = cache.get_layer(layer_idx - 1)

        delta_path = self.delta_model_dir / "deltas" / f"layer_{layer_idx:02d}.delta.safetensors"
        delta = _load_with_decompression(delta_path, self.manifest.compression, device=self.device)

        from deltastream.core.delta_decoder import _apply_delta
        return _apply_delta(prev_layer, delta, layer_idx, self.manifest)


import collections
import threading
from typing import Dict, Optional

import torch

from deltastream.core.io_backend import IOBackend
from deltastream.core.memory import pin_tensor, unpin_tensor
from deltastream.utils.logging import log_info, log_warning


class LayerCacheManager:
    """
    LRU Tiered RAM Layer Cache Manager.
    Pins fetched layers in RAM to prevent swapping, evicts least-recently-used layers,
    and uses two background threads to prefetch upcoming layers (N+1, N+2).
    """

    def __init__(self, io_backend: IOBackend, max_ram_gb: float = 11.0):
        self.io_backend = io_backend
        self.max_bytes = int(max_ram_gb * 1024**3)
        self.current_bytes = 0

        self.cache: collections.OrderedDict[int, Dict[str, torch.Tensor]] = collections.OrderedDict()
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        
        # Keep track of layers currently being fetched to avoid duplicate work
        self.fetching_layers = set()
        
        self.hits = 0
        self.misses = 0

    def _get_layer_size(self, layer: Dict[str, torch.Tensor]) -> int:
        return sum(t.element_size() * t.numel() for t in layer.values())

    def _evict_if_needed(self, required_bytes: int):
        """Evict LRU layers until we have enough space. Must be called with lock held."""
        while self.current_bytes + required_bytes > self.max_bytes and self.cache:
            evict_idx, evict_layer = self.cache.popitem(last=False)
            evict_size = self._get_layer_size(evict_layer)
            
            for t in evict_layer.values():
                unpin_tensor(t)
                
            self.current_bytes -= evict_size
            log_info(f"Evicted layer {evict_idx} from RAM cache. Freed {evict_size / 1024**2:.2f} MB.")

    def _fetch_layer_guarded(self, layer_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """Actually fetch the layer and insert it into cache. Caller must NOT hold lock."""
        try:
            layer = self.io_backend.fetch_layer(layer_idx, self)
        except Exception as e:
            return None

        layer_size = self._get_layer_size(layer)
        for t in layer.values():
            pin_tensor(t)

        with self.lock:
            # We are the designated fetcher, so just insert.
            self._evict_if_needed(layer_size)
            self.cache[layer_idx] = layer
            self.current_bytes += layer_size
            self.cache.move_to_end(layer_idx)
            
            self.fetching_layers.discard(layer_idx)
            self.cond.notify_all()
            
        return layer

    def get_layer(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """
        Primary DeltaStream hook. Returns the tensor dictionary for the requested layer.
        Triggers async prefetching for N+1 and N+2.
        """
        needs_fetch = False

        with self.lock:
            while True:
                if layer_idx in self.cache:
                    self.cache.move_to_end(layer_idx)
                    self.hits += 1
                    layer = self.cache[layer_idx]
                    break
                
                if layer_idx in self.fetching_layers:
                    # Another thread is fetching this layer. Wait.
                    self.cond.wait()
                    continue
                
                # It's not in cache, and nobody is fetching it. We must fetch.
                self.misses += 1
                self.fetching_layers.add(layer_idx)
                needs_fetch = True
                break

        if needs_fetch:
            log_info(f"Cache miss for layer {layer_idx}. Fetching from disk...")
            layer = self._fetch_layer_guarded(layer_idx)
            if layer is None:
                # Cleanup if fetch failed
                with self.lock:
                    self.fetching_layers.discard(layer_idx)
                    self.cond.notify_all()
                raise ValueError(f"Failed to fetch layer {layer_idx}")

        # Kick off prefetching for N+1 and N+2
        self._prefetch_async(layer_idx + 1)
        self._prefetch_async(layer_idx + 2)

        return layer

    def _prefetch_async(self, layer_idx: int):
        """Spawns a background thread to fetch a layer if not already cached/fetching."""
        with self.lock:
            if layer_idx in self.cache or layer_idx in self.fetching_layers:
                return
            self.fetching_layers.add(layer_idx)

        def prefetch_worker():
            self._fetch_layer_guarded(layer_idx)

        thread = threading.Thread(target=prefetch_worker, daemon=True)
        thread.start()

    def get_metrics(self) -> Dict[str, float]:
        """Return cache hit/miss ratio and current usage."""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0.0
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "usage_gb": self.current_bytes / 1024**3,
                "max_gb": self.max_bytes / 1024**3,
                "cached_layers": list(self.cache.keys())
            }

import pytest
import time
import torch
import threading
from typing import Dict
from deltastream.core.cache import LayerCacheManager
from deltastream.core.io_backend import IOBackend
from deltastream.core.memory import pin_tensor

class MockIOBackend(IOBackend):
    def __init__(self, num_layers: int, delay: float = 0.1):
        self.num_layers = num_layers
        self.delay = delay
        self.fetch_calls = 0
        self.lock = threading.Lock()

    def fetch_layer(self, layer_idx: int, cache: LayerCacheManager) -> Dict[str, torch.Tensor]:
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"Layer {layer_idx} out of bounds")
        
        with self.lock:
            self.fetch_calls += 1
            
        time.sleep(self.delay)
        # Simulate 1GB layer for memory accounting, but don't actually allocate 1GB!
        # We can mock the tensor size by subclassing or just allocate a small one and 
        # override `_get_layer_size` in the test cache.
        # But even simpler: just allocate 1 element, we will mock `_get_layer_size` in the tests.
        return {"weight": torch.empty(1)}

def patch_cache_size(cache, size_bytes):
    cache._get_layer_size = lambda layer: size_bytes

def test_cache_eviction():
    # Set max RAM to 2.5 GB (will hold exactly 2 x 1GB layers)
    backend = MockIOBackend(num_layers=10, delay=0.0)
    cache = LayerCacheManager(io_backend=backend, max_ram_gb=2.5)
    patch_cache_size(cache, 1024**3)  # 1GB
    
    # Disable prefetch to test purely synchronous eviction
    cache._prefetch_async = lambda idx: None

    # Fetch layer 0 -> size 1GB
    layer0 = cache.get_layer(0)
    assert 0 in cache.cache
    assert len(cache.cache) == 1

    # Fetch layer 1 -> size 2GB total
    layer1 = cache.get_layer(1)
    assert 1 in cache.cache
    assert len(cache.cache) == 2

    # Fetch layer 2 -> size 3GB total (exceeds 2.5GB). Layer 0 should be evicted.
    layer2 = cache.get_layer(2)
    assert 2 in cache.cache
    assert 1 in cache.cache
    assert 0 not in cache.cache

def test_cache_prefetching():
    backend = MockIOBackend(num_layers=10, delay=0.1)
    cache = LayerCacheManager(io_backend=backend, max_ram_gb=5.0)
    patch_cache_size(cache, 1024**3)

    # Fetch layer 0 synchronously.
    # This will trigger async prefetch for layers 1 and 2.
    cache.get_layer(0)
    
    # Wait for prefetch threads to finish
    time.sleep(0.4)
    
    assert 1 in cache.cache
    assert 2 in cache.cache
    assert backend.fetch_calls == 3

    # Now get layer 1 (should be a cache hit)
    cache.get_layer(1)
    metrics = cache.get_metrics()
    assert metrics["hits"] == 1
    assert metrics["misses"] == 1

def test_thread_safety():
    backend = MockIOBackend(num_layers=5, delay=0.05)
    cache = LayerCacheManager(io_backend=backend, max_ram_gb=10.0)
    patch_cache_size(cache, 1024**3)
    
    def hammer_cache():
        for i in range(5):
            cache.get_layer(i)
            
    threads = [threading.Thread(target=hammer_cache) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    
    # Ensure no layer was fetched more than once despite heavy concurrent requests
    assert backend.fetch_calls == 5

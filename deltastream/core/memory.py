import ctypes
import os
import torch
from deltastream.utils.logging import log_warning

# Load libc once for performance
_libc = None
if os.name == "posix":
    try:
        _libc = ctypes.CDLL("libc.so.6", use_errno=True)
    except OSError as e:
        log_warning(f"Failed to load libc.so.6: {e}. Memory pinning disabled.")

_mlock_disabled = False
_mlock_warned = False

def pin_tensor(tensor: torch.Tensor) -> bool:
    """
    Locks the memory pages containing the tensor into RAM using Linux mlock.
    This prevents the OS from swapping these pages to disk.
    
    Returns:
        bool: True if successfully pinned, False otherwise.
    """
    global _mlock_disabled, _mlock_warned
    
    if _libc is None or _mlock_disabled:
        return False
        
    addr = tensor.data_ptr()
    size = tensor.element_size() * tensor.numel()
    
    if size == 0:
        return True

    res = _libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(size))
    if res != 0:
        _mlock_disabled = True
        if not _mlock_warned:
            log_warning("⚠ mlock unavailable (limited permissions) — running without RAM pinning. Performance may vary.")
            _mlock_warned = True
        return False
        
    return True

def unpin_tensor(tensor: torch.Tensor) -> bool:
    """
    Unlocks the memory pages containing the tensor using Linux munlock,
    allowing the OS to swap or free them normally.
    
    Returns:
        bool: True if successfully unpinned, False otherwise.
    """
    if _libc is None or _mlock_disabled:
        return False
        
    addr = tensor.data_ptr()
    size = tensor.element_size() * tensor.numel()
    
    if size == 0:
        return True

    res = _libc.munlock(ctypes.c_void_p(addr), ctypes.c_size_t(size))
    if res != 0:
        # Ignore munlock errors if we had issues.
        return False
        
    return True

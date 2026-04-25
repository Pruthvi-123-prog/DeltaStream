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

def pin_tensor(tensor: torch.Tensor) -> bool:
    """
    Locks the memory pages containing the tensor into RAM using Linux mlock.
    This prevents the OS from swapping these pages to disk.
    
    Returns:
        bool: True if successfully pinned, False otherwise.
    """
    if _libc is None:
        return False
        
    addr = tensor.data_ptr()
    size = tensor.element_size() * tensor.numel()
    
    if size == 0:
        return True

    res = _libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(size))
    if res != 0:
        errno = ctypes.get_errno()
        # Warn but do not crash. 
        # ENOMEM (12) usually means ulimit -l is exceeded.
        # EPERM (1) means lacking CAP_IPC_LOCK.
        log_warning(f"mlock failed with errno {errno}. Tensor not pinned. Check ulimit -l or permissions.")
        return False
        
    return True

def unpin_tensor(tensor: torch.Tensor) -> bool:
    """
    Unlocks the memory pages containing the tensor using Linux munlock,
    allowing the OS to swap or free them normally.
    
    Returns:
        bool: True if successfully unpinned, False otherwise.
    """
    if _libc is None:
        return False
        
    addr = tensor.data_ptr()
    size = tensor.element_size() * tensor.numel()
    
    if size == 0:
        return True

    res = _libc.munlock(ctypes.c_void_p(addr), ctypes.c_size_t(size))
    if res != 0:
        errno = ctypes.get_errno()
        log_warning(f"munlock failed with errno {errno}.")
        return False
        
    return True

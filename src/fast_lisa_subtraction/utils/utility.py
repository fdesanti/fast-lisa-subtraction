import sys
import torch
import random
import numpy as np

try:
    import cupy as cp
except ImportError:
    pass

from .logger import log

def free_gpu_memory():
    """Free cached GPU memory for CuPy and PyTorch backends.

    This is a best-effort cleanup that silently ignores missing backends
    or cleanup failures.

    Returns
    -------
    None
    """
    try:
        # Free all cached GPU memory in the default memory pool.
        cp.get_default_memory_pool().free_all_blocks()

        # If you're using pinned (page-locked) memory, free that as well.
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except:
        pass
    try:
        #free with torch
        torch.cuda.empty_cache()
    except:
        pass

def set_seed(seed):
    """Set RNG seeds for reproducibility across common backends.

    Parameters
    ----------
    seed : int
        Seed value to set for Python's ``random``, NumPy, CuPy (if
        available), and PyTorch.

    Returns
    -------
    None
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        cp.random.seed(seed)
    except :
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    log.info(f"Setting random seed to {seed}")

def repair_cupy_runtime():
    """Undo the shadowing of ``cupy.cuda.runtime`` caused by some imports.

    ``gbgpu.utils.utility`` (gbgpu 1.1.x and 1.2.x) runs
    ``from cupy.cuda.runtime import setDevice``. That import replaces ``cupy.cuda.runtime``
    with CuPy's re-export module, which does not include the private
    ``_getLocalRuntimeVersion``, so ``cupy.cuda.get_local_runtime_version()`` then raises
    ``AttributeError``. ``gpubackendtools`` calls it before loading any CUDA backend and
    does not catch that error. This puts the missing function back.

    Does nothing if CuPy is not imported or the module is intact.
    """
    shim = sys.modules.get("cupy.cuda.runtime")
    if shim is None or hasattr(shim, "_getLocalRuntimeVersion"):
        return
    try:
        from cupy_backends.cuda.api import runtime as _runtime
        shim._getLocalRuntimeVersion = _runtime._getLocalRuntimeVersion
    except (ImportError, AttributeError):
        pass

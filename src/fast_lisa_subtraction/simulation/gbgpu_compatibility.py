"""
Build a ``GBGPU`` object that works with any gbgpu version.

- gbgpu <= 1.1 selects the device with ``GBGPU(use_gpu=...)``.
- gbgpu >= 1.2 selects it with ``GBGPU(force_backend=...)``, e.g. ``'cpu'`` or ``'cuda12x'``.

:func:`build_gbgpu` detects which version is installed and always gives GBGPU orbits on
the same device (see :mod:`.orbits`). With gbgpu >= 1.2 it never passes
``force_backend=None``: in that case gpubackendtools can pick a ``lisatools`` backend
instead of a ``gbgpu`` one.
"""

import inspect

from ..utils import log, repair_cupy_runtime
from .orbits import GPU_BACKENDS, BackendSafeMixin, _uses_gpu, get_orbits


def uses_force_backend(gbgpu_class):
    """Check which constructor ``gbgpu_class`` has.

    Parameters
    ----------
    gbgpu_class : type
        The ``GBGPU`` class.

    Returns
    -------
    bool
        True if it takes ``force_backend`` (gbgpu >= 1.2), False if it takes ``use_gpu``.
    """
    try:
        params = inspect.signature(gbgpu_class.__init__).parameters
    except (TypeError, ValueError):
        return False
    return "force_backend" in params


def _backend_base(backend):
    """Return the short name (``'cpu'``, ``'cuda12x'``, ...) of a ``force_backend`` value.

    Accepts a string, a ``(module, backend)`` tuple or a Backend object. Raises the
    gpubackendtools error if the backend cannot be loaded.
    """
    from gpubackendtools import get_backend

    if isinstance(backend, (tuple, list)):
        backend = backend[-1]
    name = getattr(backend, "name", backend)
    if not str(name).startswith("gbgpu_"):
        name = f"gbgpu_{name}"
    return get_backend(name).name.split("_")[-1]


def _reason(exc):
    """Summarise an exception and its causes on one line."""
    parts = []
    while exc is not None and len(parts) < 4:
        msg = str(exc).strip().splitlines()
        parts.append(f"{type(exc).__name__}: {msg[0] if msg else ''}")
        exc = exc.__cause__
    return " <- ".join(parts)


def build_gbgpu(gbgpu_class, use_gpu, **gbgpu_kwargs):
    """Create a ``GBGPU`` instance, with orbits on the same device.

    Parameters
    ----------
    gbgpu_class : type
        The ``GBGPU`` class (``gbgpu.gbgpu.GBGPU``).
    use_gpu : bool
        If True, run on the GPU. With gbgpu >= 1.2, fall back to the CPU (with a warning)
        if no GPU backend can be loaded.
    **gbgpu_kwargs : dict
        Passed to the ``GBGPU`` constructor. If ``orbits`` is missing or None, matching
        orbits are created. With gbgpu >= 1.2, ``force_backend`` overrides ``use_gpu``
        (without CPU fallback).

    Returns
    -------
    gbgpu.gbgpu.GBGPU
        The GBGPU instance. Its ``xp`` attribute tells which device it uses.
    """
    #make sure CuPy can report its CUDA version (see repair_cupy_runtime)
    repair_cupy_runtime()

    if not uses_force_backend(gbgpu_class):
        #gbgpu <= 1.1: convert a force_backend argument into use_gpu
        requested = gbgpu_kwargs.pop("force_backend", None)
        if isinstance(requested, (tuple, list)):
            requested = requested[-1]
        if requested is not None:
            use_gpu = str(getattr(requested, "name", requested)).split("_")[-1].lower() != "cpu"
        if gbgpu_kwargs.get("orbits") is None:
            gbgpu_kwargs["orbits"] = get_orbits(use_gpu)
        return gbgpu_class(use_gpu=use_gpu, **gbgpu_kwargs)

    #gbgpu >= 1.2: try the requested backend, or the GPU backends and then the CPU
    requested = gbgpu_kwargs.pop("force_backend", None)
    if requested is not None:
        candidates = [requested]
    elif use_gpu:
        candidates = list(GPU_BACKENDS) + ["cpu"]
    else:
        candidates = ["cpu"]

    failures = []
    for candidate in candidates:
        try:
            base = _backend_base(candidate)
        except Exception as exc:
            if requested is not None:
                raise
            failures.append(f"{candidate}: {_reason(exc)}")
            continue

        if base == "cpu" and failures:
            log.warning(
                "No GBGPU CUDA backend could be loaded, running GBGPU on the CPU.\n  "
                + "\n  ".join(failures)
            )

        kwargs = dict(gbgpu_kwargs)
        if kwargs.get("orbits") is None:
            kwargs["orbits"] = get_orbits(base != "cpu", backend=base)
        gb = gbgpu_class(force_backend=base, **kwargs)

        #user-supplied orbits on the wrong device would fail later in run_wave
        if _uses_gpu(gb) != _uses_gpu(gb.orbits) and not isinstance(gb.orbits, BackendSafeMixin):
            log.warning(
                f"GBGPU runs on '{gb.backend.name}' but the orbits run on "
                f"'{gb.orbits.backend.name}': Orbits.get_pos will reject the arrays."
            )
        return gb

    raise RuntimeError("Could not load any GBGPU backend:\n  " + "\n  ".join(failures))

"""Select a safe local renderer before importing/compiling the Warp backend."""
import importlib.util
import platform


def warp_status():
    if platform.system() == "Darwin":
        return False, "MJX Warp rendering is disabled on macOS: the ARM CPU render kernel has a confirmed native crash. Use classic."
    if importlib.util.find_spec("warp") is None:
        return False, "warp-lang is not installed; use classic or install the optional CUDA stack."
    import warp
    if not warp.is_cuda_available():
        return False, "MJX Warp rendering requires a CUDA device; CPU rendering uses classic."
    return True, "CUDA device available (throughput is not yet measured on this device)."


def select_backend(requested="auto"):
    if requested not in ("auto", "classic", "warp"):
        raise ValueError(f"unknown backend {requested!r}")
    if requested == "classic":
        return "classic"
    supported, reason = warp_status()
    if requested == "warp" and not supported:
        raise RuntimeError(reason)
    return "warp" if supported else "classic"

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.path.join(ROOT, "dist", "libmojo-fast-simplification.so")

I = ctypes.c_int64
F = ctypes.c_double


class BuildError(RuntimeError):
    pass


def _mojo_command() -> list[str]:
    override = os.environ.get("MOJO_FAST_SIMPLIFICATION_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi):
        return [
            pixi,
            "run",
            "--manifest-path",
            os.path.join(ROOT, "pixi.toml"),
            "mojo",
        ]
    raise BuildError("mojo not found; run `pixi run build`")


def build(force: bool = False) -> str:
    sources = [
        os.path.join(path, name)
        for path, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        if os.path.getmtime(LIB) >= max(os.path.getmtime(path) for path in sources):
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    command = _mojo_command() + [
        "build",
        "--emit",
        "shared-lib",
        os.path.join(SRC, "simplify.mojo"),
        "-o",
        LIB,
    ]
    process = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if process.returncode or not os.path.exists(LIB):
        raise BuildError((process.stderr or process.stdout).strip()[:4000])
    return LIB


_loaded: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        _loaded = ctypes.CDLL(build())
        _loaded.mfs_simplify.argtypes = [I] * 24 + [F, I]
        _loaded.mfs_simplify.restype = None
        _loaded.mfs_replay.argtypes = [I] * 15
        _loaded.mfs_replay.restype = None
        _loaded.mfs_version.argtypes = []
        _loaded.mfs_version.restype = I
    return _loaded


def addr(array: np.ndarray) -> int:
    return int(array.ctypes.data)

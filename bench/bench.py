"""Benchmark Mojo QEM decimation against fast-simplification 0.1.13."""

from __future__ import annotations

import gc
import importlib
import math
import os
import platform
import statistics
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_PYTHON = os.path.join(ROOT, "python")


def _import_implementations():
    sys.path = [
        entry
        for entry in sys.path
        if os.path.abspath(entry or os.getcwd()) != LOCAL_PYTHON
    ]
    upstream = importlib.import_module("fast_simplification")
    for name in list(sys.modules):
        if name == "fast_simplification" or name.startswith("fast_simplification."):
            del sys.modules[name]
    sys.path.insert(0, LOCAL_PYTHON)
    mojo = importlib.import_module("fast_simplification")
    return mojo, upstream


mfs, upstream = _import_implementations()


def torus(nu: int, nv: int):
    u = np.arange(nu) * (2 * np.pi / nu)
    v = np.arange(nv) * (2 * np.pi / nv)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    radius = 1.0 + 0.3 * np.cos(vv)
    points = np.column_stack(
        [
            (radius * np.cos(uu)).ravel(),
            (radius * np.sin(uu)).ravel(),
            (0.3 * np.sin(vv)).ravel(),
        ]
    )
    faces = []
    for i in range(nu):
        for j in range(nv):
            a = i * nv + j
            b = ((i + 1) % nu) * nv + j
            c = ((i + 1) % nu) * nv + (j + 1) % nv
            d = i * nv + (j + 1) % nv
            faces.extend(((a, b, c), (a, c, d)))
    return points, np.asarray(faces, dtype=np.int32)


def measured(function, repeat: int = 3) -> float:
    function()
    samples = []
    gc.disable()
    try:
        for _ in range(repeat):
            start = time.perf_counter()
            function()
            samples.append(time.perf_counter() - start)
    finally:
        gc.enable()
    return statistics.median(samples)


CASES = []


def case(name):
    def register(function):
        CASES.append((name, function))
        return function

    return register


@case("simplify 5,000 faces to 50%")
def _():
    points, faces = torus(50, 50)
    return (
        lambda: mfs.simplify(points, faces, 0.5),
        lambda: upstream.simplify(points, faces, 0.5),
    )


@case("simplify 20,000 faces to 25%")
def _():
    points, faces = torus(100, 100)
    return (
        lambda: mfs.simplify(points, faces, 0.75),
        lambda: upstream.simplify(points, faces, 0.75),
    )


@case("simplify 20,000 faces to 5%")
def _():
    points, faces = torus(100, 100)
    return (
        lambda: mfs.simplify(points, faces, 0.95),
        lambda: upstream.simplify(points, faces, 0.95),
    )


def _upstream_attribute_transfer(points, faces, attrs):
    _, _, collapses = upstream.simplify(
        points, faces, 0.9, return_collapses=True
    )
    out_points, out_faces, mapping = upstream.replay_simplification(
        points.astype(np.float32), faces, collapses
    )
    sums = np.bincount(mapping, weights=attrs, minlength=len(out_points))
    counts = np.bincount(mapping, minlength=len(out_points))
    return out_points, out_faces, sums / np.maximum(counts, 1)


@case("attribute-preserving 5,000 faces to 10%")
def _():
    points, faces = torus(50, 50)
    attrs = (points[:, 0] > 0).astype(np.float64)
    return (
        lambda: mfs.simplify(
            points,
            faces,
            0.9,
            attributes=attrs,
            attribute_weights=1.0,
        ),
        lambda: _upstream_attribute_transfer(points, faces, attrs),
    )


@case("replay 5,000-face collapse history")
def _():
    points, faces = torus(50, 50)
    _, _, mojo_collapses = mfs.simplify(
        points, faces, 0.75, return_collapses=True
    )
    _, _, upstream_collapses = upstream.simplify(
        points, faces, 0.75, return_collapses=True
    )
    return (
        lambda: mfs.replay_simplification(points, faces, mojo_collapses),
        lambda: upstream.replay_simplification(
            points.astype(np.float32), faces, upstream_collapses
        ),
    )


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def milliseconds(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1e6:.1f} us"
    return f"{seconds * 1e3:.2f} ms"


def main() -> None:
    print(f"Machine: {cpu_name()}, {platform.system()} {platform.machine()}")
    print(f"Python: {platform.python_version()}; upstream: {upstream.__version__}")
    print()
    print("| case | Mojo | upstream | upstream / Mojo |")
    print("|---|---:|---:|---:|")
    for name, prepare in CASES:
        mojo_function, upstream_function = prepare()
        mojo_time = measured(mojo_function)
        upstream_time = measured(upstream_function)
        ratio = upstream_time / mojo_time
        if not math.isfinite(ratio):
            raise RuntimeError("non-finite benchmark result")
        print(
            f"| {name} | {milliseconds(mojo_time)} | "
            f"{milliseconds(upstream_time)} | {ratio:.2f}x |"
        )


if __name__ == "__main__":
    main()

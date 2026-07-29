from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest


def torus(nu: int = 32, nv: int = 20):
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


@pytest.fixture(scope="session")
def upstream_reference(tmp_path_factory):
    directory = tmp_path_factory.mktemp("upstream")
    path = directory / "reference.npz"
    script = r"""
import numpy as np
import fast_simplification as fs
import sys

def torus(nu=32, nv=20):
    u = np.arange(nu) * (2 * np.pi / nu)
    v = np.arange(nv) * (2 * np.pi / nv)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    radius = 1.0 + 0.3 * np.cos(vv)
    points = np.column_stack([
        (radius * np.cos(uu)).ravel(),
        (radius * np.sin(uu)).ravel(),
        (0.3 * np.sin(vv)).ravel(),
    ])
    faces = []
    for i in range(nu):
        for j in range(nv):
            a = i * nv + j
            b = ((i + 1) % nu) * nv + j
            c = ((i + 1) % nu) * nv + (j + 1) % nv
            d = i * nv + (j + 1) % nv
            faces.extend(((a, b, c), (a, c, d)))
    return points, np.asarray(faces, dtype=np.int32)

simple_points = np.array([
    [.5, -.5, 0], [0, -.5, 0], [-.5, -.5, 0],
    [.5, 0, 0], [0, 0, 0], [-.5, 0, 0],
    [.5, .5, 0], [0, .5, 0], [-.5, .5, 0],
], dtype=float)
simple_faces = np.array([
    [0, 1, 3], [4, 3, 1], [1, 2, 4], [5, 4, 2],
    [3, 4, 6], [7, 6, 4], [4, 5, 7], [8, 7, 5],
], dtype=np.int32)
sp, sf, sc = fs.simplify(simple_points, simple_faces, .5, return_collapses=True)
torus_points, torus_faces = torus()
tp, tf = fs.simplify(torus_points, torus_faces, .75)
_, low_agg_faces = fs.simplify(torus_points, torus_faces, .5, agg=0)
oct_points = np.array([
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
], dtype=float)
oct_faces = np.array([
    [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
    [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
], dtype=np.int32)
lp, lf = fs.simplify(oct_points, oct_faces, lossless=True)
np.savez(
    sys.argv[1], version=fs.__version__,
    simple_points=sp, simple_faces=sf, simple_collapses=sc,
    torus_points=tp, torus_faces=tf,
    low_agg_face_count=len(low_agg_faces),
    lossless_points=lp, lossless_faces=lf,
)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = ""
    subprocess.run(
        [sys.executable, "-c", script, str(path)],
        cwd=directory,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return np.load(path)

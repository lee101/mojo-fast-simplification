from __future__ import annotations

import numpy as np

from ._lib import addr, lib
from .simplify import _mesh_arrays


def replay_simplification(points, triangles, collapses):
    """Replay a collapse history and return points, faces, and vertex mapping."""
    mesh_points, mesh_faces = _mesh_arrays(points, triangles)
    collapse_values = np.asarray(collapses)
    if collapse_values.size == 0:
        collapse_values = collapse_values.reshape(0, 2)
    if collapse_values.ndim != 2 or collapse_values.shape[1] != 2:
        raise ValueError("collapses must have shape (n, 2)")
    if collapse_values.size and (
        np.any(collapse_values < 0) or np.any(collapse_values >= len(mesh_points))
    ):
        raise ValueError("collapse history is invalid for this mesh")
    collapse_array = np.ascontiguousarray(collapse_values, dtype=np.int32)
    n_points = len(mesh_points)
    n_faces = len(mesh_faces)
    if n_faces == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.int32),
            np.full(n_points, -1, dtype=np.int64),
        )
    # Mojo pointers are non-nullable even when their logical length is zero.
    # Keep a one-row backing allocation for an empty collapse history.
    collapse_buffer = np.empty((max(len(collapse_array), 1), 2), dtype=np.int32)
    collapse_buffer[: len(collapse_array)] = collapse_array
    quadrics = np.empty((n_points, 10), dtype=np.float64)
    vertex_state = np.empty((5, n_points), dtype=np.int64)
    face_alive = np.empty(n_faces, dtype=np.int64)
    next_ref = np.empty(3 * n_faces, dtype=np.int64)
    mapping = vertex_state[4]
    result = np.full(3, -1, dtype=np.int64)
    vertex_state_address = addr(vertex_state)
    lib().mfs_replay(
        addr(mesh_points),
        addr(mesh_faces),
        addr(collapse_buffer),
        addr(quadrics),
        vertex_state_address,
        vertex_state_address + n_points * 8,
        vertex_state_address + 2 * n_points * 8,
        addr(face_alive),
        vertex_state_address + 3 * n_points * 8,
        addr(next_ref),
        vertex_state_address + 4 * n_points * 8,
        addr(result),
        n_points,
        n_faces,
        len(collapse_array),
    )
    if result[0] != 0:
        raise ValueError("collapse history is invalid for this mesh")
    if not (0 <= result[1] <= n_points and 0 <= result[2] <= n_faces):
        raise RuntimeError("Mojo replay returned invalid output lengths")
    return (
        mesh_points[: result[1]].astype(np.float32),
        mesh_faces[: result[2]],
        mapping,
    )

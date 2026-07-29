from __future__ import annotations

from typing import Any

import numpy as np

from ._lib import addr, lib


def _check_args(target_reduction, target_count, n_faces: int) -> int:
    if target_reduction is not None and target_count is not None:
        raise ValueError(
            "You may specify ``target_reduction`` or ``target_count``, but not both"
        )
    if target_reduction is None and target_count is None:
        raise ValueError("You must specify ``target_reduction`` or ``target_count``")
    if target_reduction is not None:
        if target_reduction > 1 or target_reduction < 0:
            raise ValueError("``target_reduction`` must be between 0 and 1")
        target_count = (1 - target_reduction) * n_faces
    if target_count < 0:
        raise ValueError("``target_count`` must be greater than 0")
    if target_count > n_faces:
        raise ValueError(
            f"``target_count`` must be less than the number of faces {n_faces}"
        )
    return int(target_count)


def _mesh_arrays(points, triangles) -> tuple[np.ndarray, np.ndarray]:
    points_array = np.array(points, dtype=np.float64, order="C", copy=True)
    if points_array.ndim != 2:
        raise ValueError("``points`` array must be 2 dimensional")
    if points_array.shape[1] != 3:
        raise ValueError(
            f"Expected ``points`` array to be (n, 3), not {points_array.shape}"
        )
    if not np.all(np.isfinite(points_array)):
        raise ValueError("``points`` array must contain only finite values")
    faces = np.asarray(triangles)
    if faces.ndim != 2:
        raise ValueError("``triangles`` array must be 2 dimensional")
    if faces.shape[1] != 3:
        raise ValueError(
            f"Expected ``triangles`` array to be (n, 3), not {faces.shape}"
        )
    if not np.issubdtype(faces.dtype, np.integer):
        raise TypeError("``triangles`` array must contain integer indices")
    if faces.size:
        if faces.min() < 0 or faces.max() >= len(points_array):
            raise ValueError("triangle index is outside the points array")
        if np.any(
            (faces[:, 0] == faces[:, 1])
            | (faces[:, 1] == faces[:, 2])
            | (faces[:, 2] == faces[:, 0])
        ):
            raise ValueError("triangles must contain three distinct vertex indices")
    int32 = np.iinfo(np.int32)
    if faces.size and (faces.min() < int32.min or faces.max() > int32.max):
        raise ValueError("triangle index cannot be represented as int32")
    faces = np.array(faces, dtype=np.int32, order="C", copy=True)
    return points_array, faces


def _attributes(attributes, n_points: int, attribute_weights):
    if attributes is None:
        values = np.empty(1, dtype=np.float64)
        weights = np.empty(1, dtype=np.float64)
        return values, weights, 0, None
    original = np.asarray(attributes)
    if original.ndim == 1:
        if len(original) != n_points:
            raise ValueError("attributes must have one row per point")
        original_shape = ()
        original = original[:, None]
    elif original.ndim >= 2 and original.shape[0] == n_points:
        original_shape = original.shape[1:]
        original = original.reshape(n_points, -1)
    else:
        raise ValueError("attributes must have one row per point")
    if not np.issubdtype(original.dtype, np.number):
        raise TypeError("attributes must be numeric")
    values = np.array(original, dtype=np.float64, order="C", copy=True)
    if not np.all(np.isfinite(values)):
        raise ValueError("attributes must contain only finite values")
    dimension = values.shape[1]
    if attribute_weights is None:
        weights = np.ones(dimension, dtype=np.float64)
    else:
        supplied = np.asarray(attribute_weights, dtype=np.float64)
        if supplied.ndim == 0:
            weights = np.full(dimension, supplied.item(), dtype=np.float64)
        else:
            weights = np.ascontiguousarray(supplied.reshape(-1), dtype=np.float64)
            if len(weights) != dimension:
                raise ValueError("attribute_weights must be scalar or match attribute width")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("attribute_weights must be finite and nonnegative")
    return values, weights, dimension, original_shape


def _decimate(
    points: np.ndarray,
    faces: np.ndarray,
    target_count: int,
    attributes,
    attribute_weights,
    lossless: bool,
    agg: float,
):
    attrs, weights, attr_dim, attr_shape = _attributes(
        attributes, len(points), attribute_weights
    )
    if len(faces) == 0 or (not lossless and target_count == len(faces)):
        empty = np.empty((0, 2), dtype=np.int32)
        attr_result = None
        if attributes is not None:
            attr_result = attrs.reshape((len(points),) + attr_shape)
        return points, faces, attr_result, empty

    n_points = len(points)
    n_faces = len(faces)
    # Immediately before adding a survivor's one-ring the lazy heap contains at
    # most 3 * n_faces entries. A triangle mesh has at most 2 * n_faces distinct
    # directed neighbors of one vertex, so this capacity makes heap_push's
    # capacity guard unreachable for valid input.
    capacity = max(16, 5 * n_faces + 16)
    quadrics = np.empty((n_points, 10), dtype=np.float64)
    face_alive = np.empty(n_faces, dtype=np.int64)
    vertex_state = np.empty((6, n_points), dtype=np.int64)
    next_ref = np.empty(3 * n_faces, dtype=np.int64)
    heap_errors = np.empty(capacity, dtype=np.float64)
    heap_indices = np.empty((2, capacity), dtype=np.uint64)
    collapses = np.empty((max(n_points, 1), 2), dtype=np.int32)
    result = np.full(4, -1, dtype=np.int64)
    if lossless:
        max_error = np.finfo(np.float64).eps
    else:
        exponent = np.log(1.0e-9) + agg * np.log(102.0)
        max_error = (
            np.finfo(np.float64).max
            if exponent > 700
            else float(np.exp(exponent))
        )
    effective_target = 0 if lossless else target_count
    vertex_state_address = addr(vertex_state)
    heap_indices_address = addr(heap_indices)

    lib().mfs_simplify(
        addr(points),
        addr(faces),
        addr(attrs),
        addr(weights),
        addr(quadrics),
        addr(face_alive),
        vertex_state_address,
        vertex_state_address + n_points * 8,
        vertex_state_address + 2 * n_points * 8,
        addr(next_ref),
        vertex_state_address + 3 * n_points * 8,
        vertex_state_address + 4 * n_points * 8,
        vertex_state_address + 5 * n_points * 8,
        addr(heap_errors),
        heap_indices_address,
        heap_indices_address + capacity * 8,
        heap_indices_address,
        heap_indices_address,
        addr(collapses),
        addr(result),
        n_points,
        n_faces,
        attr_dim,
        effective_target,
        max_error,
        capacity,
    )
    if result[0] != 0:
        raise RuntimeError(f"Mojo simplifier failed with status {result[0]}")
    if not (
        0 <= result[1] <= n_points
        and 0 <= result[2] <= n_faces
        and 0 <= result[3] <= n_points
    ):
        raise RuntimeError("Mojo simplifier returned invalid output lengths")
    out_points = points[: result[1]]
    out_faces = faces[: result[2]]
    out_collapses = collapses[: result[3]]
    out_attrs = None
    if attributes is not None:
        out_attrs = attrs[: result[1]].reshape((result[1],) + attr_shape)
    return out_points, out_faces, out_attrs, out_collapses


def simplify(
    points,
    triangles,
    target_reduction: float | None = None,
    target_count: int | None = None,
    agg: float = 7.0,
    verbose: bool = False,
    return_collapses: bool = False,
    lossless: bool = False,
    *,
    attributes=None,
    attribute_weights=None,
):
    """Simplify a triangular mesh with quadric error metrics.

    The first eight parameters and the ordinary return value mirror
    ``fast_simplification.simplify``. Supplying vertex ``attributes`` enables
    the Mojo extension: weighted attribute discontinuities affect collapse
    priority and the returned tuple includes the propagated attribute array.
    """
    mesh_points, mesh_faces = _mesh_arrays(points, triangles)
    if not np.isfinite(agg):
        raise ValueError("``agg`` must be finite")
    target = 0 if lossless else _check_args(
        target_reduction, target_count, len(mesh_faces)
    )
    out_points, out_faces, out_attrs, collapses = _decimate(
        mesh_points,
        mesh_faces,
        target,
        attributes,
        attribute_weights,
        lossless,
        float(agg),
    )
    if verbose:
        print(
            f"simplified {len(mesh_faces)} triangles to {len(out_faces)} "
            f"with {len(collapses)} collapses"
        )
    values: list[Any] = [out_points, out_faces]
    if attributes is not None:
        values.append(out_attrs)
    if return_collapses:
        values.append(collapses)
    return tuple(values)


def simplify_with_attributes(
    points,
    triangles,
    attributes,
    target_reduction: float | None = None,
    target_count: int | None = None,
    agg: float = 7.0,
    verbose: bool = False,
    return_collapses: bool = False,
    lossless: bool = False,
    attribute_weights=None,
):
    """Explicit attribute-preserving form of :func:`simplify`."""
    return simplify(
        points,
        triangles,
        target_reduction,
        target_count,
        agg,
        verbose,
        return_collapses,
        lossless,
        attributes=attributes,
        attribute_weights=attribute_weights,
    )


def simplify_mesh(
    mesh,
    target_reduction: float | None = None,
    target_count: int | None = None,
    agg: float = 7.0,
    verbose: bool = False,
):
    """Simplify a triangular PyVista mesh and preserve numeric point data."""
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError(
            "Please install pyvista to use this feature with:\npip install pyvista"
        ) from None
    padded = np.asarray(mesh.faces)
    if padded.size != mesh.n_cells * 4:
        raise ValueError(
            "Input mesh ``mesh`` must consist of only triangles.\n"
            "Run ``.triangulate()`` to convert to an all triangle mesh."
        )
    padded = padded.reshape(-1, 4)
    if np.any(padded[:, 0] != 3):
        raise ValueError(
            "Input mesh ``mesh`` must consist of only triangles.\n"
            "Run ``.triangulate()`` to convert to an all triangle mesh."
        )
    arrays = []
    metadata = []
    for name in mesh.point_data:
        value = np.asarray(mesh.point_data[name])
        if value.shape[0] == mesh.n_points and np.issubdtype(value.dtype, np.number):
            flat = value.reshape(mesh.n_points, -1)
            metadata.append((name, value.shape[1:], flat.shape[1]))
            arrays.append(flat)
    attributes = np.concatenate(arrays, axis=1) if arrays else None
    result = simplify(
        mesh.points,
        padded[:, 1:],
        target_reduction,
        target_count,
        agg,
        verbose,
        True,
        attributes=attributes,
        attribute_weights=1.0 if attributes is not None else None,
    )
    if attributes is None:
        out_points, out_faces, collapses = result
        out_attrs = None
    else:
        out_points, out_faces, out_attrs, collapses = result
    vtk_faces = np.empty((len(out_faces), 4), dtype=np.int64)
    vtk_faces[:, 0] = 3
    vtk_faces[:, 1:] = out_faces
    output = pv.PolyData(out_points, vtk_faces.ravel())
    if out_attrs is not None:
        offset = 0
        for name, shape, width in metadata:
            output.point_data[name] = out_attrs[:, offset : offset + width].reshape(
                (len(out_points),) + shape
            )
            offset += width
    output.field_data["fast_simplification_collapses"] = collapses
    return output

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import fast_simplification as mfs
from conftest import torus


SIMPLE_POINTS = np.array(
    [
        [0.5, -0.5, 0],
        [0, -0.5, 0],
        [-0.5, -0.5, 0],
        [0.5, 0, 0],
        [0, 0, 0],
        [-0.5, 0, 0],
        [0.5, 0.5, 0],
        [0, 0.5, 0],
        [-0.5, 0.5, 0],
    ],
    dtype=np.float64,
)
SIMPLE_FACES = np.array(
    [
        [0, 1, 3],
        [4, 3, 1],
        [1, 2, 4],
        [5, 4, 2],
        [3, 4, 6],
        [7, 6, 4],
        [4, 5, 7],
        [8, 7, 5],
    ],
    dtype=np.int32,
)


def torus_residual(points):
    radial = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2) - 1.0
    return np.abs(radial**2 + points[:, 2] ** 2 - 0.09)


def test_real_upstream_013_is_parity_reference(upstream_reference):
    assert str(upstream_reference["version"]) == "0.1.13"


def test_planar_example_matches_upstream_behavior(upstream_reference):
    points, faces, collapses = mfs.simplify(
        SIMPLE_POINTS, SIMPLE_FACES, 0.5, return_collapses=True
    )
    assert points.shape == upstream_reference["simple_points"].shape == (5, 3)
    assert faces.shape == upstream_reference["simple_faces"].shape == (4, 3)
    assert collapses.shape == upstream_reference["simple_collapses"].shape == (4, 2)
    assert points.dtype == upstream_reference["simple_points"].dtype == np.float64
    assert faces.dtype == upstream_reference["simple_faces"].dtype == np.int32
    assert np.all(points[:, 2] == 0)
    assert np.ptp(points[:, 0]) == pytest.approx(1.0)
    assert np.ptp(points[:, 1]) == pytest.approx(1.0)


def test_torus_count_and_qem_quality_match_upstream(upstream_reference):
    points, faces = torus()
    ours_points, ours_faces = mfs.simplify(points, faces, 0.75)
    upstream_points = upstream_reference["torus_points"]
    upstream_faces = upstream_reference["torus_faces"]
    assert ours_faces.shape == upstream_faces.shape == (320, 3)
    assert ours_points.shape == upstream_points.shape == (160, 3)
    ours_error = torus_residual(ours_points)
    upstream_error = torus_residual(upstream_points)
    assert ours_error.mean() <= upstream_error.mean() * 1.25
    assert ours_error.max() <= upstream_error.max() * 1.25


def test_torus_extent_numerically_tracks_upstream(upstream_reference):
    points, faces = torus()
    ours, _ = mfs.simplify(points, faces, target_count=320)
    upstream = upstream_reference["torus_points"]
    np.testing.assert_allclose(ours.min(axis=0), upstream.min(axis=0), atol=0.025)
    np.testing.assert_allclose(ours.max(axis=0), upstream.max(axis=0), atol=0.025)


def test_low_aggression_stops_at_same_error_envelope(upstream_reference):
    points, faces = torus()
    _, actual_faces = mfs.simplify(points, faces, 0.5, agg=0)
    assert len(actual_faces) == int(upstream_reference["low_agg_face_count"])
    assert len(actual_faces) == len(faces)


def test_lossless_matches_upstream_on_octahedron(upstream_reference):
    points = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ],
        dtype=np.int32,
    )
    actual_points, actual_faces = mfs.simplify(points, faces, lossless=True)
    np.testing.assert_array_equal(actual_points, upstream_reference["lossless_points"])
    np.testing.assert_array_equal(actual_faces, upstream_reference["lossless_faces"])


def test_zero_reduction_is_exact_passthrough():
    points, faces = torus(12, 8)
    actual_points, actual_faces = mfs.simplify(points, faces, 0.0)
    np.testing.assert_array_equal(actual_points, points)
    np.testing.assert_array_equal(actual_faces, faces)


def test_collapse_replay_matches_simplification():
    points, faces = torus(20, 12)
    out_points, out_faces, collapses = mfs.simplify(
        points, faces, 0.6, return_collapses=True
    )
    replay_points, replay_faces, mapping = mfs.replay_simplification(
        points, faces, collapses
    )
    np.testing.assert_allclose(replay_points, out_points, atol=2e-7)
    np.testing.assert_array_equal(replay_faces, out_faces)
    assert mapping.shape == (len(points),)
    assert mapping.min() >= 0
    assert mapping.max() < len(out_points)


@pytest.mark.parametrize("collapse", [[-1, 0], [0, 999], [2**32, 0]])
def test_replay_rejects_out_of_range_collapses(collapse):
    with pytest.raises(ValueError, match="collapse history is invalid"):
        mfs.replay_simplification(
            SIMPLE_POINTS, SIMPLE_FACES, np.asarray([collapse], dtype=np.int64)
        )


def test_replay_marks_unreferenced_vertices_unmapped():
    points = np.vstack([SIMPLE_POINTS[:3], [10.0, 10.0, 10.0]])
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    replay_points, replay_faces, mapping = mfs.replay_simplification(
        points, faces, np.empty((0, 2), dtype=np.int32)
    )
    np.testing.assert_array_equal(replay_points, points[:3].astype(np.float32))
    np.testing.assert_array_equal(replay_faces, faces)
    np.testing.assert_array_equal(mapping, [0, 1, 2, -1])


def test_replay_empty_mesh_does_not_cross_null_buffers():
    points, faces, mapping = mfs.replay_simplification(
        np.empty((0, 3)),
        np.empty((0, 3), dtype=np.int32),
        np.empty((0, 2), dtype=np.int32),
    )
    assert points.shape == (0, 3) and points.dtype == np.float32
    assert faces.shape == (0, 3) and faces.dtype == np.int32
    assert mapping.shape == (0,) and mapping.dtype == np.int64


def test_closed_manifold_collapse_count_matches_vertex_delta():
    points, faces = torus(18, 10)
    out_points, _, collapses = mfs.simplify(
        points, faces, target_count=180, return_collapses=True
    )
    assert len(out_points) == len(points) - len(collapses)


def test_attribute_weights_prevent_color_blending():
    points, faces = torus(40, 20)
    colors = (points[:, 0] > 0).astype(np.float64)
    _, _, unweighted = mfs.simplify(
        points,
        faces,
        0.95,
        attributes=colors,
        attribute_weights=0.0,
    )
    _, _, weighted = mfs.simplify(
        points,
        faces,
        0.95,
        attributes=colors,
        attribute_weights=1.0,
    )
    unweighted_mix = np.mean(np.minimum(np.abs(unweighted), np.abs(unweighted - 1)))
    weighted_mix = np.mean(np.minimum(np.abs(weighted), np.abs(weighted - 1)))
    assert unweighted_mix > 0.02
    assert weighted_mix == pytest.approx(0.0)


def test_multichannel_attributes_keep_shape_and_range():
    points, faces = torus(16, 10)
    attrs = np.column_stack([points, np.linspace(0, 1, len(points))])
    out_points, out_faces, out_attrs = mfs.simplify_with_attributes(
        points, faces, attrs, target_reduction=0.5, attribute_weights=[0, 0, 0, 1]
    )
    assert out_attrs.shape == (len(out_points), 4)
    assert out_faces.max() < len(out_points)
    assert np.all(np.isfinite(out_attrs))
    assert out_attrs[:, 3].min() >= 0 and out_attrs[:, 3].max() <= 1


def test_simd_attribute_tail_matches_scalar_channel():
    points, faces = torus(18, 10)
    signal = np.linspace(0, 1, len(points))
    scalar = mfs.simplify(
        points,
        faces,
        0.6,
        attributes=signal,
        attribute_weights=1.0,
        return_collapses=True,
    )
    padded = np.column_stack([np.zeros((len(points), 8)), signal])
    vector = mfs.simplify(
        points,
        faces,
        0.6,
        attributes=padded,
        attribute_weights=[0] * 8 + [1],
        return_collapses=True,
    )
    np.testing.assert_array_equal(vector[0], scalar[0])
    np.testing.assert_array_equal(vector[1], scalar[1])
    np.testing.assert_array_equal(vector[2][:, -1], scalar[2])
    np.testing.assert_array_equal(vector[3], scalar[3])


def test_output_is_deterministic():
    points, faces = torus(20, 12)
    first = mfs.simplify(points, faces, 0.7, return_collapses=True)
    second = mfs.simplify(points, faces, 0.7, return_collapses=True)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_output_closed_torus_remains_two_manifold():
    points, faces = torus(24, 16)
    _, out_faces = mfs.simplify(points, faces, 0.8)
    edges = np.sort(
        np.concatenate(
            [
                out_faces[:, [0, 1]],
                out_faces[:, [1, 2]],
                out_faces[:, [2, 0]],
            ]
        ),
        axis=1,
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2)
    assert len(np.unique(np.sort(out_faces, axis=1), axis=0)) == len(out_faces)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "You must specify"),
        ({"target_reduction": 0.5, "target_count": 2}, "but not both"),
        ({"target_reduction": 1.1}, "between 0 and 1"),
        ({"target_count": -1}, "greater than 0"),
        ({"target_count": 9}, "less than the number of faces"),
    ],
)
def test_target_validation_matches_upstream(kwargs, message):
    with pytest.raises(ValueError, match=message):
        mfs.simplify(SIMPLE_POINTS, SIMPLE_FACES, **kwargs)


def test_zero_reduction_and_target_count_are_mutually_exclusive():
    with pytest.raises(ValueError, match="but not both"):
        mfs.simplify(
            SIMPLE_POINTS,
            SIMPLE_FACES,
            target_reduction=0.0,
            target_count=4,
        )


def test_noncontiguous_inputs_and_int64_faces():
    points, faces = torus(16, 10)
    point_storage = np.empty((len(points), 6))
    point_storage[:, ::2] = points
    noncontiguous = point_storage[:, ::2]
    assert not noncontiguous.flags.c_contiguous
    out_points, out_faces = mfs.simplify(
        noncontiguous, faces.astype(np.int64), target_count=160
    )
    assert out_points.flags.c_contiguous
    assert out_faces.dtype == np.int32
    assert out_faces.shape == (160, 3)


def test_noninteger_faces_are_not_silently_narrowed():
    fractional = SIMPLE_FACES.astype(np.float64)
    fractional[0, 0] = 0.5
    with pytest.raises(TypeError, match="integer indices"):
        mfs.simplify(SIMPLE_POINTS, fractional, 0.5)


def test_simplify_mesh_preserves_numeric_point_data(monkeypatch):
    class PointData(dict):
        def __iter__(self):
            return iter(self.keys())

    class Mesh:
        points = SIMPLE_POINTS
        faces = np.column_stack(
            [np.full(len(SIMPLE_FACES), 3), SIMPLE_FACES]
        ).ravel()
        n_points = len(SIMPLE_POINTS)
        n_cells = len(SIMPLE_FACES)
        point_data = PointData(
            scalar=np.arange(len(SIMPLE_POINTS), dtype=np.float64),
            label=np.asarray(["x"] * len(SIMPLE_POINTS)),
        )

    class PolyData:
        def __init__(self, points, faces):
            self.points = points
            self.faces = faces
            self.point_data = PointData()
            self.field_data = {}

    monkeypatch.setitem(sys.modules, "pyvista", types.SimpleNamespace(PolyData=PolyData))
    output = mfs.simplify_mesh(Mesh(), target_reduction=0.5)
    assert output.points.shape == (5, 3)
    assert output.faces.reshape(-1, 4).shape == (4, 4)
    assert output.point_data["scalar"].shape == (5,)
    assert "label" not in output.point_data
    assert output.field_data["fast_simplification_collapses"].shape == (4, 2)


def test_verbose_reports_real_counts(capsys):
    mfs.simplify(SIMPLE_POINTS, SIMPLE_FACES, 0.5, verbose=True)
    text = capsys.readouterr().out
    assert "8 triangles to 4" in text


def test_invalid_shapes_and_indices_are_rejected():
    with pytest.raises(ValueError, match="points.*2 dimensional"):
        mfs.simplify(np.arange(9), SIMPLE_FACES, 0.5)
    with pytest.raises(ValueError, match="triangle index"):
        bad = SIMPLE_FACES.copy()
        bad[0, 0] = 99
        mfs.simplify(SIMPLE_POINTS, bad, 0.5)
    with pytest.raises(ValueError, match="distinct"):
        bad = SIMPLE_FACES.copy()
        bad[0] = [0, 0, 1]
        mfs.simplify(SIMPLE_POINTS, bad, 0.5)
    with pytest.raises(ValueError, match="finite"):
        bad = SIMPLE_POINTS.copy()
        bad[0, 0] = np.nan
        mfs.simplify(bad, SIMPLE_FACES, 0.5)
    with pytest.raises(ValueError, match="finite"):
        bad = np.arange(len(SIMPLE_POINTS), dtype=np.float64)
        bad[0] = np.inf
        mfs.simplify(SIMPLE_POINTS, SIMPLE_FACES, 0.5, attributes=bad)

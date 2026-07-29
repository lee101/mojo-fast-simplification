# mojo-fast-simplification

`mojo-fast-simplification` is a standalone Mojo port of the compute-heavy part
of [fast-simplification](https://github.com/pyvista/fast-simplification): quadric
error metric (QEM) decimation of indexed triangle meshes. It provides the same
top-level Python package name and the public array API used by upstream, while
running the decimation kernel from a compiled Mojo shared library.

This port prioritizes geometric correctness, stable topology, and vertex
attribute preservation. The optimized Mojo paths are competitive with the
mature upstream C++ implementation on the smaller simplification cases and
substantially faster for collapse replay; the largest decimation cases remain
slower.

## Coverage

Covered:

- `simplify(points, triangles, target_reduction=None, target_count=None,
  agg=7.0, verbose=False, return_collapses=False, lossless=False)`
- `replay_simplification(points, triangles, collapses)`
- `simplify_mesh(...)` for triangular PyVista meshes, including numeric point
  data
- the extension `simplify_with_attributes(...)`, and the equivalent
  `attributes=` and `attribute_weights=` keywords on `simplify`
- QEM plane quadrics, optimal contraction positions, lazy edge priorities,
  boundary checks, the manifold link condition, normal-flip rejection,
  collapse histories, and output compaction

Not covered:

- byte-for-byte reproduction of upstream's collapse order or output indexing
- upstream's private Cython modules and OBJ helpers
- preservation of PyVista cell data or nonnumeric point data
- upstream's exact 100-iteration aggression schedule; `agg` controls the
  admissible QEM-error envelope, but the Mojo implementation uses a priority
  queue
- detailed per-iteration verbose logging; `verbose=True` prints a final summary

For ordinary calls, the return types and tuple shape match upstream. When
`attributes` is supplied, the propagated attribute array is inserted after the
faces array. Attribute weights add a squared discontinuity penalty to the edge
priority, so color, UV, scalar-field, or feature seams can survive aggressive
reductions.

## Install

Clone the repository, enter its directory, and use the pinned Pixi environment:

```bash
pixi install
pixi run build
pixi run test
```

The compiled library is written to
`dist/libmojo-fast-simplification.so`. PyVista integration is optional; install
PyVista in the environment before calling `simplify_mesh`.
This release is source-checkout based; it does not publish a prebuilt wheel.

## Usage

```python
import numpy as np
import fast_simplification

points = np.array([
    [0.5, -0.5, 0.0], [0.0, -0.5, 0.0], [-0.5, -0.5, 0.0],
    [0.5,  0.0, 0.0], [0.0,  0.0, 0.0], [-0.5,  0.0, 0.0],
    [0.5,  0.5, 0.0], [0.0,  0.5, 0.0], [-0.5,  0.5, 0.0],
])
triangles = np.array([
    [0, 1, 3], [4, 3, 1], [1, 2, 4], [5, 4, 2],
    [3, 4, 6], [7, 6, 4], [4, 5, 7], [8, 7, 5],
], dtype=np.int32)
colors = (points[:, 0] > 0).astype(np.float64)

new_points, new_triangles, new_colors = fast_simplification.simplify(
    points,
    triangles,
    target_reduction=0.5,
    attributes=colors,
    attribute_weights=1.0,
)

assert new_points.shape == (5, 3)
assert new_triangles.shape == (4, 3)
assert new_colors.shape == (5,)
```

Removing the two attribute keywords gives the upstream-compatible two-value
return. Set `return_collapses=True` to append the `(n, 2)` collapse history.

## How it works

Python validates and converts points to contiguous `float64` and triangle
indices to contiguous `int32`, then allocates caller-owned output and scratch
storage.
Buffers cross the C ABI as integer addresses. The exported Mojo function
reconstructs them as `UnsafePointer[..., AnyOrigin[mut=True]]`; the library
does not retain pointers or allocate mesh-owned memory. Contiguous NumPy
buffers remain zero-copy across that boundary, and compacted outputs are views
of the working buffers rather than another full copy.

The kernel stores each symmetric 4-by-4 plane quadric as ten row-major scalar
coefficients per vertex. Faces and per-vertex attributes are flat row-major
arrays. A lazy binary heap holds edge errors with packed edge and vertex-version
pairs. Initial heap construction uses bottom-up linear-time heapification;
stale versioned edges are discarded and only the survivor's unique incident
edges are requeued.
Accepted contractions update the survivor's quadric and attributes, relink its
face incidence list, and enqueue only the affected one-ring. Link-condition,
boundary, degeneracy, and normal checks reject topology-damaging contractions.
The final pass compacts referenced vertices and remaps faces.

Quadric initialization and merge, attribute-error, interpolation, and
attribute-compaction loops use native-width `float64` SIMD blocks plus scalar
remainder loops.
There is no CPU thread-pool or GPU path: the dominant heap/topology collapse
sequence is dependent and irregular, while the independent initialization and
compaction passes are memory-bound or scatter-heavy and did not amortize thread
launch overhead. No kernel combines roughly 2 flops per byte with enough
independent work to justify device transfers, so adding a GPU path would lose
and was intentionally skipped.

Attribute values are linearly propagated at the chosen contraction position.
Their weights also contribute to edge priority; a zero weight propagates a
field without asking the simplifier to preserve its discontinuities.

## Benchmarks

Run only through `pixi run bench`; the task holds a machine-wide flock. These
are real median-of-three measurements from 2026-07-29 on an Intel Xeon
E5-2697 v4 at 2.30 GHz, Linux x86_64, Python 3.13.14, Mojo
1.0.0b3.dev2026072406, and upstream fast-simplification 0.1.13.

The ratio is upstream time divided by Mojo time, so values below `1.00x` mean
Mojo is slower.

| case | Mojo | upstream | upstream / Mojo |
|---|---:|---:|---:|
| simplify 5,000 faces to 50% | 3.59 ms | 2.93 ms | 0.82x |
| simplify 20,000 faces to 25% | 24.60 ms | 14.32 ms | 0.58x |
| simplify 20,000 faces to 5% | 32.45 ms | 17.11 ms | 0.53x |
| attribute-preserving 5,000 faces to 10% | 6.84 ms | 7.48 ms | 1.09x |
| replay 5,000-face collapse history | 733.5 us | 3.43 ms | 4.68x |

The upstream attribute baseline uses its documented collapse-history replay
and correspondence mapping, followed by a NumPy group mean. Upstream has no
attribute-aware edge-error parameter. Results are mixed: Mojo is 1.09 times
faster for the attribute-preserving case and 4.68 times faster for replay.
Upstream is 1.22 to 1.89 times faster for the ordinary simplification cases.

## Verification

The 30-test pytest suite compares directly with installed upstream 0.1.13. Assertions
cover requested face and vertex counts, QEM surface residuals, mesh extents,
lossless behavior, dtypes and validation, deterministic output, two-manifold
topology, exact local replay, weighted attribute seams, and SIMD remainder
handling.

## License

MIT. The public API and algorithmic behavior were checked against PyVista's
MIT-licensed fast-simplification project and Sven Forstmann's
Fast-Quadric-Mesh-Simplification implementation.

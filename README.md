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
There is no CPU thread-pool and no GPU path; see *GPU phase* below for the
measurements behind that decision.

## GPU phase

CPU only: memory-bound. There is no GPU kernel in this port, and none is
planned at the current kernel shapes.

The gate for this phase is roughly 2 flops per byte for a kernel to be a GPU
candidate. The flop and byte counts below were taken by reading each
expression in `src/simplify.mojo` and counting every multiply, add, subtract,
divide, and compare on the element's own data; bytes are the distinct bytes the
element must touch, counting read-modify-write quadric traffic once as read.

| kernel | flops per element | bytes per element | flops / byte |
|---|---:|---:|---:|
| `candidate` (QEM evaluate + optimal position) | ~120 | 224 | 0.54 |
| `candidate`, counting only the 2x10 quadric block it must read | ~120 | 160 | 0.75 |
| `face_start` (normal, plane, 3x quadric accumulate) | ~30 | 564 | 0.05 |
| `total_error` attribute penalty | 3 | 24 | 0.13 |
| quadric merge / attribute interpolation / compaction | <=6 | 16-80 | <=0.4 |

`candidate` is the most compute-dense kernel and it is still roughly three to
four times below the threshold: about 120 flops of contraction evaluation are
fed by two 160-byte quadrics, and the arithmetic finishes well before the
operands arrive from memory. The remaining
passes are outright memory-bound, so a device copy in and out would cost more
than the compute it replaces. The dominant collapse loop is also a dependent
sequence of heap pops, link-condition counts, and relinks, which a
per-element kernel cannot cover in one launch.

Host check at the time of this phase (2026-11-11, NVIDIA GeForce RTX 5090,
32607 MiB total): `nvidia-smi --query-gpu=memory.free --format=csv,noheader`
reported 4984 MiB free at the first check and 5294 MiB at the second, both
above the 4000 MiB floor. The phase was therefore not skipped for memory
reasons; it stopped on the flops-per-byte gate.

Because no GPU code was added, there are no GPU tests to skip on a CPU-only
host, and the CPU path, its tests, and the benchmark timings are unchanged from
the numbers above.

Attribute values are linearly propagated at the chosen contraction position.
Their weights also contribute to edge priority; a zero weight propagates a
field without asking the simplifier to preserve its discontinuities.

## Benchmarks

Run only through `pixi run bench`; the task holds a machine-wide flock. These
are real median-of-three measurements from 2026-08-24 on an Intel Xeon
E5-2697 v4 at 2.30 GHz, Linux x86_64, Python 3.13.14, Mojo
1.1.0.dev2026081105, and upstream fast-simplification 0.1.13.

The ratio is upstream time divided by Mojo time, so values below `1.00x` mean
Mojo is slower.

| case | Mojo | upstream | upstream / Mojo |
|---|---:|---:|---:|
| simplify 5,000 faces to 50% | 3.64 ms | 3.10 ms | 0.85x |
| simplify 20,000 faces to 25% | 26.29 ms | 14.81 ms | 0.56x |
| simplify 20,000 faces to 5% | 35.17 ms | 17.75 ms | 0.50x |
| attribute-preserving 5,000 faces to 10% | 7.23 ms | 8.30 ms | 1.15x |
| replay 5,000-face collapse history | 754.2 us | 3.81 ms | 5.05x |

The upstream attribute baseline uses its documented collapse-history replay
and correspondence mapping, followed by a NumPy group mean. Upstream has no
attribute-aware edge-error parameter. Results are mixed: Mojo is 1.15 times
faster for the attribute-preserving case and 5.05 times faster for replay.
Upstream is 1.18 to 2.00 times faster for the ordinary simplification cases.

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

# VERIFY phase: reproducing the README benchmarks

Run on 2026-11-11, workspace `/nvme0n1-disk/code/mojo-fast-simplification`,
branch `master` at `fdc9783` ("phase gpu"). The working tree was clean at the
start, and is clean at the end apart from the README and this file.

Host: Intel Xeon E5-2697 v4 @ 2.30 GHz, Linux x86_64, Python 3.13.14,
Mojo 1.2.0.dev2026092605, upstream fast-simplification 0.1.13.
The host is shared. `pixi run bench` calls `flock /tmp/mojo-bench.lock`, and
during this session that lock was held by at least eight other processes from
several unrelated checkouts. Every `pixi run bench` here therefore ran after a
variable wait for the lock, which is the most likely source of cold-start
run-to-run noise below.

## What I ran

1. `pixi run build && pixi run test`: build exit 0 (only deprecation warnings,
   e.g. `UnsafePointer` -> `Pointer`), and `30 passed`.
2. `pixi run bench`: completed after a lock wait, with no table printed.
   Re-running `pixi run bench > log 2>&1` hung behind the same lock: that run
   sat for over an hour. Running `python bench/bench.py` directly, from inside
   the pixi environment, completed each time (about 1 s). All table figures in
   this file come from the same script, run 5 times back to back immediately
   before the README was updated.
3. Re-ran every README row, plus the attribution checks described below.

## Reproduced (median of 5 runs, each run a median of 3)

| case | Mojo (README -> measured) | upstream (README -> measured) | ratio (README -> measured) | change |
|---|---:|---|---:|---|
| simplify 5,000 faces to 50% | 3.64 -> 4.16 ms | 3.10 -> 3.52 ms | 0.85x -> 0.85x | corrected (~14%) |
| simplify 20,000 faces to 25% | 26.29 -> 27.00 ms | 14.81 -> 15.33 ms | 0.56x -> 0.57x | corrected (~3%) |
| simplify 20,000 faces to 5% | 35.17 -> 34.96 ms | 17.75 -> 17.96 ms | 0.50x -> 0.51x | corrected (<1%) |
| attribute-preserving 5,000 faces to 10% | 7.23 -> 7.30 ms | 8.30 -> 8.09 ms | 1.15x -> 1.11x | corrected (upstream ~2.5%, ratio ~3.5%) |
| replay 5,000-face collapse history | 754.2 -> 790.5 us | 3.81 -> 3.83 ms | 5.05x -> 4.85x | corrected (~4%) |

Run ranges: 5,000 faces 3.56-4.44 ms (Mojo) and 2.93-3.80 ms (upstream);
20,000 faces to 25% 26.70-33.80 ms (Mojo) and 14.62-17.61 ms (upstream).
The table uses medians of these runs.

The README header was also wrong. It cited date 2026-08-24, Mojo
`1.1.0.dev2026081105` (the `pixi.toml` lock is `1.2.0.dev2026092605`), and
"median-of-three"; the bench's first measurement is a cold one, so a
median of the five runs' medians was used instead. All of that is corrected.

Slower rows stayed slower: the three ordinary simplification rows remain below
`1.00x`, unchanged in direction, and the prose now states the 1.18-1.96 range
from the measured ratios (0.85, 1/0.57, 1/0.51).

## Same input, same work on both sides?

I compared actual output sizes for every row, not just the printed digest.

| row | Mojo output | upstream output |
|---|---|---|
| simplify 5k 50% | 2,500 faces | 2,500 faces |
| simplify 20k 25% | 5,000 faces | 5,000 faces |
| simplify 20k 5% | 1,000 faces | 1,000 faces |
| attribute 5k 10% | 498 faces | 498 faces |
| replay | 1,250 faces, 625 points, 1,875 collapses | 1,250 faces, 625 points, 1,875 collapses |

Both sides receive the same `torus(50, 50)` or `torus(100, 100)` arrays in
every row, and neither side gets extra warm-up. The per-call work differs in
structurally honest ways, and is noted rather than changed:

- The replay row replays each library's *own* collapse history, each length
  1,875 and each from the same input mesh and reduction. Upstream's
  `replay_simplification` is called with a `float32` copy of the points, while
  the Mojo side is called with the `float64` points. The measured cost of that
  cast alone is about 4 us, so it does not explain the gap.
- The Mojo side copies and validates its input on every call; the upstream side
  runs `np.asarray` on already-`float64`/`int32` arrays, which copies nothing.
  On these inputs that per-call cost is about 50 us plus copies (measured: the
  Python-side finite/face-index checks cost about 54 us for 5,000 faces). It is
  not removed: it is a deliberate input contract, and the tests pin it (see
  below).
- The upstream side reloads its whole mesh on every call (`load_int32` is
  about 0.19 ms at 5,000 faces and 0.74 ms at 20,000); the Mojo side does not.
  Both are compared on the same input with no work skipped.
- The upstream attribute baseline is `upstream.simplify` with replay plus a
  NumPy group mean. Upstream has no attribute-aware simplify; the Mojo side
  passes `attribute_weights=1.0`, which I confirmed changes collapse choice
  (its history differs from the unweighted one on the same mesh). Both sides
  return 498 faces. This is the single least like-for-like row, and it is now
  labelled as such in the README rather than reworded to look better.

I also checked the earlier phases' optimizations for a benchmark-only shortcut:
I split `face_start` into two passes (face linking, then quadric accumulation).
That has identical semantics, passes all 30 tests, and benches within noise of
the original. Nothing rests on a special case. During that check the tree was
restored to the original file, and `src/simplify.mojo` is unmodified
(`git diff` empty).

## GPU section (also checked, since it cites "the numbers above")

- "No CPU thread-pool and no GPU path": no threading or GPU code exists in
  `src/`, `python/`, or `pixi.toml`, so the claim holds.
- The flop/byte table says it was taken "by reading each expression in
  `src/simplify.mojo`". Only `face_start` changed since; `candidate` and the
  attribute loop are byte-identical to the commit they were counted at
  (`git log -p --follow -- src/simplify.mojo`). The counts themselves are
  analytical and not reproducible by measurement here.
- The RTX 5090 host check reproduces today:
  `nvidia-smi --query-gpu=name,memory.total,memory.free` reports
  `NVIDIA GeForce RTX 5090, 32607 MiB, 5294 MiB` (the free figure is the last
  of the two readings, as above).
- One README sentence claimed "no GPU code was added... and the benchmark
  timings are unchanged from the numbers above" even though the table below it
  had changed between commits. Replaced with a factual version-constrained
  sentence.

## Tests still assert parity

- `git log -p -- tests/` shows a single commit adding `tests/conftest.py` and
  `tests/test_parity.py`, with no later modification; no assertions were
  removed and no tolerances were loosened. Nothing to restore there.
- I checked the other direction instead, whether the tests still bite. Removing
  the `np.all(np.isfinite(...))` check and switching the input copies to
  zero-copy `ascontiguousarray` fails 4 tests
  (`test_collapse_replay_matches_simplification`,
  `test_simd_attribute_tail_matches_scalar_channel`,
  `test_output_is_deterministic`,
  `test_invalid_shapes_and_indices_are_rejected`). The file was restored
  immediately; the tree is clean.
- All parity tests compare against genuine upstream 0.1.13 (via a subprocess
  with `PYTHONPATH` stripped). Note for future phases: inside this workspace
  the pixi environment puts `./python` ahead of site-packages, so
  `import fast_simplification` resolves to this port. Upstream must be loaded
  from `.pixi/envs/default/lib/python3.13/site-packages` or from outside the
  workspace, as `bench.py` and the fixture already do.

## What I could not reproduce

- The exact 2026-08-24 per-run figures. Their source tree (Mojo 1.1.0 toolchain,
  no `bench/bench.py` at that revision) is not present, so I did not try to
  rebuild it. The README now carries only numbers measured in this session, on
  the current toolchain, and says so.
- Noise-free single-run numbers: the shared host's lock queue and CPU
  contention gave 15-25% run-to-run spread on the smallest case. The table uses
  medians of five runs and states the observed range.

## Changes made

- `README.md` benchmarks table, header, and prose: every figure replaced with
  the measured median of five runs, toolchain/date corrected, per-call
  asymmetry caveats added, slower rows kept and still marked slower.
- `README.md` intro ("largest decimation cases remain slower" clarified) and the
  GPU-section closing paragraph rewritten to state what was and wasn't
  measured.
- `bench/verify.md`: this file.
- No source, test, or benchmark-script changes.

## Final gate

`pixi run build && pixi run test`: command exit 0, and `30 passed in 0.73s`.
The build emits only deprecation warnings and no errors. This was the last
command run on the tree. `git status` shows only `M README.md` and the new,
untracked `bench/verify.md`; no source, test, or bench-script files changed,
and nothing was committed.

# Kernel profile before the unroll phase

No `bench/analysis.md` existed for this port, so this file records the starting
point for the UNROLL phase.

## Wall clock (`pixi run bench`, median of 3, Xeon E5-2697 v4)

| case | Mojo | upstream | upstream / Mojo |
|---|---:|---:|---:|
| simplify 5,000 faces to 50% | 3.70 ms | 3.38 ms | 0.91x |
| simplify 20,000 faces to 25% | 26.21 ms | 15.59 ms | 0.59x |
| simplify 20,000 faces to 5% | 34.68 ms | 25.50 ms | 0.74x |
| attribute-preserving 5,000 faces to 10% | 7.24 ms | 8.29 ms | 1.14x |
| replay 5,000-face collapse history | 825.3 us | 3.87 ms | 4.68x |

The worst rows are the two 20,000-face cases, so the profile below targets them.

## Phase split

`perf` is unavailable here (`perf_event_paranoid`), so the kernel was
instrumented temporarily with `perf_counter_ns` brackets, run with the counters
off and on, and then the instrumentation was removed. One-shot timers inflate
the run, so treat the shares, not the absolute times, as the signal.

Median of 7 single-invocation runs, `torus(100, 100)`:

| phase | to 25% (red 0.75) | to 5% (red 0.95) |
|---|---:|---:|
| vertex init | 1.7% | 1.1% |
| face link + quadric build | 3.6% | 2.6% |
| border detection | 4.9% | 3.5% |
| initial heap build | 22% | 23% |
| collapse loop, of which: | 68% | 70% |
| - heap pop and sift-down | 27% | 29% |
| - candidate (QEM evaluate) | 8% | 8% |
| - collapse validity | 11% | 10% |
| - requeue `total_error` | 11% | 11% |
| - splice / relink | 5% | 5% |
| compaction | 1.3% | 1.2% |

Skips are small: ~32k pops, of which 18k are stale-version, 1 stale/alive and
essentially none rejected by validity.

## Targets for this phase

1. `heap_pop` / the sift-down loop — largest single block.
2. The requeue path via `total_error` -> `candidate` -> `qvalue`, which reloads
   the same ten quadric coefficients three times per call.
3. `collapse_valid`'s link-condition count, a serial integer dependency.
4. The fixed 10-trip quadric loops, which are compile-time trip counts.

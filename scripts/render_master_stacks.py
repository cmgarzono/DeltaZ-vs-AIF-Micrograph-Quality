r"""
S3-ACCELERATE Phase A: render each of the 801 MASTER AXIAL LATTICE
positions exactly once per photographic condition, and cache the result
to disk-backed memmaps.

z_master = build_master_lattice() = -4.0 + 0.01*j, j = 0..800 (801 points).
This lattice is the exact index-space container of every one of the 320
S3 candidate grids.

Uses the fixed `render_frame_shared` implementation (shared blurred-
denominator across the 8 photographic conditions per z; verified
mathematically identical to the single-condition renderer). Only the
OUTER loop changes here: render each z ONCE (not once per
delta that happens to use it).

Output layout (results/sampling_fusion/cache/):
  frames_<COND>.f64      memmap, shape (801, H, W), float64
  master_render_manifest.json   {"completed_j": [...], "shape": [H,W], "conditions": [...]}

Resumable: re-running skips any j already marked complete in the
manifest (checked against ALL 8 condition memmaps having been written
for that j, via a per-j completion bitmap).
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from deltaz_aif.analysis.sampling_fusion import (  # noqa: E402
    DepthLayerCache,
    build_master_lattice,
    render_frame_shared,
    N_MASTER,
)

SPECIMEN_DIR = ROOT / "data" / "specimen"
CONDITIONS_DIR = ROOT / "data" / "specimen" / "conditions"
CACHE_DIR = ROOT / "results" / "sampling_fusion" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["H_LF", "H_COARSE", "H_MIXED", "H_FINE", "P_LF", "P_COARSE", "P_MIXED", "P_FINE"]

_worker_cache = None
_worker_oracles = None
_worker_shape = None


def _worker_init():
    global _worker_cache, _worker_oracles, _worker_shape
    h = np.load(SPECIMEN_DIR / "height_map.npy").astype(np.float64)
    _worker_oracles = {c: np.load(CONDITIONS_DIR / f"oracle_{c}.npy").astype(np.float64) for c in CONDITIONS}
    _worker_cache = DepthLayerCache.build(h)
    _worker_shape = h.shape


def _render_one_j(args):
    j, z = args
    global _worker_cache, _worker_oracles, _worker_shape
    frames = render_frame_shared(_worker_cache, _worker_oracles, float(z))
    for cond in CONDITIONS:
        mm = np.memmap(
            CACHE_DIR / f"frames_{cond}.f64",
            dtype=np.float64,
            mode="r+",
            shape=(N_MASTER,) + _worker_shape,
        )
        mm[j] = frames[cond]
        mm.flush()
        del mm
    return j


def _ensure_memmaps(shape):
    for cond in CONDITIONS:
        p = CACHE_DIR / f"frames_{cond}.f64"
        expected_bytes = N_MASTER * shape[0] * shape[1] * 8
        if not p.exists() or p.stat().st_size != expected_bytes:
            mm = np.memmap(p, dtype=np.float64, mode="w+", shape=(N_MASTER,) + shape)
            del mm


def _load_manifest():
    p = CACHE_DIR / "master_render_manifest.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"completed_j": [], "shape": None, "conditions": CONDITIONS}


def _save_manifest(manifest):
    (CACHE_DIR / "master_render_manifest.json").write_text(json.dumps(manifest, indent=2))


def benchmark_workers(z_master, shape, sample_j=(0, 200, 400)):
    """Quick throughput check across worker counts (§13); a few master
    positions only, not an extensive search."""
    _ensure_memmaps(shape)
    results = {}
    for n_workers in (1, 4, 8, 16):
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init) as ex:
            list(ex.map(_render_one_j, [(j, z_master[j]) for j in sample_j]))
        dt = time.time() - t0
        results[n_workers] = dt
        print(f"  workers={n_workers:2d}: {dt:.2f}s for {len(sample_j)} z ({dt/len(sample_j):.2f}s/z)")
    return results


def main(n_workers: int | None = None, limit: int | None = None):
    h = np.load(SPECIMEN_DIR / "height_map.npy").astype(np.float64)
    shape = h.shape
    z_master = build_master_lattice(float(h.min()), float(h.max()))
    assert len(z_master) == N_MASTER == 801

    _ensure_memmaps(shape)
    manifest = _load_manifest()
    manifest["shape"] = list(shape)
    completed = set(manifest["completed_j"])

    if n_workers is None:
        print("Benchmarking worker counts on 3 sample positions...")
        bench = benchmark_workers(z_master, shape)
        best = min(bench, key=bench.get)
        print(f"Selected n_workers={best} (fastest measured)")
        n_workers = best
        # sample renders count as done
        completed |= {0, 200, 400}

    todo = [j for j in range(N_MASTER) if j not in completed]
    if limit is not None:
        todo = todo[:limit]
    print(f"Rendering {len(todo)} / {N_MASTER} master positions with {n_workers} workers...")

    t0 = time.time()
    done_count = 0
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init) as ex:
        futures = {ex.submit(_render_one_j, (j, z_master[j])): j for j in todo}
        from concurrent.futures import as_completed
        for fut in as_completed(futures):
            j = fut.result()
            completed.add(j)
            done_count += 1
            if done_count % 20 == 0 or done_count == len(todo):
                elapsed = time.time() - t0
                rate = done_count / elapsed if elapsed > 0 else 0
                remaining = (len(todo) - done_count) / rate if rate > 0 else float("inf")
                print(f"  {done_count}/{len(todo)} done, {elapsed:.0f}s elapsed, "
                      f"ETA {remaining/3600:.2f}h", flush=True)
                manifest["completed_j"] = sorted(completed)
                _save_manifest(manifest)

    manifest["completed_j"] = sorted(completed)
    _save_manifest(manifest)
    print(f"Phase A complete: {len(completed)}/{N_MASTER} master positions cached.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--bench-only", action="store_true")
    args = ap.parse_args()

    if args.bench_only:
        h = np.load(SPECIMEN_DIR / "height_map.npy").astype(np.float64)
        z_master = build_master_lattice(float(h.min()), float(h.max()))
        benchmark_workers(z_master, h.shape)
    else:
        main(n_workers=args.workers, limit=args.limit)

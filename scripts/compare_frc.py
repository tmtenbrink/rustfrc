"""Times frc.one_frc (method 1: rustfrc.binom_split, DIPlib's FFTs and RadialSum) against
rustfrc's one_frc on the same images, and compares their curves and how they bin rings.

    uv run --with frc --with diplib --with scipy python scripts/compare_frc.py [--sizes ...]

Set RAYON_NUM_THREADS before starting it to limit rustfrc's threads; --threads sets DIPlib's.
"""

import argparse
import importlib.metadata
import os
import time

import diplib as dip
import frc
import numpy as np

import rustfrc

WAVES = [(3, 5, 0.4), (17, -9, 0.2), (40, 33, 0.1), (120, 7, 0.05)]


def pattern_image(n: int, mean: float, seed: int) -> np.ndarray:
    """Poisson counts around mean with cosine waves at rings 5, 19, 51 and 120."""
    y, x = np.mgrid[0:n, 0:n]
    waves = sum(a * np.cos(2 * np.pi * (u * x + v * y) / n) for u, v, a in WAVES)
    return np.random.default_rng(seed).poisson(mean * (1 + waves)).astype(np.int32)


def median_time(reps: int, f) -> float:
    f()
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        f()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def ring_counts(n: int, rounding) -> np.ndarray:
    f = np.fft.fftfreq(n, 1 / n)
    r = rounding(np.hypot(f[:, None], f[None, :])).astype(int).ravel()
    return np.bincount(r[r < n // 2], minlength=n // 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2048, 4000, 4096, 8192])
    parser.add_argument("--threads", type=int, default=0, help="DIPlib's threads; 0 leaves its default")
    parser.add_argument("--mean", type=float, default=100.0)
    args = parser.parse_args()
    if args.threads:
        dip.SetNumberOfThreads(args.threads)
    print(
        f"DIPlib {dip.__version__} with {dip.GetNumberOfThreads()} threads, frc {importlib.metadata.version("frc")}, rustfrc {importlib.metadata.version("rustfrc")}, "
        f"RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS', 'all')}, mean {args.mean}\n"
    )
    print("| n | frc.one_frc | rustfrc.binom_split alone | FrcPlan(n) | plan.one_frc | rustfrc.one_frc | speedup |")
    print("|--:|--:|--:|--:|--:|--:|--:|")
    checks = []
    for n in args.sizes:
        reps = 7 if n <= 4096 else 3
        img = pattern_image(n, args.mean, n)
        old = median_time(reps, lambda: frc.one_frc(img))
        split = median_time(reps, lambda: rustfrc.binom_split(img))
        start = time.perf_counter()
        plan = rustfrc.FrcPlan(n)
        make_plan = time.perf_counter() - start
        new = median_time(reps, lambda: plan.one_frc(img))
        new_with_plan = median_time(reps, lambda: rustfrc.one_frc(img))
        ms = lambda t: f"{t * 1e3:.1f} ms"
        print(f"| {n} | {ms(old)} | {ms(split)} | {ms(make_plan)} | {ms(new)} | {ms(new_with_plan)} | {old / new:.0f}x |")

        old_curve = np.asarray(frc.one_frc(img))
        new_curve = plan.one_frc(img, seed=1)
        dip_counts = np.asarray(dip.RadialSum(np.ones((n, n)), None)).astype(int)[: n // 2]
        checks.append((n, old_curve, new_curve, dip_counts, plan.ring_counts().astype(int)))

    print("\nCurves (different random splits, so they agree within noise) and ring binning:\n")
    print("| n | rings | old at 5, 19, 51, 120 | new at 5, 19, 51, 120 | RMS(old - new), rings 130+ | ring counts = DIPlib's | DIPlib = floor(abs f) |")
    print("|--:|--:|--|--|--:|--|--|")
    for n, old_curve, new_curve, dip_counts, ours in checks:
        rings = [r for r in (5, 19, 51, 120) if r < n // 2]
        fmt = lambda c: ", ".join(f"{c[r]:.3f}" for r in rings)
        rest = slice(130, min(len(old_curve), len(new_curve)))
        rms = float(np.sqrt(np.mean((old_curve[rest] - new_curve[rest]) ** 2))) if n > 260 else float("nan")
        same_counts = np.array_equal(dip_counts, ours)
        same_floor = np.array_equal(dip_counts, ring_counts(n, np.floor))
        print(f"| {n} | {len(old_curve)} / {len(new_curve)} | {fmt(old_curve)} | {fmt(new_curve)} | {rms:.4f} | {same_counts} | {same_floor} |")


if __name__ == "__main__":
    main()

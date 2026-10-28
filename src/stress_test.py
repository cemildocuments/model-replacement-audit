"""Controlled finite-population stress scenarios, parallelized by composition."""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import numpy as np
import pandas as pd
from bounds import paired_upper

ROOT = Path(__file__).resolve().parents[1]


def scenario(spec):
    D, zero_fraction, gap = spec
    N, reps = 10000, 20000
    budgets = np.array([16, 32, 64, 128, 256])
    zeros = round(D * zero_fraction)
    plus = round((D - zeros + gap * N) / 2)
    minus = D - zeros - plus
    if minus < 0:
        return None
    truth = (plus - minus) / N
    rng = np.random.default_rng(314159 + D + plus + zeros)
    remaining = np.tile([plus, minus, zeros], (reps, 1))
    cumulative = np.zeros((reps, 3), dtype=int)
    last = 0
    good = np.ones(reps, dtype=bool)
    naivepass = np.zeros(reps, dtype=bool)
    singlebad = np.zeros(reps, dtype=bool)
    cert = np.zeros(reps, dtype=bool)
    for B in budgets:
        n = min(B, D)
        step = n - last
        # Two conditional hypergeometric draws give an exact multivariate draw.
        total = remaining.sum(axis=1)
        xp = rng.hypergeometric(remaining[:, 0], total - remaining[:, 0], step)
        xm = rng.hypergeometric(remaining[:, 1], remaining[:, 2], step - xp)
        draws = np.stack([xp, xm, step - xp - xm], axis=1)
        remaining -= draws
        cumulative += draws
        last = n
        U = paired_upper(D, n, cumulative[:, 0], cumulative[:, 1], N, 0.05 / (2 * 5))
        good &= U + 1e-12 >= truth
        cert |= U <= 0.01 + 1e-12
        U_single = paired_upper(D, n, cumulative[:, 0], cumulative[:, 1], N, 0.05 / 2)
        singlebad |= U_single + 1e-12 < truth
        estimate = D / N * (cumulative[:, 0] - cumulative[:, 1]) / n
        naivepass |= estimate <= 0.01 + 1e-12
    return dict(
        N=N,
        D=D,
        zeros=zeros,
        plus=plus,
        minus=minus,
        truth=truth,
        repetitions=reps,
        simultaneous_undercoverage=float(1 - good.mean()),
        uncorrected_peeking_undercoverage=float(singlebad.mean()),
        false_approval=float(cert.mean()) if truth > 0.01 + 1e-12 else 0,
        empirical_false_approval=float(naivepass.mean()) if truth > 0.01 + 1e-12 else 0,
        certify_rate=float(cert.mean()),
    )


def run(jobs=6):
    specs = [
        (D, z, gap)
        for D in [100, 300, 1000, 3000]
        for z in [0.0, 0.4]
        for gap in [0.0, 0.005, 0.01, 0.011, 0.02]
    ]
    # map preserves scenario order; each composition has its own fixed RNG seed.
    out = []
    with ProcessPoolExecutor(jobs) as pool:
        for value in pool.map(scenario, specs):
            if value is not None:
                out.append(value)
                print(f"Completed stress composition {len(out)}/34", flush=True)
    pd.DataFrame(out).to_csv(ROOT / "results/stress.csv", index=False)
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    run(args.jobs)

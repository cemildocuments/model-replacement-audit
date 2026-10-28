import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from pathlib import Path
import argparse, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from bounds import paired_upper

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = np.array([16, 32, 64, 128, 256])
TREES = np.array([8, 16, 32, 64])
ALPHA = 0.05 / (2 * 4 * len(BUDGETS))
METHODS = [
    "uniform_paired",
    "uniform_conditional",
    "disagreement_exact",
    "disagreement_binomial",
    "disagreement_harm",
]


def audit_file(arg):
    path, reps = arg
    d = np.load(path)
    y = d["y"]
    pred = d["pred"]
    N = len(y)
    M = 4
    diff = (pred[:, :M] != y[:, None]).astype(np.int8) - (
        pred[:, M:] != y[:, None]
    ).astype(np.int8)
    disagree = pred[:, :M] != pred[:, M:]
    D = disagree.sum(axis=0)
    union = np.flatnonzero(disagree.any(axis=1))
    U = len(union)
    assert np.all(diff[~disagree] == 0)
    bounds = np.empty((reps, 5, len(BUDGETS), M))
    queries = np.empty((reps, 5, len(BUDGETS)), dtype=np.int32)
    counts = np.zeros((2, reps, len(BUDGETS), M, 3), dtype=np.int32)
    for r in range(reps):
        rng = np.random.default_rng(20261027 + r)
        orders = [rng.permutation(N), rng.permutation(union)]
        for scheme, order in enumerate(orders):
            for b, B in enumerate(BUDGETS):
                ix = order[:B]
                counts[scheme, r, b, :, 0] = disagree[ix].sum(axis=0)
                counts[scheme, r, b, :, 1] = (diff[ix] == 1).sum(axis=0)
                counts[scheme, r, b, :, 2] = (diff[ix] == -1).sum(axis=0)
    for b, B in enumerate(BUDGETS):
        n = np.full((reps, M), min(B, N))
        p = counts[0, :, b, :, 1]
        q = counts[0, :, b, :, 2]
        bounds[:, 0, b] = np.minimum(D / N, paired_upper(N, n, p, q, N, ALPHA))
        n = counts[0, :, b, :, 0]
        bounds[:, 1, b] = np.minimum(D / N, paired_upper(D, n, p, q, N, ALPHA))
        n = counts[1, :, b, :, 0]
        p = counts[1, :, b, :, 1]
        q = counts[1, :, b, :, 2]
        for j, method in enumerate(["hypergeom", "binomial", "harm"], 2):
            bounds[:, j, b] = np.minimum(
                D / N, paired_upper(D, n, p, q, N, ALPHA, method)
            )
        queries[:, :2, b] = min(B, N)
        queries[:, 2:, b] = min(B, U)
    # Running minimum includes the label-free bound and is justified jointly across looks.
    bounds = np.minimum.accumulate(bounds, axis=2)
    # A completed disagreement census is exact and supersedes earlier confidence bounds.
    # This prevents retaining a rare earlier statistical error after all relevant labels are known.
    for m in range(4):
        scheme = 0 if m < 2 else 1
        census = counts[scheme, :, :, :, 0] == D
        observed = (counts[scheme, :, :, :, 1] - counts[scheme, :, :, :, 2]) / N
        bounds[:, m] = np.where(census, observed, bounds[:, m])
    out = ROOT / "results/audits"
    out.mkdir(exist_ok=True)
    np.savez_compressed(
        out / path.name,
        bounds=bounds,
        queries=queries,
        truth=diff.mean(axis=0),
        D=D,
        N=N,
        U=U,
        budgets=BUDGETS,
        trees=TREES,
        methods=METHODS,
        counts=counts,
    )
    return dict(
        task=path.stem,
        N=N,
        U=U,
        disagreement=D.tolist(),
        truth=diff.mean(axis=0).tolist(),
        unsafe=int((diff.mean(axis=0) > 0.01).sum()),
        certified_rate=(bounds[:, 2, -1] <= 0.01).mean(),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--reps", type=int, default=500)
    a = p.parse_args()
    start = time.perf_counter()
    res = []
    with ProcessPoolExecutor(a.jobs) as pool:
        for f in as_completed(
            [
                pool.submit(audit_file, (p, a.reps))
                for p in sorted((ROOT / "results/predictions").glob("*.npz"))
            ]
        ):
            r = f.result()
            res.append(r)
            print(json.dumps(r), flush=True)
    (ROOT / "results/audit_manifest.json").write_text(
        json.dumps(
            dict(
                reps=a.reps,
                wall_seconds=time.perf_counter() - start,
                alpha=ALPHA,
                delta=0.05,
                budgets=BUDGETS.tolist(),
                methods=METHODS,
                runs=sorted(res, key=lambda r: r["task"]),
            ),
            indent=2,
        )
    )

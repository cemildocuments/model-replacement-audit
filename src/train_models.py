import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
from pathlib import Path
import json, time, argparse, resource
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GroupShuffleSplit
from concurrent.futures import ProcessPoolExecutor, as_completed
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]


def task(spec):
    name, seed, family = spec
    started = time.perf_counter()
    d = np.load(ROOT / "data/processed" / f"{name}.npz")
    X, y, g = d["X"], d["y"], d["groups"]
    tr, te = next(
        GroupShuffleSplit(n_splits=1, train_size=0.6, random_state=seed).split(X, y, g)
    )
    assert not set(g[tr]) & set(g[te])
    cls = RandomForestClassifier if family == "RF" else ExtraTreesClassifier
    with threadpool_limits(limits=1):
        model = cls(
            n_estimators=128,
            max_features="sqrt",
            min_samples_leaf=1,
            random_state=seed,
            n_jobs=1,
        ).fit(X[tr], y[tr])
        proba = np.zeros((len(te), len(model.classes_)))
        preds = []
        nodes = []
        for j, tree in enumerate(model.estimators_, 1):
            proba += tree.predict_proba(X[te])
            nodes.append(tree.tree_.node_count)
            if j in [8, 16, 32, 64, 128]:
                preds.append(model.classes_[proba.argmax(axis=1)])
    out = ROOT / "results/predictions"
    out.mkdir(exist_ok=True)
    np.savez_compressed(
        out / f"{name}_{family}_{seed}.npz",
        y=y[te],
        pred=np.stack(preds, axis=1),
        row_id=te,
        train_row_id=tr,
        nodes=np.cumsum(nodes)[np.array([8, 16, 32, 64, 128]) - 1],
    )
    record = dict(
        dataset=name,
        seed=seed,
        family=family,
        n_train=len(tr),
        n_pool=len(te),
        seconds=time.perf_counter() - started,
        max_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        accuracy=(np.stack(preds, axis=1) == y[te, None]).mean(axis=0).tolist(),
    )
    return record


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=6)
    a = p.parse_args()
    names = [
        d["dataset"]
        for d in json.loads((ROOT / "data/processed/manifest.json").read_text())
    ]
    specs = [
        (n, s, f) for n in names for s in [11, 22, 33, 44, 55] for f in ["RF", "ET"]
    ]
    records = []
    start = time.perf_counter()
    with ProcessPoolExecutor(a.jobs) as pool:
        for fut in as_completed([pool.submit(task, s) for s in specs]):
            r = fut.result()
            records.append(r)
            print(
                f"{r['dataset']} {r['family']} {r['seed']} pool={r['n_pool']} ref_acc={r['accuracy'][-1]:.4f} {r['seconds']:.1f}s",
                flush=True,
            )
    (ROOT / "results/training.json").write_text(
        json.dumps(
            {
                "wall_seconds": time.perf_counter() - start,
                "jobs": a.jobs,
                "runs": sorted(
                    records, key=lambda r: (r["dataset"], r["family"], r["seed"])
                ),
            },
            indent=2,
        )
    )

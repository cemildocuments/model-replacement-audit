from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def summarize():
    rows = []
    tasks = []
    for f in sorted((ROOT / "results/audits").glob("*.npz")):
        d = np.load(f)
        dataset, family, seed = f.stem.split("_")
        N = int(d["N"])
        U = int(d["U"])
        truth = d["truth"]
        B = d["bounds"]
        Q = d["queries"]
        D = d["D"]
        reps = B.shape[0]
        tasks.append(
            dict(
                dataset=dataset,
                family=family,
                seed=int(seed),
                N=N,
                U=U,
                union_fraction=U / N,
                disagreement_overlap=D.sum() / max(U, 1),
                safe_candidates=int((truth <= 0.01).sum()),
                zero_label_candidates=int((D / N <= 0.01).sum()),
            )
        )
        for e in [0, 0.005, 0.01, 0.02]:
            safe = truth <= e + 1e-12
            oracle = int(np.min(np.r_[d["trees"][safe], 128]))
            for m, method in enumerate(d["methods"]):
                anyfalse = ((B[:, m] <= e) & ~safe).any(axis=(1, 2)).mean()
                for b, budget in enumerate(d["budgets"]):
                    cert = B[:, m, b] <= e + 1e-12
                    sel = np.min(np.where(cert, d["trees"], 128), axis=1)
                    selected_delta = np.array(
                        [
                            truth[np.where(d["trees"] == v)[0][0]] if v < 128 else 0
                            for v in sel
                        ]
                    )
                    rows.append(
                        dict(
                            dataset=dataset,
                            family=family,
                            seed=int(seed),
                            method=str(method),
                            budget=int(budget),
                            epsilon=e,
                            N=N,
                            U=U,
                            queries=float(Q[:, m, b].mean()),
                            safe_rate=(
                                float(cert[:, safe].mean()) if safe.any() else np.nan
                            ),
                            unsafe_rate=(
                                float(cert[:, ~safe].mean())
                                if (~safe).any()
                                else np.nan
                            ),
                            certified_rate=float(cert.mean()),
                            replace_rate=float((sel < 128).mean()),
                            selected_trees=float(sel.mean()),
                            tree_reduction=float(1 - sel.mean() / 128),
                            selected_excess=float(selected_delta.mean()),
                            max_selected_excess=float(selected_delta.max()),
                            family_false=float(anyfalse),
                            oracle_trees=oracle,
                            bound_slack=float((B[:, m, b] - truth).mean()),
                            safe_count=int(safe.sum()),
                            unsafe_count=int((~safe).sum()),
                        )
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "results/audit_summary.csv", index=False)
    pd.DataFrame(tasks).to_csv(ROOT / "results/task_summary.csv", index=False)
    main = frame.query("epsilon==0.01")
    per = (
        main.groupby(["dataset", "method", "budget"])
        .mean(numeric_only=True)
        .reset_index()
    )
    per.to_csv(ROOT / "results/by_dataset.csv", index=False)
    agg = per.groupby(["method", "budget"]).mean(numeric_only=True).reset_index()
    agg.to_csv(ROOT / "results/aggregate.csv", index=False)
    # Dataset-level bootstrap, preserving paired settings and audit randomness within task summaries.
    a = per.query("method=='disagreement_exact'").pivot(
        index="dataset", columns="budget", values="safe_rate"
    )
    b = per.query("method=='uniform_conditional'").pivot(
        index="dataset", columns="budget", values="safe_rate"
    )
    dif = a - b
    rng = np.random.default_rng(20261027)
    boot = dif.to_numpy()[rng.integers(0, 6, (10000, 6))].mean(axis=1)
    ci = {
        str(col): {
            "mean": float(dif[col].mean()),
            "lo": float(np.quantile(boot[:, j], 0.025)),
            "hi": float(np.quantile(boot[:, j], 0.975)),
        }
        for j, col in enumerate(dif.columns)
    }
    (ROOT / "results/paired_contrasts.json").write_text(json.dumps(ci, indent=2))
    print(
        agg[
            [
                "method",
                "budget",
                "safe_rate",
                "replace_rate",
                "tree_reduction",
                "family_false",
                "queries",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print("CI", ci)
    print(
        "TASKS",
        pd.DataFrame(tasks)
        .groupby("dataset")
        .mean(numeric_only=True)
        .round(3)
        .to_string(),
    )


if __name__ == "__main__":
    summarize()

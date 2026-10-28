"""Recreate the recorded model-replacement experiments."""

from pathlib import Path
import argparse, subprocess, sys, os

ROOT = Path(__file__).resolve().parent
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["full", "cached"], default="cached")
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    env = os.environ.copy()
    for name in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "MKL_NUM_THREADS",
    ]:
        env[name] = "1"

    def run(script, *extra):
        print("Running", script, flush=True)
        subprocess.run(
            [sys.executable, str(ROOT / "src" / script), *extra],
            cwd=ROOT,
            env=env,
            check=True,
        )

    if args.mode == "full":
        run("prepare_data.py")
        run("train_models.py", "--jobs", str(args.jobs))
    run("run_audits.py", "--jobs", str(args.jobs), "--reps", "500")
    run("stress_test.py", "--jobs", str(args.jobs))
    run("summarize.py")
    run("verify_results.py")

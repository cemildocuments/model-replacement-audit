"""Compare numerical results with the recorded deterministic reference."""

from pathlib import Path
import hashlib, json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(path):
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            h = hashlib.sha256()
            for key in sorted(data.files):
                a = np.ascontiguousarray(data[key])
                h.update(key.encode())
                h.update(str(a.dtype).encode())
                h.update(str(a.shape).encode())
                h.update(a.tobytes())
            return h.hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    expected = json.loads((ROOT / "reference_checksums.json").read_text())
    bad = []
    for rel, value in expected.items():
        path = ROOT / rel
        if not path.exists() or fingerprint(path) != value:
            bad.append(rel)
    if bad:
        raise SystemExit("Numerical comparison differs for: " + ", ".join(bad))
    print(f"PASS: {len(expected)} numerical files match the recorded reference.")


if __name__ == "__main__":
    verify()

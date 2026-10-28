from pathlib import Path
import json, zipfile, io, hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def prepare():
    summaries = []
    for spec in json.loads((ROOT / "data/raw/manifest.json").read_text()):
        name = spec["name"]
        path = ROOT / "data/raw" / f"{name}.zip"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == spec["sha256"]
        with zipfile.ZipFile(path) as z:
            if name == "wine":
                frames = []
                for kind, tag in [("red", 0), ("white", 1)]:
                    f = pd.read_csv(z.open(f"winequality-{kind}.csv"), sep=";")
                    y = f.pop("quality")
                    f["wine_type"] = tag
                    f["label"] = y
                    frames.append(f)
                frame = pd.concat(frames, ignore_index=True)
            elif name == "pendigits":
                frame = pd.concat(
                    [
                        pd.read_csv(z.open(f"pendigits.{s}"), header=None)
                        for s in ["tra", "tes"]
                    ],
                    ignore_index=True,
                )
            else:
                fn = {
                    "banknote": "data_banknote_authentication.txt",
                    "spambase": "spambase.data",
                    "magic": "magic04.data",
                    "letter": "letter-recognition.data",
                }[name]
                frame = pd.read_csv(z.open(fn), header=None)
                if name == "letter":
                    frame = frame[list(frame.columns[1:]) + [0]]
            raw_n = len(frame)
            frame = frame.drop_duplicates().reset_index(drop=True)
            X = frame.iloc[:, :-1].to_numpy(dtype=np.float64)
            y, classes = pd.factorize(frame.iloc[:, -1], sort=True)
            _, groups = np.unique(X, axis=0, return_inverse=True)
            assert np.isfinite(X).all()
            out = ROOT / "data/processed" / f"{name}.npz"
            np.savez_compressed(out, X=X, y=y.astype("int16"), groups=groups)
            summaries.append(
                dict(
                    dataset=name,
                    raw_n=raw_n,
                    n=len(y),
                    features=X.shape[1],
                    classes=len(classes),
                    duplicates_removed=raw_n - len(y),
                    feature_groups=int(groups.max() + 1),
                    class_labels=list(map(str, classes)),
                    sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                )
            )
    (ROOT / "data/processed/manifest.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    prepare()

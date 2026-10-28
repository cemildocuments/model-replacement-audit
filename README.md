# Certifying Model Replacement with Fewer Labels

A reproducible benchmark and exact finite-population audit for deciding whether a smaller classifier can replace a frozen reference at a declared excess-error tolerance. The audit samples the union of prediction disagreements and shares each queried label across candidate comparisons.

## Reproduce the study

Use Python 3.12 and the pinned dependencies. Six parallel CPU workers are the default for model families, audits, and stress scenarios. The recorded experiment used 12 CPU cores and 18 GB RAM; the audit does not require a GPU. Lower `--jobs` to reduce simultaneous model fits.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --mode full --jobs 6
```

`full` verifies the six bundled source archives, constructs grouped datasets, refits 60 ensemble families, recreates 240 candidate predictions, repeats all audit and stress experiments, and writes numerical summaries. It then compares numerical outputs with the recorded reference. The recorded versions and ARM64 environment are described in `environment.json`. Model-training reproducibility can depend on numerical libraries and processor architecture; audit-only reproduction uses the frozen prediction benchmark to remove this source of variation.

```sh
python run.py --mode cached --jobs 6
python -m unittest discover -s tests -v
python src/verify_results.py
```

`cached` repeats audits, stress scenarios, and aggregation using the included predictions. Both modes overwrite numerical output files under `results/`. `src/verify_results.py` checks arrays and numerical summaries against `reference_checksums.json`; runtime measurements are excluded from numerical comparisons.

## Experimental design

Six UCI datasets are used: Banknote, Spambase, MAGIC, Pendigits, Letter, and Wine Quality. Exact feature-label duplicates are removed. Identical feature vectors, including conflicting labels, remain in the same split. Each split assigns 60% of feature groups to training and 40% to the finite evaluation pool. Seeds are 11, 22, 33, 44, and 55. Pendigits train/test source files are pooled before these splits. Red and white Wine files are combined with a wine-type indicator and retain their original quality classes.

RandomForestClassifier and ExtraTreesClassifier each use 128 trees, `max_features="sqrt"`, `min_samples_leaf=1`, and other scikit-learn defaults. Candidate models average the first 8, 16, 32, or 64 trees. Candidate construction never uses evaluation labels. Each family receives 500 random audit permutations, using seed `20261027 + repetition` (repetition 0 through 499). Full-pool labels supply offline ground truth; the audit calculations use only sampled labels.

Primary excess-error tolerance is 0.01, familywise error allowance is 0.05, and inspection budgets are 16, 32, 64, 128, and 256. Each of two count tails receives 0.05/(2 × 4 × 5). The same bound supports secondary tolerances 0, 0.005, and 0.02. A census is exact for the paired methods. All comparisons receive the deterministic zero-label disagreement cap. The guarantee concerns the frozen finite pool and predetermined inspections.

Methods are full-pool paired hypergeometric auditing, full-pool conditional auditing, shared disagreement auditing with exact hypergeometric inversion, a binomial ablation, and a harm-only ablation. The binomial method is an empirical comparison, rather than the subject of the finite-population guarantee.

`src/stress_test.py` constructs 34 feasible harm/benefit/zero count scenarios with 20,000 sequential audits per scenario. It compares corrected bounds with uncorrected repeated inspection and records unsafe approvals. Aggregates weight datasets equally; paired percentile intervals resample the six datasets 10,000 times with seed 20261027. The intervals describe variability across these benchmark tasks.

## Files and outputs

- `data/raw/`: six original UCI ZIP archives and their URL, license, and SHA-256 manifest.
- `data/processed/`: deduplicated feature arrays, class encodings, feature-group identifiers, and a manifest.
- `DATA_LICENSES.md`: source citations, links, license attribution, and transformation details.
- `src/prepare_data.py`: verifies archives and constructs the six processed datasets.
- `src/train_models.py`: trains reference ensembles and saves prefix candidate predictions and split row indices.
- `src/bounds.py`: exact count inversion and paired error bounds.
- `src/run_audits.py`: shared query simulation, comparison procedures, and candidate-level bounds.
- `src/stress_test.py`: controlled finite-population stress experiments.
- `src/summarize.py`: task, dataset, and aggregate numerical results plus paired bootstrap intervals.
- `src/verify_results.py`: deterministic numerical verification against recorded reference outputs.
- `tests/test_bounds.py`: exhaustive small-population coverage, monotonicity, zero-query, and census checks.
- `results/predictions/`: 60 frozen evaluation pools with labels, predictions, training/test row indices, and cumulative tree-node counts.
- `results/audits/`: 60 arrays containing bounds `[repetition, method, budget, candidate]`, query counts, signed true excess errors, disagreement sizes, and sampled counts. The count array axes are `[sampling scheme, repetition, budget, candidate, count type]`; schemes are full pool and disagreement union, and count types are disagreement, harmful, and beneficial.
- `results/audit_summary.csv`: all model-family, tolerance, budget, and method combinations.
- `results/task_summary.csv`, `by_dataset.csv`, and `aggregate.csv`: increasingly aggregated results.
- `results/paired_contrasts.json`: dataset-bootstrap intervals for eligible-candidate certification gains.
- `results/stress.csv`: all 34 controlled scenarios.
- `results/training.json` and `audit_manifest.json`: settings, task records, and runtime observations.
- `environment.json`, `requirements.txt`, `reference_checksums.json`, `CHECKSUMS.txt`, `LICENSE`, and `CITATION.cff`: runtime, verification, licensing, and citation metadata.

At budget 256, the recorded shared audit selects a replacement in approximately 83.4% of runs, compared with 49.8% for uniform conditional auditing, and reduces selected tree count by approximately 64.3%. Four of 30,000 shared-audit trajectories include an erroneous approval at the primary tolerance. These results measure fixed-pool certification and structural tree reduction.

## License and citation

Original software and documentation are released under the MIT License. UCI source data and derived data retain CC BY 4.0 attribution, detailed in `DATA_LICENSES.md`. Cite the study using `CITATION.cff` and retain the source-dataset attribution when redistributing data.

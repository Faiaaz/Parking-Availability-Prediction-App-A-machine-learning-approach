# Publication audit — 14 November 2026

| Dataset | Actual loaded file | Training | Validation | Calibration | Test |
|---|---|---|---|---|---|
| Birmingham | data/raw/dataset.csv | Yes | Yes | Yes | Yes |
| Melbourne | None | No | No | No | No |
| SFpark | None | No | No | No | No |

reliability.run fixes the Birmingham path and calls clean_data once. The helper reads that CSV. All partitions and corrupted copies descend from this frame. Imports do not load datasets. The previous completion log lists three horizons but no filenames. A fresh reproduction instrumenting pandas.read_csv recorded only data/raw/dataset.csv. Reproduced scores.csv and validation.csv match saved tables at absolute tolerance 1e-12 and relative tolerance 1e-10. Input, runner and target-builder SHA-256 values match hashes.txt. The input digest is f1e28b9c697769a1a05cd20148241b60b1a73fee4907611163efc6fd9d12770a.

| Horizon | Unique training | Augmented training | Validation | Calibration | Test per condition |
|---|---:|---:|---:|---:|---:|
| 30 | 23207 | 139242 | 2260 | 2894 | 2907 |
| 60 | 17703 | 106218 | 1750 | 2187 | 2244 |
| 120 | 15438 | 92628 | 1469 | 1684 | 1886 |

Counts were recomputed using the executable loader, target builder and partition function. Training combines clean inputs and five corrupted copies sharing measured targets. The 378 rows equal 3 horizons × 21 conditions × 6 methods. Only Birmingham supplied observations.

## Publication review

Initial tracked diff was empty. Reliability files were untracked. The review branch starts at eede06f. All nine tests passed and the full experiment reproduced both saved tables. Original metrics and experiment code were not altered during publication preparation.

A text scan of tracked/candidate files, including decompressed CSV predictions, found no matches for private keys, common cloud/GitHub/API tokens, credential assignments, email addresses or home-directory paths. This bounded scan does not guarantee absence of every secret format. PNG/PDF binaries were inventoried but excluded from text scanning. Git author identity remains in inherited history.

The private progress brief is excluded and explicitly ignored. Public documentation includes no private email transcript. Public parking identifiers and source attribution remain. The largest tracked artifact is research/results/main/predictions.csv.gz at 4,822,146 bytes. Existing figures, metrics and predictions are retained as research evidence. Models (approximately 263 MB), environment, caches, ZIP download and processed data are ignored. The prior failed-attempt audit is already tracked. New reliability result tables and hashes are intentionally included. Temporary reproduction output is excluded.

The README credits Birmingham under UCI CC BY 4.0 with council OGL provenance and Melbourne under CC BY 4.0. SFpark archive-specific reuse terms remain to be verified. No software license exists in the repository. The owner's choice of software license remains open.

## Staged scope

README corrects dataset usage, legacy claims, negative gate results and limitations. .gitignore excludes private correspondence and reproduction directories. Public research documents explain audit, fields, feasibility, questions, methodology, protocol and limitations. Code and tests implement the recorded experiment. Results include original metrics/hashes and recomputed partition counts. SFpark requirements are in sfpark_external_validation.md. Approval is requested on the exact staged diff before any commit or push.

# Task brief: pre-publication fixes for Point-Cloud-Reconstruction

## Context

This repo accompanies a published paper (Harb et al., ISPRS Archives, 2026) on
reconstructing CAD command sequences from point clouds. The associated dataset is about
to be uploaded to a public data repository with a DOI.

An audit found a correctness bug in the dataset generator and several issues that prevent
the published code from regenerating the published data. The **data itself has already
been corrected** by separate scripts — your job is the repository code, so that the code
and the released data agree.

### Pipeline, for orientation

```
cad_json ──json2pc.py──> pc_cad                                       (baseline / MEM)

cad_json ──DeepCAD──> cad_vec ──code/create_data.py──> pc_from_vec
                                                       pc_from_vec_labels
                                  └──code/create_extr_data.py──> pc_extrusion
                                                                 pc_extrusion_labels  (SEM)
```

- `pc_from_vec/<bucket>/<id>.ply` — 10,000 points per model
- `pc_from_vec_labels/<bucket>/<id>.h5` — `labels` (10000,) per-point extrusion id;
  `sequences/<k>` (60,17) one padded subsequence per extrusion
- `pc_extrusion/<bucket>/<id>/<id>_<j>.ply` — that cloud partitioned by label
- `pc_extrusion_labels/<bucket>/<id>/<id>_<j>.h5` — `extrusion_id` scalar; `sequence` (N,17)
  truncated at first EOS

### Hard constraints

- **Do not modify anything under `data/`.** The data is corrected already and a
  re-run would overwrite it. Treat `data/` as read-only.
- Do not reformat files, reorder imports, or refactor beyond what each task asks.
  Diffs should be reviewable line by line.
- Do not change model architecture, hyperparameters, or anything that would alter
  published results.
- Python 3.9 compatible (the conda env in `environment.yml` is 3.9).

---

## Task 1 — Fix the extrusion pairing bug (P0)

**File:** `code/create_extr_data.py`

`split_pc_by_labels` groups points by `np.unique(labels)`, which returns only the labels
actually present. The caller then uses the *enumeration index* `j` to look up the CAD
subsequence:

```python
pcs = split_pc_by_labels(pc, label)
for j, pc in enumerate(pcs):
    save_pc(pc, ..._{j}.ply)
    sequence = sequences[str(j)]     # BUG: j is a position, not an extrusion id
    save_h5(j, sequence, ..._{j}.h5)
```

When an extrusion loses all its points to the proximity filter in `create_data.py`, its
label is absent from `labels`, `np.unique` returns a sequence with a gap, and every
surviving extrusion after the gap is paired with its neighbour's subsequence. Measured
impact on the released data: 1,137 of 362,225 samples across 425 models.

**Change:** make `split_pc_by_labels` return the label values alongside the point groups,
and use those values for both the sequence lookup and the stored `extrusion_id`.

```python
def split_pc_by_labels(pc, labels):
    present = np.unique(labels)
    class_pcs = [pc[labels == class_id] for class_id in present]
    return class_pcs, present.tolist()
```

```python
pcs, present = split_pc_by_labels(pc, label)
for j, pc in enumerate(pcs):
    extr_id = present[j]                    # true extrusion id
    ...
    save_pc(pc, ..._{j}.ply)
    save_h5(extr_id, sequences[str(extr_id)], ..._{j}.h5)
```

**Keep the filename index as `j`** so paths stay contiguous (`_0`, `_1`, `_2`, …) and match
the already-published data. The authoritative id is the `extrusion_id` dataset inside the
`.h5`, which may now skip values.

**Acceptance:** for a model whose labels are `[0, 1, 3, 4, 6]`, the five written `.h5`
files must carry `extrusion_id` values `0, 1, 3, 4, 6` and the matching subsequences.

---

## Task 2 — Make N_POINTS correct and explicit (P0)

**File:** `code/dataset.py` line 17, plus call sites

`N_POINTS = 2048`, but the released data was generated with `N_POINTS = 10000`.
`create_extr_data.py` asserts `pc.shape[0] == 10000`, so running the generator as
committed raises on every sample — and because of Task 3 the exception is swallowed,
so it fails silently and produces an empty output directory.

**Change:** set the module default to `10000`, and make the generation scripts state the
value they need explicitly rather than inheriting a shared global that training code may
legitimately want to change. A module-level constant with an override argument on the
dataset classes is fine; a config object is fine. Do not silently couple the training
sample size to the generation sample size.

**Acceptance:** `python code/create_extr_data.py --data-root <tmp>` on a small fixture
completes without assertion errors and writes the expected files.

---

## Task 3 — Stop swallowing exceptions in the generators (P1)

**Files:** `code/create_data.py`, `code/create_extr_data.py`

Both wrap their per-sample work in a bare `except Exception as e` that records the error
into a dict. `create_extr_data.py` then only prints at the end; `create_data.py` pickles
the dict. This is how a systematic failure looked like a successful run for a year.

**Change:**

- Print a running count of failures during the loop, not only at the end.
- At the end, print a clear summary: `N of M samples failed`, plus the top few distinct
  exception types with counts.
- Exit non-zero if the failure rate exceeds a threshold (suggest 1%) — a systematic
  failure should not exit 0.
- Keep collecting into the dict, but store `repr(e)` rather than the exception object so
  the pickle is loadable without the original traceback context.

Do not remove the broad catch — some samples legitimately fail on malformed CAD. The goal
is that failure is loud, not that it is fatal.

---

## Task 4 — Remove hardcoded relative data paths (P1)

**Files:** `code/create_data.py`, `code/create_extr_data.py`

Both hardcode `DATA_DIR = "../data"` and must be run from inside `code/`.
`create_data.py::check_sequence` additionally builds `os.path.join("..", "data",
"cad_json", ...)` inline — the author left a `### REFACTOR` marker on that line.

**Change:** add `--data-root` to both scripts (default `data`, resolved relative to the
current working directory, matching the existing convention in `pc2cad.py` and
`code/train.py`). Thread it through to every path construction, including
`check_sequence`. Both scripts should then run from the repo root.

**Acceptance:** `python code/create_data.py --data-root data` and
`python code/create_extr_data.py --data-root data` both resolve paths correctly when run
from the repository root.

---

## Task 5 — Add the validation scripts (P1)

Two scripts exist and should be committed to the repo root (they will be provided
separately — do not write them from scratch):

- `check_extrusion_pairing.py` — characterises the source data. Reports models with
  dropped extrusions and how many samples the enumeration-index bug would affect.
  Its numbers are a property of `pc_from_vec_labels` and are **identical before and
  after any fix** — it is not a post-fix verifier.
- `verify_extrusion_pairing.py` — reads `pc_extrusion_labels` itself and checks every
  cloud is paired with the correct subsequence. Exits non-zero on any mismatch.

**Change:** commit both, add a short "Validating the dataset" section to `README.md`
showing how to run them and what output to expect, and note the distinction above
explicitly so nobody mistakes the first for the second.

---

## Task 6 — Make the point subsample reproducible (P2)

**File:** `code/dataset.py`

`PCExtrusionSegmentationDataset.__getitem__` and `PointCloudEmbeddingDataset.__getitem__`
call `random.sample(...)` on the global RNG with no seed. During generation this
permuted point order in the output files; during training it is a legitimate
augmentation.

**Change:** accept an optional `seed` / `rng` on the dataset classes. Default behaviour
must stay unchanged (unseeded) so training is unaffected; the generation scripts should
pass a fixed seed so regeneration is deterministic.

---

## Task 7 — README (P2)

**File:** `README.md`

The current README documents the original thesis pipeline and predates the extrusion
segmentation work entirely. Add:

- The segmentation strategy: what `create_data.py` and `create_extr_data.py` produce,
  in what order, and the schema of both `.h5` formats
- A `Known issues and corrections` section (content will be provided separately)
- The `Validating the dataset` section from Task 5
- A pointer to where the dataset is hosted, once the DOI exists (leave a `TODO(DOI)`)

Do not rewrite the existing DeepCAD / PointNet++ setup sections.

---

## Optional cleanups (P3 — only if the above is done and reviewed)

- `code/dataset.py` carries a ~1,000-element `IGNORE_INDICES` list commented out inline.
  Delete it or move it to a data file.
- `code/create_extr_data.py` ends with a bare `# TODO README`.
- `.gitignore` excludes `**/*.json`, which also excludes `pyrightconfig.json` and any
  future config. Narrow it to the data paths that actually need excluding.

---

## Verification before you report done

1. `python -m pyflakes code/` or equivalent — no new warnings introduced
2. `pytest tests/` passes (there is one test file, `tests/test_dataloader.py`)
3. Build a small fixture: copy 3–4 models' worth of `pc_from_vec` + `pc_from_vec_labels`
   into a temp tree, including at least one model whose labels have a gap
   (e.g. `[0, 1, 3]`), run `create_extr_data.py --data-root <tmp>`, then run
   `verify_extrusion_pairing.py --data-root <tmp>` and confirm it exits 0
4. Confirm `git status` shows no modifications under `data/`
5. Summarise the diff per file and flag anything you changed that this brief did not ask for
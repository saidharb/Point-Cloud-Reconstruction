#!/usr/bin/env python3
"""
check_extrusion_pairing.py

Verifies that every point cloud in data/pc_extrusion/ is paired with the CAD
subsequence of the extrusion it was actually sampled from.

WHY THIS EXISTS
---------------
code/create_extr_data.py splits the full labelled cloud like this:

    pcs = split_pc_by_labels(pc, label)      # grouped by np.unique(labels), ascending
    for j, pc in enumerate(pcs):
        save_pc(pc,   ..._{j}.ply)
        save_h5(j, sequences[str(j)], ..._{j}.h5)

`j` is the *enumeration index*, not the extrusion id. During generation
(create_data.py) an extrusion can lose all of its points to the radius filter,
in which case its label never appears in `labels`. When the lost extrusion is
not the last one, every subsequent point cloud is shifted down by one and gets
paired with the WRONG CAD subsequence.

    labels present = [0, 1, 3, 4, 6],  sequences = 0..6
      _0.ply -> extrusion 0 -> sequences['0']   OK
      _1.ply -> extrusion 1 -> sequences['1']   OK
      _2.ply -> extrusion 3 -> sequences['2']   WRONG
      _3.ply -> extrusion 4 -> sequences['3']   WRONG
      _4.ply -> extrusion 6 -> sequences['4']   WRONG

The correct sequence for the j-th written cloud is sequences[str(u[j])] where
u = sorted(np.unique(labels)).

USAGE
-----
    python check_extrusion_pairing.py --data-root data                 # report only
    python check_extrusion_pairing.py --data-root data --report r.csv  # + CSV
    python check_extrusion_pairing.py --data-root data --fix           # rewrite bad h5

--fix only rewrites data/pc_extrusion_labels/<bucket>/<id>/<id>_<j>.h5 for the
affected models, setting `extrusion_id` and `sequence` to the correct extrusion.
The .ply point clouds are already correct and are never touched.
"""

import argparse
import csv
import os
import sys
from glob import glob

import h5py
import numpy as np


def truncate_at_eos(seq):
    """Keep rows up to and including the first EOS (command id 3), as create_extr_data.py did."""
    eos = np.where(seq[:, 0] == 3)[0]
    return seq[: eos[0] + 1, :] if len(eos) else seq


def scan(data_root):
    """Yield one dict per model that has a label/sequence count mismatch."""
    label_root = os.path.join(data_root, "pc_from_vec_labels")
    files = sorted(glob(os.path.join(label_root, "*", "*.h5")))
    if not files:
        sys.exit(f"No label files found under {label_root}")

    print(f"Scanning {len(files):,} models under {label_root} ...")
    findings = []
    for n, path in enumerate(files, 1):
        if n % 5000 == 0:
            print(f"  {n:,}/{len(files):,}", end="\r", flush=True)
        model_id = os.path.splitext(os.path.basename(path))[0]
        try:
            with h5py.File(path, "r") as f:
                labels = f["labels"][:]
                n_seq = len(f["sequences"])
        except Exception as exc:                       # noqa: BLE001
            findings.append({"id": model_id, "kind": "unreadable", "detail": repr(exc)})
            continue

        present = np.unique(labels).tolist()
        if present == list(range(n_seq)):
            continue                                    # clean

        # An extrusion produced no surviving points.
        mispaired = [j for j, lab in enumerate(present) if j != lab]
        findings.append(
            {
                "id": model_id,
                "kind": "mispaired" if mispaired else "tail_drop",
                "n_extrusions": n_seq,
                "labels_present": present,
                "n_clouds": len(present),
                "bad_indices": mispaired,
            }
        )
    print(f"  {len(files):,}/{len(files):,}   done.")
    return findings, len(files)


def fix(data_root, findings):
    """Rewrite the per-extrusion label h5 files for mispaired models."""
    fixed = 0
    for f in findings:
        if f.get("kind") != "mispaired":
            continue
        model_id = f["id"]
        bucket = model_id[:4]
        src = os.path.join(data_root, "pc_from_vec_labels", bucket, model_id + ".h5")
        with h5py.File(src, "r") as fp:
            sequences = {k: v[:] for k, v in fp["sequences"].items()}

        for j in f["bad_indices"]:
            true_label = f["labels_present"][j]
            dst = os.path.join(
                data_root, "pc_extrusion_labels", bucket, model_id, f"{model_id}_{j}.h5"
            )
            if not os.path.exists(dst):
                print(f"  ! missing {dst}, skipped")
                continue
            seq = truncate_at_eos(sequences[str(true_label)])
            with h5py.File(dst, "w") as out:
                out.create_dataset("extrusion_id", data=true_label)
                out.create_dataset("sequence", data=seq)
            fixed += 1
    return fixed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data", help="path to the data/ directory")
    ap.add_argument("--report", help="write a CSV of all findings here")
    ap.add_argument("--fix", action="store_true",
                    help="rewrite the mispaired pc_extrusion_labels h5 files IN PLACE")
    args = ap.parse_args()

    findings, n_models = scan(args.data_root)

    mispaired = [f for f in findings if f.get("kind") == "mispaired"]
    tail = [f for f in findings if f.get("kind") == "tail_drop"]
    unreadable = [f for f in findings if f.get("kind") == "unreadable"]
    n_bad_samples = sum(len(f["bad_indices"]) for f in mispaired)
    n_dropped = sum(f["n_extrusions"] - f["n_clouds"] for f in mispaired + tail)

    print()
    print("=" * 62)
    print(f"models scanned                     {n_models:>10,}")
    print(f"models with a dropped extrusion    {len(mispaired) + len(tail):>10,}")
    print(f"  - dropped at the tail (harmless) {len(tail):>10,}")
    print(f"  - dropped mid-sequence (BAD)     {len(mispaired):>10,}")
    print(f"extrusions with no surviving points{n_dropped:>10,}")
    print(f"MISPAIRED point cloud / sequence   {n_bad_samples:>10,}")
    if unreadable:
        print(f"unreadable label files             {len(unreadable):>10,}")
    print("=" * 62)

    if args.report:
        with open(args.report, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "kind", "n_extrusions", "n_clouds",
                        "labels_present", "mispaired_indices"])
            for f in findings:
                w.writerow([f["id"], f["kind"], f.get("n_extrusions", ""),
                            f.get("n_clouds", ""), f.get("labels_present", ""),
                            f.get("bad_indices", "")])
        print(f"report written to {args.report}")

    if args.fix:
        if not mispaired:
            print("nothing to fix.")
            return
        print(f"\nrewriting {n_bad_samples:,} label files ...")
        print(f"done, {fix(args.data_root, findings):,} files rewritten.")
    elif mispaired:
        print("\nrun again with --fix to correct the pairing "
              "(only the label .h5 files change; the .ply clouds are already correct)")


if __name__ == "__main__":
    main()
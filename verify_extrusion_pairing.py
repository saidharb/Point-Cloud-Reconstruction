#!/usr/bin/env python3
"""
verify_extrusion_pairing.py

Post-fix verifier. Reads data/pc_extrusion_labels/ and checks that every written
point cloud is paired with the CAD subsequence of the extrusion it was actually
sampled from. Exits non-zero if any pairing is wrong.

HOW THIS DIFFERS FROM check_extrusion_pairing.py
------------------------------------------------
check_extrusion_pairing.py characterises the *source* data: it reads
pc_from_vec_labels/ and reports which models lost an extrusion to the proximity
filter in create_data.py, and how many samples the enumeration-index bug would
therefore affect. Those numbers are a property of pc_from_vec_labels and are
identical before and after any fix -- it is not a verifier.

This script reads the *generated* data in pc_extrusion_labels/ and compares what
is on disk against what should be there. Its numbers do change when the bug is
fixed, and it is the one to run to confirm a regeneration is correct.

WHAT IS CHECKED
---------------
For each model, `present = sorted(np.unique(labels))` from pc_from_vec_labels
gives the extrusions that survived, in the order create_extr_data.py writes them.
The j-th written file <id>_<j>.h5 must therefore carry:

    extrusion_id == present[j]
    sequence     == sequences[str(present[j])] truncated at the first EOS

With --check-clouds, the point count of <id>_<j>.ply is additionally compared
against the size of label group present[j], which confirms the cloud and the
sequence describe the same extrusion rather than only re-deriving the h5.

USAGE
-----
    python verify_extrusion_pairing.py --data-root data
    python verify_extrusion_pairing.py --data-root data --check-clouds
    python verify_extrusion_pairing.py --data-root data --report mismatches.csv
"""

import argparse
import csv
import os
import sys
from glob import glob

import h5py
import numpy as np


def truncate_at_eos(seq):
    """Keep rows up to and including the first EOS (command id 3), as create_extr_data.py does."""
    eos = np.where(seq[:, 0] == 3)[0]
    return seq[: eos[0] + 1, :] if len(eos) else seq


def verify(data_root, check_clouds=False):
    """Return (mismatches, n_models, n_samples)."""
    src_root = os.path.join(data_root, "pc_from_vec_labels")
    out_root = os.path.join(data_root, "pc_extrusion_labels")
    model_dirs = sorted(glob(os.path.join(out_root, "*", "*")))
    model_dirs = [d for d in model_dirs if os.path.isdir(d)]
    if not model_dirs:
        sys.exit(f"No generated models found under {out_root}")

    print(f"Verifying {len(model_dirs):,} models under {out_root} ...")
    mismatches = []
    n_samples = 0

    for n, model_dir in enumerate(model_dirs, 1):
        if n % 5000 == 0:
            print(f"  {n:,}/{len(model_dirs):,}", end="\r", flush=True)
        model_id = os.path.basename(model_dir)
        bucket = model_id[:4]
        src = os.path.join(src_root, bucket, model_id + ".h5")

        try:
            with h5py.File(src, "r") as fp:
                labels = fp["labels"][:]
                sequences = {k: v[:] for k, v in fp["sequences"].items()}
        except Exception as exc:                       # noqa: BLE001
            mismatches.append({"id": model_id, "j": "", "reason": "source unreadable",
                               "expected": "", "found": repr(exc)})
            continue

        present = np.unique(labels).tolist()
        written = sorted(glob(os.path.join(model_dir, model_id + "_*.h5")),
                         key=lambda p: int(os.path.splitext(p)[0].rsplit("_", 1)[1]))

        if len(written) != len(present):
            mismatches.append({"id": model_id, "j": "", "reason": "wrong number of files",
                               "expected": len(present), "found": len(written)})
            continue

        for j, path in enumerate(written):
            n_samples += 1
            true_label = present[j]
            expected_seq = truncate_at_eos(sequences[str(true_label)])

            try:
                with h5py.File(path, "r") as fp:
                    extr_id = fp["extrusion_id"][()]
                    seq = fp["sequence"][:]
            except Exception as exc:                   # noqa: BLE001
                mismatches.append({"id": model_id, "j": j, "reason": "unreadable",
                                   "expected": "", "found": repr(exc)})
                continue

            if int(extr_id) != int(true_label):
                mismatches.append({"id": model_id, "j": j, "reason": "extrusion_id",
                                   "expected": int(true_label), "found": int(extr_id)})
                continue

            if seq.shape != expected_seq.shape or not np.array_equal(seq, expected_seq):
                mismatches.append({"id": model_id, "j": j, "reason": "sequence",
                                   "expected": f"sequences['{true_label}'] {expected_seq.shape}",
                                   "found": str(seq.shape)})
                continue

            if check_clouds:
                ply = os.path.join(data_root, "pc_extrusion", bucket, model_id,
                                   f"{model_id}_{j}.ply")
                if not os.path.exists(ply):
                    mismatches.append({"id": model_id, "j": j, "reason": "missing ply",
                                       "expected": ply, "found": ""})
                    continue
                import open3d as o3d
                n_pts = len(o3d.io.read_point_cloud(ply).points)
                n_expected = int((labels == true_label).sum())
                if n_pts != n_expected:
                    mismatches.append({"id": model_id, "j": j, "reason": "cloud size",
                                       "expected": n_expected, "found": n_pts})

    print(f"  {len(model_dirs):,}/{len(model_dirs):,}   done.")
    return mismatches, len(model_dirs), n_samples


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data", help="path to the data/ directory")
    ap.add_argument("--check-clouds", action="store_true",
                    help="also compare the .ply point counts against the label groups (slow)")
    ap.add_argument("--report", help="write a CSV of all mismatches here")
    args = ap.parse_args()

    mismatches, n_models, n_samples = verify(args.data_root, check_clouds=args.check_clouds)

    print()
    print("=" * 62)
    print(f"models verified                    {n_models:>10,}")
    print(f"samples verified                   {n_samples:>10,}")
    print(f"MISMATCHED point cloud / sequence  {len(mismatches):>10,}")
    print("=" * 62)

    if args.report:
        with open(args.report, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "j", "reason", "expected", "found"])
            for m in mismatches:
                w.writerow([m["id"], m["j"], m["reason"], m["expected"], m["found"]])
        print(f"report written to {args.report}")

    if mismatches:
        for m in mismatches[:20]:
            print(f"  {m['id']}_{m['j']}: {m['reason']} "
                  f"(expected {m['expected']}, found {m['found']})")
        if len(mismatches) > 20:
            print(f"  ... and {len(mismatches) - 20:,} more")
        sys.exit(1)

    print("all point clouds are paired with the correct CAD subsequence.")


if __name__ == "__main__":
    main()

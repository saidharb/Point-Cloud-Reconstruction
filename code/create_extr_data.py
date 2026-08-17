import argparse
import os
import sys
from collections import Counter

import h5py
import numpy as np

import open3d as o3d

from dataset import PCExtrusionSegmentationDataset, N_POINTS

# The clouds in pc_from_vec hold N_POINTS points and this script needs all of them,
# so the sample size is stated here instead of inheriting whatever the training
# default happens to be.
N_POINTS_SAMPLE = N_POINTS

# Fixed seed for the point subsample, so that regeneration is deterministic.
SEED = 0

# Exit non-zero if more than this fraction of the samples fail.
MAX_FAILURE_RATE = 0.01


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser(
        'Split labelled point clouds into one cloud + CAD subsequence per extrusion'
    )
    parser.add_argument('--data-root', type=str, default='data',
                        help='data directory, relative to the current working directory')

    return parser.parse_args()


def save_pc(pc, path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.io.write_point_cloud(path, pcd)

def save_h5(extr_id, sequence, path):
    with h5py.File(path, "w") as f:
        f.create_dataset("extrusion_id", data=extr_id)
        f.create_dataset("sequence", data=sequence)

def split_pc_by_labels(pc: np.ndarray, labels: np.ndarray):
    """Group the points by label.

    Returns the point groups and the label values they belong to. The labels are
    the extrusion ids and are *not* necessarily 0..n-1: an extrusion that lost all
    of its points to the proximity filter in create_data.py leaves a gap. Callers
    must use the returned values, not the enumeration index, to look up the CAD
    subsequence of a group.
    """
    present = np.unique(labels)
    class_pcs = []
    for class_id in present:
        class_mask = (labels == class_id)
        class_pc = pc[class_mask]
        class_pcs.append(class_pc)
    return class_pcs, present.tolist()

def main(args):

    DATA_DIR = args.data_root

    train_dataset = PCExtrusionSegmentationDataset(DATA_DIR, 'train', use_normals=False, verbose=False,
                                                   n_points=N_POINTS_SAMPLE, seed=SEED)
    val_dataset = PCExtrusionSegmentationDataset(DATA_DIR, 'validation', use_normals=False, verbose=False,
                                                 n_points=N_POINTS_SAMPLE, seed=SEED)
    test_dataset = PCExtrusionSegmentationDataset(DATA_DIR, 'test', use_normals=False, verbose=False,
                                                  n_points=N_POINTS_SAMPLE, seed=SEED)
    datasets = [train_dataset, val_dataset, test_dataset]

    error_dict = {}
    error_types = Counter()
    n_total = 0

    for dataset in datasets:
        length = len(dataset)

        for i in range(length):
            n_total += 1
            print(f"\r{dataset.split} {i}/{length} | {len(error_dict)} failed", end="")

            try:
                data = dataset[i]
                id = data['id']
                pc = data['pc']
                label = data['label']
                assert pc.shape[0] == N_POINTS_SAMPLE, "Point cloud from pc_from_vec has {} points, but got {}".format(N_POINTS_SAMPLE, pc.shape[0])

                pcs, present = split_pc_by_labels(pc, label)

                sequences = {}
                h5_path = os.path.join(DATA_DIR, "pc_from_vec_labels", id[:4], id + ".h5")
                with h5py.File(h5_path, 'r') as f:
                    for class_id, sequence in f['sequences'].items():
                        sequence = sequence[:]
                        seq_length = np.where(sequence[:, 0] == 3)[0][0] + 1 # only save up to including the first EOS command
                        sequence = sequence[:seq_length, :]
                        sequences[class_id] = sequence

                for j, pc in enumerate(pcs):
                    # j keeps the filenames contiguous (_0, _1, _2, ...); extr_id is the
                    # true extrusion this cloud was sampled from and may skip values.
                    extr_id = present[j]

                    pc_path = os.path.join(os.path.abspath(DATA_DIR), "pc_extrusion", id[:4], id, id + "_" + str(j) + ".ply")
                    os.makedirs(os.path.dirname(pc_path), exist_ok=True)
                    save_pc(pc, pc_path)

                    h5_path = os.path.join(os.path.abspath(DATA_DIR), "pc_extrusion_labels", id[:4], id, id + "_" + str(j) + ".h5")
                    os.makedirs(os.path.dirname(h5_path), exist_ok=True)
                    sequence = sequences[str(extr_id)]
                    save_h5(extr_id, sequence, h5_path)

            except Exception as e:
                error_dict[f"{dataset.split}/{i}"] = repr(e)
                error_types[type(e).__name__] += 1
        print()

    n_failed = len(error_dict)
    print(f"{n_failed} of {n_total} samples failed")
    for name, count in error_types.most_common(5):
        print(f"  {count:>8,}  {name}")

    failure_rate = n_failed / n_total if n_total else 0.0
    if failure_rate > MAX_FAILURE_RATE:
        print(f"failure rate {failure_rate:.2%} exceeds the {MAX_FAILURE_RATE:.0%} threshold "
              f"-- this looks systematic, not like malformed CAD", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main(parse_args())

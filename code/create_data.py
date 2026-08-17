import os
import shutil
import h5py
import numpy as np
import sys
import argparse
import pandas as pd
import json
import pickle
import matplotlib.pyplot as plt

from dataset import PointCloudEmbeddingSequenceDataset, N_POINTS_TRAIN
from models.DeepCAD.cadlib.visualize import vec2CADsolid
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Extend.DataExchange import write_step_file
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from models.DeepCAD.cadlib.extrude import CADSequence
from models.DeepCAD.cadlib.visualize import create_CAD
from models.DeepCAD.cadlib.visualize import CADsolid2pc
from models.DeepCAD.utils.pc_utils import write_ply
import open3d as o3d
from scipy.spatial import cKDTree
from collections import Counter

# This script only reads the CAD sequence and the id from the dataset, never the
# sampled point cloud, but the dataset samples one anyway. pc_cad holds 8096 points,
# so the generation size N_POINTS (10000) is not a legal sample size here; state the
# training size explicitly rather than inheriting whatever the default happens to be.
N_POINTS_SAMPLE = N_POINTS_TRAIN

# Fixed seed for the point subsample, so that regeneration is deterministic.
SEED = 0

# Exit non-zero if more than this fraction of the samples fail.
MAX_FAILURE_RATE = 0.01

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser(
        'Create point clouds with extrusion labels from CAD seqeunces'
    )
    parser.add_argument('--data-root', type=str, default='data',
                        help='data directory, relative to the current working directory')

    return parser.parse_args()


def pad_seq(seq):
    """Takes custom sequence (N,17) and pads it to (60,17)"""
    eos_row = [3] + 16 * [-1]
    seq_np = np.array(seq, dtype=np.float32)
    num_pad_rows = 60 - seq_np.shape[0]
    pad_array = np.tile(eos_row, (num_pad_rows, 1))
    seq_np_pad = np.concatenate([seq_np, pad_array], axis=0)  
    return seq_np_pad

def seq2shape(seq):
    cad_seq = CADSequence.from_vector(seq, is_numerical=True)
    shape = create_CAD(cad_seq)
    return shape

def seq2pc(sequence, nr_points=8096):
    """ Input:  sequence:  Sequence as np (60,17)
                nr_points: Number of points to sample for the sequence as int
        Output: out_pc:    Output point cloud as np (N, 3)
    """
    
    shape = seq2shape(sequence)
    out_pc = CADsolid2pc(shape, n_points=nr_points)
    return out_pc

def get_gt_pc(sequence, n_points=20000):
    """ Input:  sequence: Sequence as np (60,17)
                n_points: Number of points which are sampled per extrusion in the sequence as int
        Output: Point cloud of the sequence as np (N, 3)
    """
    # Determine number of extrusions in sequence
    num_ext = 0
    for command in sequence:
        if command[0] == 5:
            num_ext += 1
    pc = seq2pc(sequence, nr_points=num_ext*n_points)
    return pc

def split_and_pad_sequence_by_extrusion(matrix, delimiter=5):
    """Takes (60,17) sequence and splits it by the extrusions and pads it and returns a (60,17) for each extrusion""" 
    matrix = np.array(matrix)
    assert matrix.shape == (60, 17), "Input must be (60, 17)"

    commands = matrix[:,0]
    split_indices = []
    start_idx = 0

    for idx, val in enumerate(commands):
        if val == delimiter:
            split_indices.append((start_idx, idx))
            start_idx = idx + 1

    output = []
    for start, end in split_indices:
        length = 59 - (end - start)
        pad_row = [[3] + 16 * [-1]] * length
        new_matrix = matrix[start:end+1]
       
        pad_matrix = np.vstack([new_matrix, pad_row])
        output.append(pad_matrix)
    return output

def get_labled_pc_per_ext(sequence, nr_points=8096):
    """ Input:  sequence: Sequence as np (60,17)
                nr_points: Number of points which are sampled per extrusion as int
        Output: merged_pc:            Point cloud where each point is labled by the extrusion that created it as np (N, 3)
                merged_labels:        Labels for each point as np (N)
                extrusion_splits_seq: Extrusion sequence per label as list of np (60,17)
    """
    
    extrusion_splits_seq = split_and_pad_sequence_by_extrusion(sequence)
    point_clouds = []
    for i, ext_seq in enumerate(extrusion_splits_seq):
        pc = seq2pc(ext_seq, nr_points=nr_points)
        point_clouds.append(pc)
    merged_pc = np.concatenate(point_clouds, axis=0)
        
    labels = []
    for i, pc in enumerate(point_clouds):
        labels.append(np.full((pc.shape[0],), i, dtype=int))
    merged_labels = np.concatenate(labels, axis=0)
    
    return merged_pc, merged_labels, extrusion_splits_seq

def filter_by_nearest_neighbor(pc_A, pc_B, epsilon=0.5):
    """
    Filters point cloud B using nearest neighbors from point cloud A.
    Keeps only points in B that are within `epsilon` distance to any point in A.
    
    :param pc_A: Nx3 point cloud (gt_pc)
    :param pc_B: Mx3 point cloud (pc with labled extrusions)
    
    :return: pc_B filtered point cloud, mask (M,), where mask[i] = True if point i is kept
    """
    tree = cKDTree(pc_A)
    distances, _ = tree.query(pc_B, k=1)
    mask = distances < epsilon
    return pc_B[mask], mask

def filter_pc_and_labels(gt_pc, ext_pc, ext_labels, epsilon=0.01):
    """
    Input: gt_pc: Point cloud sampled from the original shape as np (N, 3)
           ext_pc: Point cloud where points are sampled from all extrusions as np (M, 3)
           ext_labels: Extrusion label for each point in ext_pc
    Output: filtered_pc: ext_pc where only points are retained that are within a distance of epsilon to a point in gt_pc
            filtered_labels: labels for each point in filtered_pc
    """
    filtered_pc, mask = filter_by_nearest_neighbor(gt_pc, ext_pc, epsilon=epsilon)
    filtered_labels = ext_labels[mask]
    return filtered_pc, filtered_labels

def save_pc(pc, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.io.write_point_cloud(path, pcd)

def save_labels_with_ext(labels, ext_list, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, 'w') as f:
        f.create_dataset("labels", data=labels, compression='gzip')
    
        grp = f.create_group("sequences")
        for i, extr in enumerate(ext_list):
            grp.create_dataset(str(i), data=extr, compression='gzip')

def get_extrusion_distance_per_extrusion(data):
    distances_per_extrusion = []
    for item in data["sequence"]:
        if item["type"] == "ExtrudeFeature":
            
            extrude_id = item["entity"]
            extrude_entity = data["entities"][extrude_id]
            num_profiles = len(extrude_entity["profiles"])
            
            extent_one = extrude_entity["extent_one"]["distance"]["value"]
            extent_two = extrude_entity["extent_two"]["distance"]["value"]

            extrusion_distances = (extent_one, extent_two)

            for i in range(num_profiles):
                distances_per_extrusion.append(extrusion_distances)
    return distances_per_extrusion

def check_sequence(sequence, id, data_root):
    """
    Checks if the extrude distances are not zero, otherwise error.
    """
    json_path = os.path.join(data_root, "cad_json", id[:4], id + ".json")
    flag = False

    with open(json_path, "r") as file:
        data = json.load(file)
    distances = get_extrusion_distance_per_extrusion(data)

    ext_counter = 0
    for i, command in enumerate(sequence):
        if command[0] == 5: # only extrusion
            if command[12] == 0: # scale is numericalized at 0
                sequence[i][12] += 1
                flag = True
            if (command[16] == 0 or command[16] == 1) and command[13] == 128: # one-sided or symmetric
                if distances[ext_counter][0] > 0:
                    sequence[i][13] += 1
                    flag = True
                else:
                    sequence[i][13] -= 1
                    flag = True
            if command[16] == 2: # two-sided
                if command[13] == 128:
                    if distances[ext_counter][0] > 0:
                        sequence[i][13] += 1
                        flag = True
                    else:
                        sequence[i][13] -= 1
                        flag = True
                if command[14] == 128:
                    if distances[ext_counter][1] > 0:
                        sequence[i][14] += 1
                        flag = True
                    else:
                        sequence[i][14] -= 1
                        flag = True
            ext_counter += 1

    return sequence, flag   

def adjust_pointcloud_to_fixed_size(points, labels, target_n=10000):
    """
    Adjust a point cloud and its labels to a fixed number of points.

    - If len(points) > target_n: randomly downsample
    - If len(points) < target_n: randomly upsample with replacement

    :param points: (N, 3) np.ndarray
    :param labels: (N,) np.ndarray
    :param target_n: int, desired number of points
    :return: (target_n, 3) points, (target_n,) labels
    """
    n = points.shape[0]

    if n == target_n:
        return points, labels
    elif n > target_n:
        idx = np.random.choice(n, target_n, replace=False)
    else:
        idx_extra = np.random.choice(n, target_n - n, replace=True)
        idx = np.concatenate([np.arange(n), idx_extra])

    return points[idx], labels[idx]

def main(args):
    # Data creation settings

    DATA_DIR = args.data_root
    nr_points_gt = 40000
    nr_points_ext = 40000
    epsilon = 0.005
    num_points_seg = 10000

    dataset_train = PointCloudEmbeddingSequenceDataset(DATA_DIR, 'train', use_normals=False,
                                                       n_points=N_POINTS_SAMPLE, seed=SEED)
    dataset_val = PointCloudEmbeddingSequenceDataset(DATA_DIR, 'validation', use_normals=False,
                                                     n_points=N_POINTS_SAMPLE, seed=SEED)
    dataset_test = PointCloudEmbeddingSequenceDataset(DATA_DIR, 'test', use_normals=False,
                                                      n_points=N_POINTS_SAMPLE, seed=SEED)
    datasets = [dataset_train, dataset_val, dataset_test]

    error_dict = {}
    error_types = Counter()
    update_dict = {}
    n_total = 0

    for dataset in datasets:
        length = len(dataset)
        
        for i in range(length): #:
            n_total += 1

            try:
                data = dataset[i]
                sequence = data['tgt_vec'].numpy()
                id = data['id']
                sequence, flag = check_sequence(sequence, id, DATA_DIR)

                if flag: # If sequence contains errors, update it in the orignal targets/h5 file
                    h5_path = dataset.get_cad_seq_path(i)
                    with h5py.File(h5_path, 'w') as f:
                        for k, command in enumerate(sequence):
                            if command[0] == 3:
                                end = k
                                break
                        f.create_dataset("vec", data=sequence[:end + 1], compression='gzip')
                    update_dict[f"{dataset.split}/{i}"] = f"Updated sequence for {h5_path} with id {id}."
        
                print(f"\rProcessing {dataset.split} {i + 1}/{length} | ID: {id} | {len(error_dict)} failed", end="")
                sys.stdout.flush()
        
                pc_path = os.path.join(DATA_DIR, "pc_from_vec", id[:4], id + ".ply")
                h5_path = os.path.join(DATA_DIR, "pc_from_vec_labels", id[:4], id + ".h5")
                
                gt_pc = get_gt_pc(sequence, n_points=nr_points_gt)
                labled_ext_pc, ext_labels, extrusions = get_labled_pc_per_ext(sequence, nr_points=nr_points_ext)
                pc, labels = filter_pc_and_labels(gt_pc, labled_ext_pc, ext_labels, epsilon=epsilon)
                
                pc, labels = adjust_pointcloud_to_fixed_size(pc, labels, target_n=num_points_seg)
        
                save_pc(pc, pc_path)
                save_labels_with_ext(labels, extrusions, h5_path)

            except Exception as e:
                error_dict[f"{dataset.split}/{i}"] = repr(e)
                error_types[type(e).__name__] += 1
        print()

    # Save
    with open("error_dict.pkl", "wb") as f:
        pickle.dump(error_dict, f)
    with open("update_dict.pkl", "wb") as f:
        pickle.dump(update_dict, f)

    n_failed = len(error_dict)
    print(f"{n_failed} of {n_total} samples failed")
    for name, count in error_types.most_common(5):
        print(f"  {count:>8,}  {name}")

    failure_rate = n_failed / n_total if n_total else 0.0
    if failure_rate > MAX_FAILURE_RATE:
        print(f"failure rate {failure_rate:.2%} exceeds the {MAX_FAILURE_RATE:.0%} threshold "
              f"-- this looks systematic, not like malformed CAD", file=sys.stderr)
        sys.exit(1)



if __name__ == '__main__':
    args = parse_args()
    main(args)
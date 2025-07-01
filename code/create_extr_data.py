import os
import h5py
import numpy as np

import open3d as o3d

from dataset import PCExtrusionSegmentationDataset


def save_pc(pc, path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.io.write_point_cloud(path, pcd)

def save_h5(extr_id, sequence, path):
    with h5py.File(path, "w") as f:
        f.create_dataset("extrusion_id", data=extr_id)
        f.create_dataset("sequence", data=sequence)

def split_pc_by_labels(pc: np.ndarray, labels: np.ndarray):
    class_pcs = []
    for class_id in np.unique(labels):
        class_mask = (labels == class_id)
        class_pc = pc[class_mask]
        class_pcs.append(class_pc)
    return class_pcs

def main():

    train_dataset = PCExtrusionSegmentationDataset("../data", 'train', use_normals=False, verbose=False)
    val_dataset = PCExtrusionSegmentationDataset("../data", 'validation', use_normals=False, verbose=False)
    test_dataset = PCExtrusionSegmentationDataset("../data", 'test', use_normals=False, verbose=False)
    datasets = [train_dataset, val_dataset, test_dataset]

    DATA_DIR = "../data"
    error_dict = {}

    for dataset in datasets:
        length = len(dataset)
        
        for i in range(length):
            print(f"\r{i}/{length}", end="")

            try:
                data = dataset[i]
                id = data['id']
                pc = data['pc']
                label = data['label']
                assert pc.shape[0] == 10000, "Point cloud from pc_from_vec has 10000 points, but got {}".format(pc.shape[0])
        
                pcs = split_pc_by_labels(pc, label)
        
                sequences = {}
                h5_path = os.path.join(DATA_DIR, "pc_from_vec_labels", id[:4], id + ".h5") 
                with h5py.File(h5_path, 'r') as f:
                    for class_id, sequence in f['sequences'].items():
                        sequence = sequence[:]
                        seq_length = np.where(sequence[:, 0] == 3)[0][0] + 1 # only save up to including the first EOS command
                        sequence = sequence[:seq_length, :]
                        sequences[class_id] = sequence
                
                for j, pc in enumerate(pcs):
                    pc_path = os.path.join(os.path.abspath(DATA_DIR), "pc_extrusion", id[:4], id, id + "_" + str(j) + ".ply")
                    os.makedirs(os.path.dirname(pc_path), exist_ok=True)
                    save_pc(pc, pc_path)
                    
                    h5_path = os.path.join(os.path.abspath(DATA_DIR), "pc_extrusion_labels", id[:4], id, id + "_" + str(j) + ".h5")
                    os.makedirs(os.path.dirname(h5_path), exist_ok=True)
                    sequence = sequences[str(j)]
                    save_h5(j, sequence, h5_path)
                    
            except Exception as e:
                error_dict[i] = e

    if error_dict:
        for k,v in error_dict.items():
            print(f"Error at index {k}: {v}")
    else:
        print("All point clouds processed successfully.")

if __name__ == "__main__":
    main()
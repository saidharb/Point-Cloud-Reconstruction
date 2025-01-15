import os
from glob import glob

import h5py
import numpy as np
import argparse
import open3d as o3d
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

class PointCloudEmbeddingDataset(Dataset):
    
    def __init__(self, root, split):
        
        self.root = root
        assert split in {'train', 'validation', 'test'}, f"Invalid split '{split}'. Valid options are: 'train', 'validation', 'test'"
        self.split = split
        print(f"### Loading {self.split} dataset ###\n", flush=True)
        self.split_path = os.path.join(root, "train_val_test_split.json")
        self.pc_path = os.path.join(root, "pc_cad")
        self.latent_path = os.path.join(root, "latent/pretrained/results/all_zs_ckpt1000.h5")

        # Point Clouds
        file_pattern = "**/*.ply"
        self.all_files = set(glob(f"{self.pc_path}/{file_pattern}", recursive=True))
        pc_all = self.read_split()
        self.pc, self.corrupt_idx = self.filter_pc(pc_all)
        assert(self.check_valid_pc(self.pc))

        # Latent Representations
        with h5py.File(self.latent_path, 'r') as f:
            latent_all = np.array(f[self.split + '_zs'])
        assert(self.check_latent_valid(latent_all))
        self.latent = self.filter_latent(latent_all)
        
    def __len__(self):
        assert(len(self.pc) == len(self.latent)), f"The number of point clouds {len(self.pc)} is different from the number of latent representations {len(self.latent)}"
        return len(self.pc)
        
    def __getitem__(self, idx):
        point_cloud = o3d.io.read_point_cloud(self.pc[idx])
        point_cloud = torch.tensor(np.asarray(point_cloud.points), dtype = torch.float32)
        latent_rep = torch.tensor(self.latent[idx], dtype = torch.float32)
        return point_cloud, latent_rep

    def read_split(self):
        with open(self.split_path, "r") as fp:
            all_data = json.load(fp)
        print(f"Number of samples that should be in the {self.split} set: {len(all_data[self.split])}", flush=True)
        pc_set = [os.path.join(self.pc_path, f"{idx}.ply") for idx in all_data[self.split]]
        return pc_set

    def filter_latent(self, latent_set):
        latent = list(np.delete(latent_set, self.corrupt_idx, axis = 0))
        return latent

    def filter_pc(self, pc_set):
        pc_set_filtered = [entry for entry in pc_set if entry in self.all_files]
        corrupt_files = [i for i, entry in enumerate(pc_set) if entry not in self.all_files]
        print(f"Files on disk: {len(pc_set_filtered)} --> There are {len(corrupt_files)} missing point cloud files in the {self.split} set.\n", flush=True)
        return pc_set_filtered, corrupt_files

    def check_valid_pc(self, paths_list):
        print(f"Checking if all point clouds are valid in the {self.split} set.", flush=True)
        corrupt_counter = 0
        for pc_file in tqdm(paths_list):
            try:
                pc = o3d.io.read_point_cloud(pc_file)
                if not pc.has_points():
                    corrupt_counter += 1
            except Exception as e:
                corrupt_counter += 1
        print(f"There are {corrupt_counter} corrupt files in the {self.split} set.\n", flush=True)
        if corrupt_counter == 0:
            return True
        else:
            return False

    def check_latent_valid(self, data):
        print("Checking latent representation.", flush=True)
        
        # Zero rows
        zero_rows = np.all(data == 0, axis=1)
        num_zero_rows = np.sum(zero_rows)
        if num_zero_rows > 0:
            print(f"There are {num_zero_rows} zero rows.", flush=True)
        
        # NaN or Inf values
        nan_or_inf_rows = np.any(np.isnan(data) | np.isinf(data), axis=1)
        num_nan_or_inf_rows = np.sum(nan_or_inf_rows)
        if num_nan_or_inf_rows > 0:
            print(f"There are {num_nan_or_inf_rows} rows with NaN or Inf values.", flush=True)
        
        if num_zero_rows == 0 and num_nan_or_inf_rows == 0:
            print("All latent represenations are valid.\n", flush=True)
            return True
        else:
            return False
        
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser(
        'Train PointNet++ to encode point clouds into the same latent space as DeepCAD '
        'does with CAD Sequences.'
    )
    # for the data_root argument: 
    # I want the user to give the data directory from the root directory
    # It would be kind of laborious to construct the relative file path to data from this script
    parser.add_argument('--data_root', type=str, default='data', help='data directory relative to root directory')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    return parser.parse_args()

def main(args):

    # Find data directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if os.getcwd() != root_dir:
        DATA_DIR = os.path.join(root_dir, args.data_root)
    else:
        DATA_DIR = os.path.abspath(args.data_root)

    # Print parameters
    print("### Parameters ###\n", flush=True)
    print(f"Data directory: {DATA_DIR}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)
    print("", flush=True)

    train_dataset = PointCloudEmbeddingDataset(DATA_DIR, 'test')
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)

if __name__ == '__main__':
    args = parse_args()
    main(args)

    

# For README:
# Execute train.py either from root or from ./code
# Give the data directory always relative to root (makes it easier)
import os
from glob import glob
import json
from abc import ABC, abstractmethod
import random

import h5py
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from models.DeepCAD.cadlib.macro import EOS_VEC, MAX_TOTAL_LEN

N_POINTS = 2048

class BaseDataset(Dataset, ABC):
    def __init__(self, root, split, verbose=False):
        self.root = root
        self.split = split
        self.verbose = verbose
        assert split in {'train', 'validation', 'test'}, f"Invalid split '{split}'. Valid options are: 'train', 'validation', 'test'"
        if self.verbose:
            print(f"Loading {self.split} dataset \n", flush=True)

        self.split_path = os.path.join(root, "train_val_test_split.json")
        self.pc_path = os.path.join(root, "pc_cad")
        self.latent_path = os.path.join(root, "latent/pretrained/results/all_zs_ckpt1000.h5")
        self.cad_seq_path = os.path.join(root, "cad_vec")

        # Point-Clouds
        pc_all = self.read_split() # Files in official split
        pc_file_pattern = os.path.join("**", "*") + ".ply"
        self.all_pc_files = set(glob(f"{os.path.join(self.pc_path, pc_file_pattern)}", recursive=True)) # Files on disk
        self.pc, self.corrupt_idx = self.filter_pc(pc_all)
        # assert(self.check_valid_pc(self.pc))

        # CAD-sequences
        self.cad_seq = [os.path.splitext(f)[0] + '.h5' for f in self.pc]
        self.cad_seq = [f.replace('pc_cad', 'cad_vec') for f in self.cad_seq]

        # Latent Representations
        with h5py.File(self.latent_path, 'r') as f:
            latent_all = np.array(f[self.split + '_zs'])
        assert(self.check_latent_valid(latent_all))
        self.latent = self.filter_latent(latent_all)
        if self.verbose:
            print("--- DONE ---\n", flush=True)
    
    def __len__(self):
        assert(len(self.pc) == len(self.latent)), f"The number of point clouds {len(self.pc)} is different from the number of latent representations {len(self.latent)}"
        return len(self.pc)
    
    @abstractmethod
    def __getitem__(self, idx):
        pass
    
    def get_pc_path(self, idx):
        return self.pc[idx]
    
    def get_id(self, idx):
        pc_path = self.get_pc_path(idx)
        id = os.path.splitext(os.path.basename(pc_path))[0]
        return id
    
    def get_cad_seq_path(self, idx):
        return self.cad_seq[idx]
    
    def read_split(self):
        with open(self.split_path, "r") as fp:
            all_data = json.load(fp)
        if self.verbose:
            print(f"Number of samples that should be in the {self.split} set: {len(all_data[self.split])}", flush=True)
        pc_set = [os.path.join(self.pc_path, f"{idx}.ply") for idx in all_data[self.split]]
        return pc_set
    
    def filter_latent(self, latent_set):
        latent = list(np.delete(latent_set, self.corrupt_idx, axis = 0))
        return latent
    
    def filter_pc(self, pc_set):
        pc_set_filtered = [entry for entry in pc_set if entry in self.all_pc_files]
        corrupt_files = [i for i, entry in enumerate(pc_set) if entry not in self.all_pc_files]
        if self.verbose:
            print(f"Files on disk: {len(pc_set_filtered)} --> There are {len(corrupt_files)} missing point cloud files in the {self.split} set.\n", flush=True)
        return pc_set_filtered, corrupt_files
    
    def check_valid_pc(self, paths_list):
        if self.verbose:
            print(f"Checking if all point clouds are valid in the {self.split} set:", flush=True)
        corrupt_counter = 0
        for pc_file in tqdm(paths_list):
            try:
                pc = o3d.io.read_point_cloud(pc_file)
                if not pc.has_points():
                    corrupt_counter += 1
            except Exception as e:
                corrupt_counter += 1
        if self.verbose:
            print(f"There are {corrupt_counter} corrupt files in the {self.split} set.\n", flush=True)
        if corrupt_counter == 0:
            return True
        else:
            return False
        
    def check_latent_valid(self, data):
        if self.verbose:
            print("Checking latent representation:", flush=True)
        zero_rows = np.all(data == 0, axis=1)
        num_zero_rows = np.sum(zero_rows)
        if num_zero_rows > 0:
            if self.verbose:
                print(f"There are {num_zero_rows} zero rows.", flush=True)
        nan_or_inf_rows = np.any(np.isnan(data) | np.isinf(data), axis=1)
        num_nan_or_inf_rows = np.sum(nan_or_inf_rows)
        if num_nan_or_inf_rows > 0:
            if self.verbose:
                print(f"There are {num_nan_or_inf_rows} rows with NaN or Inf values.", flush=True)
        if num_zero_rows == 0 and num_nan_or_inf_rows == 0:
            if self.verbose:
                print("All latent represenations are valid.\n", flush=True)
            return True
        else:
            return False

class PointCloudEmbeddingDataset(BaseDataset):
    
    def __init__(self, root, split, verbose=False):
        super().__init__(root, split, verbose=verbose)
        
    def __getitem__(self, idx):
        point_cloud = o3d.io.read_point_cloud(self.pc[idx])
        point_cloud = torch.tensor(np.asarray(point_cloud.points), dtype = torch.float32) # [B, N, C]
        sample_idx = random.sample(list(range(point_cloud.shape[0])), N_POINTS)
        point_cloud = point_cloud[sample_idx]
        latent_rep = torch.tensor(self.latent[idx], dtype = torch.float32) # [B, D]
        return point_cloud, latent_rep


class PointCloudEmbeddingSequenceDataset(BaseDataset):
    
    def __init__(self, root, split, verbose=False):
        super().__init__(root, split, verbose=verbose)
        
    def __getitem__(self, idx):
        point_cloud = o3d.io.read_point_cloud(self.pc[idx])
        point_cloud = torch.tensor(np.asarray(point_cloud.points), dtype = torch.float32) # [B, N, C]
        latent_rep = torch.tensor(self.latent[idx], dtype = torch.float32) # [B, D]
        
        with h5py.File(self.cad_seq[idx], 'r') as fp:
            cad_vec = fp["vec"][:]
        pad_len = MAX_TOTAL_LEN - cad_vec.shape[0]   
        cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
        cad_vec = torch.tensor(cad_vec, dtype=torch.long)
        #cad_seq =  #torch.tensor(self.cad_seq[idx], dtype=torch.int64)
        id = self.get_id(idx)
        data = {"pc": point_cloud,
                "z": latent_rep,
                "tgt_vec": cad_vec,
                "id":  id
            }
        return data

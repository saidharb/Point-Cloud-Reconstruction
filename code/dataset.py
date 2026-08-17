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
import pickle

from models.DeepCAD.cadlib.macro import EOS_VEC, MAX_TOTAL_LEN

# Number of points per cloud in the generated dataset (`pc_from_vec`). This is the
# value the released data was generated with and the size `create_extr_data.py`
# requires, so the generation scripts pass it explicitly.
N_POINTS = 10000

# Number of points sampled from a cloud during training / inference. This is the
# value the published results were produced with. It is deliberately kept separate
# from N_POINTS: `pc_cad` only holds 8096 points, so the generation size is not even
# a legal sample size here, and training may legitimately want to change it without
# touching the generation size.
N_POINTS_TRAIN = 2048

class BaseDataset(Dataset, ABC):
    def __init__(self, root, split, use_normals=False, verbose=False, n_points=None, seed=None):
        self.root = root
        self.split = split
        self.use_normals = use_normals
        self.verbose = verbose
        # Number of points to sample per cloud. Defaults to the training size; the
        # generation scripts override it with the size they need.
        self.n_points = N_POINTS_TRAIN if n_points is None else n_points
        # Unseeded by default, so training augmentation is unchanged. The generation
        # scripts pass a seed so that regeneration is deterministic.
        self.rng = random if seed is None else random.Random(seed)
        assert split in {'train', 'validation', 'test'}, f"Invalid split '{split}'. Valid options are: 'train', 'validation', 'test'"
        if self.verbose:
            print(f"Loading {self.split} dataset \n", flush=True)

        self.split_path = os.path.join(root, "train_val_test_split.json")
        if self.use_normals:
            self.pc_path = os.path.join(root, "pc_cad_norm")
        else:
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
        if not self.use_normals:
            self.cad_seq = [os.path.splitext(f)[0] + '.h5' for f in self.pc]
            self.cad_seq = [f.replace('pc_cad', 'cad_vec') for f in self.cad_seq]
        else:
            self.cad_seq = [os.path.splitext(f)[0] + '.h5' for f in self.pc]
            self.cad_seq = [f.replace('pc_cad_norm', 'cad_vec') for f in self.cad_seq]


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
    
    def __init__(self, root, split, use_normals=False, verbose=False, n_points=None, seed=None):
        super().__init__(root, split, use_normals=use_normals, verbose=verbose,
                         n_points=n_points, seed=seed)

    def __getitem__(self, idx):
        point_cloud = o3d.io.read_point_cloud(self.pc[idx])
        if self.use_normals: 
            normals = np.asarray(point_cloud.normals)
            point_cloud = torch.tensor(np.hstack((point_cloud.points, normals)), dtype = torch.float32) # [N, C + 3]
        else:
            point_cloud = torch.tensor(np.asarray(point_cloud.points), dtype = torch.float32) # [N, C]
        sample_idx = self.rng.sample(list(range(point_cloud.shape[0])), self.n_points)
        point_cloud = point_cloud[sample_idx]
        latent_rep = torch.tensor(self.latent[idx], dtype = torch.float32) # [256]
        return point_cloud, latent_rep


class PointCloudEmbeddingSequenceDataset(BaseDataset):
    
    def __init__(self, root, split, use_normals=False, verbose=False, n_points=None, seed=None):
        super().__init__(root, split, use_normals=use_normals, verbose=verbose,
                         n_points=n_points, seed=seed)

    def __getitem__(self, idx):
        point_cloud = o3d.io.read_point_cloud(self.pc[idx])
        if self.use_normals: 
            normals = np.asarray(point_cloud.normals)
            point_cloud = torch.tensor(np.hstack((point_cloud.points, normals)), dtype = torch.float32) # [N, C + 3]
        else:
            point_cloud = torch.tensor(np.asarray(point_cloud.points), dtype = torch.float32) # [N, C]
        sample_idx = self.rng.sample(list(range(point_cloud.shape[0])), self.n_points)
        point_cloud = point_cloud[sample_idx]
        latent_rep = torch.tensor(self.latent[idx], dtype = torch.float32) # [B, D]
        
        with h5py.File(self.cad_seq[idx], 'r') as fp:
            cad_vec = fp["vec"][:]
        pad_len = MAX_TOTAL_LEN - cad_vec.shape[0]   
        cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
        cad_vec = torch.tensor(cad_vec, dtype=torch.long)
        id = self.get_id(idx)
        data = {"pc": point_cloud,
                "z": latent_rep,
                "tgt_vec": cad_vec,
                "id":  id
            }
        return data

class PCExtrusionSegmentationDataset(BaseDataset):
    
    def __init__(self, root, split, use_normals=False, verbose=False, n_points=None, seed=None):
        super().__init__(root, split, use_normals=use_normals, verbose=verbose,
                         n_points=n_points, seed=seed)
        self.pc_path = os.path.join(root, "pc_from_vec")

        pc_all = self.read_split() # Files in official split
        pc_file_pattern = os.path.join("**", "*") + ".ply"
        self.all_pc_files = set(glob(f"{os.path.join(self.pc_path, pc_file_pattern)}", recursive=True)) # Files on disk
        self.pc, self.corrupt_idx = self.filter_pc(pc_all)

        self.label_files = [os.path.splitext(f)[0] + '.h5' for f in self.pc]
        self.all_label_files = [f.replace("pc_from_vec", "pc_from_vec_labels") for f in self.label_files]

    def __getitem__(self, idx):
        point_cloud = o3d.io.read_point_cloud(self.pc[idx])
        point_cloud = torch.tensor(np.asarray(point_cloud.points), dtype = torch.float32) # [N, C]
        sample_idx = self.rng.sample(list(range(point_cloud.shape[0])), self.n_points)
        point_cloud = point_cloud[sample_idx]

        with h5py.File(self.all_label_files[idx], 'r') as fp:
            labels = fp["labels"][:]
        labels = torch.tensor(labels, dtype=torch.long)
        labels = labels[sample_idx]

        id = self.get_id(idx)
        data = {"pc": point_cloud,
                "label": labels,
                "id":  id
            }
        return data
    
    def __len__(self):
        return len(self.pc)
    
class PCExtrusionSequenceDataset():
    def __init__(self, root, split, cfg, verbose=False, n_points=None):
        self.n_points = N_POINTS_TRAIN if n_points is None else n_points
        self.pc_path = os.path.join(root, "pc_extrusion")
        self.split_path = os.path.join(root, "train_val_test_split.json")
        self.verbose = verbose
        self.split = split
        self.cfg = cfg

        # Scrutinize if the id's mentioned in the split actually exist as directories
        valid_dirs = []
        pc_all = self.read_split()
        for dir in pc_all:
            if os.path.isdir(dir):
                valid_dirs.append(dir)
        
        self.pc = []
        self.labels = []

        CACHE_FILE = f"pc_dataset_cache_{self.split}.pkl"
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'rb') as f:
                self.pc, self.labels = pickle.load(f)
        else:
            for dir in valid_dirs:
                pc_paths = glob(os.path.join(dir, "*.ply"))
                for ply_path in pc_paths:
                    h5_path = ply_path.replace("pc_extrusion", "pc_extrusion_labels").replace(".ply", ".h5")
            
                    if not os.path.exists(h5_path):
                        continue
                    try:
                        with h5py.File(h5_path, "r") as f:
                            if "sequence" in f and len(f["sequence"]) > 0:
                                self.pc.append(ply_path)
                                self.labels.append(h5_path)
                    except Exception as e:
                        if self.verbose:
                            print(f"Skipped corrupted or unreadable file: {h5_path} ({e})")
            with open(CACHE_FILE, "wb") as f:
                pickle.dump((self.pc, self.labels), f)

    def __getitem__(self, idx):
        point_cloud = o3d.io.read_point_cloud(self.pc[idx])
        point_cloud = np.asarray(point_cloud.points)
        point_cloud = self.adjust_pointcloud_to_fixed_size(point_cloud, self.n_points)
        point_cloud = torch.tensor(point_cloud, dtype = torch.float32) # [N, C]

        with h5py.File(self.labels[idx], "r") as f:
            extr_id = f['extrusion_id'][()]
            sequence = f['sequence'][:]

        extr_id = torch.tensor(extr_id, dtype=torch.long)
        sequence = torch.tensor(sequence, dtype=torch.long)

        pad_len = self.cfg.max_total_len - sequence.shape[0]
        cad_vec = np.concatenate([sequence, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
        cad_vec = torch.tensor(cad_vec, dtype=torch.long)

        id = self.get_id(idx)
        
        return {'pc': point_cloud,
               'extrusion_id': extr_id,
               'sequence': cad_vec,
               'id': id,
               'pc_path': self.pc[idx]}

    def read_split(self):
        with open(self.split_path, "r") as fp:
            all_data = json.load(fp)
        if self.verbose:
            print(f"Number of samples that should be in the {self.split} set: {len(all_data[self.split])}", flush=True)
        pc_set = [os.path.join(self.pc_path, f"{idx}") for idx in all_data[self.split]]
        return pc_set

    def filter_pc(self, pc_set):
        pc_set_filtered = [entry for entry in pc_set if entry in self.all_pc_files]
        corrupt_files = [i for i, entry in enumerate(pc_set) if entry not in self.all_pc_files]
        if self.verbose:
            print(f"Files on disk: {len(pc_set_filtered)} --> There are {len(corrupt_files)} missing point cloud files in the {self.split} set.\n", flush=True)
        return pc_set_filtered, corrupt_files

    def __len__(self):
        assert len(self.pc) == len(self.labels), "Number of point clouds and labels doesn't match"
        return len(self.pc)

    def get_indices_for_id(self, id):
        """
        Returns the indices of all subcomponents of a point cloud with the given id,
        sorted lexicographically by their point cloud paths.
        """
        indice_list = [i for i, pc_path in enumerate(self.pc) if id in os.path.basename(pc_path)]
        indexed_paths = [(i, self.pc[i]) for i in indice_list]
        sorted_indexed_paths = sorted(indexed_paths, key=lambda x: x[1])
        sorted_indices = [i for i, _ in sorted_indexed_paths]
        
        return sorted_indices

    def get_pc_path(self, idx):
        return self.pc[idx] 

    def get_id(self, idx):
        pc_path = self.get_pc_path(idx)
        return os.path.basename(pc_path).replace(".ply", "")

    def get_label_path(self, idx):
        return self.labels[idx]

    def adjust_pointcloud_to_fixed_size(self, points, target_n):
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
            return points
        elif n > target_n:
            idx = np.random.choice(n, target_n, replace=False)
        else:
            idx_extra = np.random.choice(n, target_n - n, replace=True)
            idx = np.concatenate([np.arange(n), idx_extra])
    
        return points[idx]
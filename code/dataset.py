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

N_POINTS = 2048

class BaseDataset(Dataset, ABC):
    def __init__(self, root, split, use_normals=False, verbose=False):
        self.root = root
        self.split = split
        self.use_normals = use_normals
        self.verbose = verbose
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
    
    def __init__(self, root, split, use_normals=False, verbose=False):
        super().__init__(root, split, use_normals=use_normals, verbose=verbose)
        
    def __getitem__(self, idx):
        point_cloud = o3d.io.read_point_cloud(self.pc[idx])
        if self.use_normals: 
            normals = np.asarray(point_cloud.normals)
            point_cloud = torch.tensor(np.hstack((point_cloud.points, normals)), dtype = torch.float32) # [N, C + 3]
        else:
            point_cloud = torch.tensor(np.asarray(point_cloud.points), dtype = torch.float32) # [N, C]
        sample_idx = random.sample(list(range(point_cloud.shape[0])), N_POINTS)
        point_cloud = point_cloud[sample_idx]
        latent_rep = torch.tensor(self.latent[idx], dtype = torch.float32) # [256]
        return point_cloud, latent_rep

# IGNORE_INDICES = [5238, 912, 204, 6074, 2253, 2006, 1828, 1143, 6033, 839, 5543, 6067, 7308, 4467, 712, 4837, 3456, 260, 244, 767, 1791, 1905, 4139, 4931, 217, 4597, 1628, 5865, 5323, 5745, 4464, 3436, 1805, 3679, 4827, 2278, 6630, 7121, 53, 6216, 6601, 1307, 5719, 3462, 2787, 2276, 1273, 1763, 7841, 6254, 2757, 837, 759, 3112, 792, 2940, 6942, 2817, 4945, 2166, 6611, 355, 5977, 3763, 4392, 1022, 7971, 7555, 3100, 645, 4522, 2401, 6794, 5149, 5066, 7253, 7059, 2962, 4729, 1575, 5771, 569, 375, 5417, 1866, 6332, 2370, 653, 7006, 1907, 7098, 827, 3113, 2277, 3714, 5207, 6833, 2988, 1332, 3032, 2910, 1716, 5490, 2187, 5749, 7673, 5599, 5308, 584, 4990, 5201, 1401, 4375, 5973, 2005, 1338, 3786, 3108, 2211, 7580, 5242, 5637, 4562, 1799, 5608, 2656, 6904, 6294, 6356, 458, 1876, 6732, 262, 6594, 2584, 3286, 2193, 542, 1728, 7480, 7724, 4646, 7179, 5881, 2577, 1741, 5369, 4089, 3241, 7247, 7491, 5266, 3758, 1170, 2169, 8088, 2020, 6102, 4598, 4415, 2152, 6119, 4788, 3509, 7354, 4780, 3271, 2965, 1796, 1133, 4174, 4042, 744, 6191, 385, 7054, 898, 1252, 5140, 1310, 6488, 5574, 3458, 4885, 520, 3152, 3126, 4881, 3834, 4334, 2059, 4532, 7051, 7720, 94, 5572, 5904, 938, 5584, 7248, 4398, 6151, 2185, 6296, 5250, 2786, 913, 2404, 3561, 1295, 3716, 26, 7813, 5915, 7174, 5895, 2157, 4100, 6241, 1463, 4158, 7476, 871, 7131, 5122, 2444, 6894, 5234, 7875, 4988, 1629, 7918, 3063, 6246, 1323, 4418, 7811, 6378, 7556, 4344, 7524, 4, 4906, 2655, 4002, 159, 916, 7611, 2973, 7198, 6813, 6609, 2519, 1961, 474, 1973, 7192, 4647, 7757, 8026, 701, 5995, 3981, 6684, 566, 6230, 4363, 6273, 1030, 1051, 5404, 3893, 7756, 4503, 1352, 2171, 4322, 7146, 4969, 3466, 1735, 7609, 4417, 6187, 5979, 5651, 1647, 5840, 2553, 3268, 5502, 8067, 3059, 3588, 7369, 4239, 3698, 991, 2030, 1840, 524, 2769, 172, 4819, 4537, 1885, 4820, 1804, 58, 581, 5798, 5169, 482, 1875, 552, 7417, 257, 7042, 2706, 580, 4211, 1949, 2281, 5480, 3976, 1755, 7883, 1083, 5925, 7663, 7228, 4677, 4720, 3872, 1990, 6427, 3874, 6615, 3334, 1559, 772, 794, 5398, 3531, 2902, 3469, 3367, 3825, 7076, 5972, 443, 5516, 5353, 5293, 806, 496, 3298, 5965, 2779, 6558, 7058, 895, 2036, 1569, 1558, 4393, 3675, 1148, 8079, 1503, 7773, 3789, 2046, 7163, 7563, 617, 3630, 6619, 7056, 7012, 4508, 802, 414, 5342, 4428, 6848, 120, 764, 7588, 6173, 6952, 1936, 1362, 3329, 3978, 3943, 1751, 7083, 3285, 7392, 480, 1348, 3104, 17, 3198, 2172, 7590, 6423, 6431, 3727, 2336, 3465, 5706, 5984, 6417, 4552, 5422, 5885, 3986, 1268, 1555, 2430, 1783, 479, 4744, 6027, 4441, 499, 6127, 2569, 468, 410, 4785, 3905, 4119, 7531, 6985, 4350, 1289, 465, 4160, 656, 6974, 1522, 561, 4874, 556, 5531, 7060, 1926, 3307, 982, 7293, 4666, 2016, 4742, 4870, 325, 5073, 671, 3434, 5385, 4781, 4630, 4282, 2591, 2136, 1673, 5486, 5867, 2573, 1955, 2175, 3242, 1072, 7805, 5287, 2457, 3745, 2590, 7610, 6159, 594, 76, 3754, 5088, 4612, 819, 600, 4404, 1746, 4144, 7690, 1085, 2859, 7216, 563, 7202, 2001, 3027, 2334, 1292, 3589, 6830, 4450, 5763, 2478, 5010, 8035, 5357, 4333, 64, 5471, 6693, 4543, 2452, 5434, 848, 7193, 1100, 8036, 945, 7288, 876, 6081, 7904, 8049, 2231, 2308, 4954, 1725, 5878, 2808, 1667, 5631, 5195, 6987, 2162, 4140, 7851, 2057, 7416, 7439, 6931, 416, 756, 5196, 7750, 8023, 2266, 361, 29, 2732, 6317, 1071, 5219, 2145, 7861, 6072, 3619, 4519, 5780, 3503, 4594, 79, 7849, 616, 7234, 5660, 7404, 1221, 4469, 295, 6837, 3024, 4771, 4526, 1213, 3520, 1044, 342, 2525, 2987, 7364, 6521, 7050, 326, 7363, 2931, 1720, 5587, 2044, 5463, 842, 2897, 6390, 4586, 7242, 7165, 7702, 5084, 6139, 1266, 1939, 7082, 1331, 6552, 6641, 1450, 7220, 3377, 203, 1469, 8087, 2721, 6409, 3372, 6571, 5487, 7078, 6021, 7488, 2032, 7893, 1304, 6450, 5744, 885, 3133, 7144, 317, 7033, 3855, 1822, 1634, 6689, 3770, 2864, 2500, 6721, 6515, 7137, 1864, 1826, 193, 5406, 1582, 3264, 2689, 2282, 7080, 568, 6335, 2286, 2876, 5255, 4173, 3274, 5566, 6911, 8031, 2712, 226, 944, 7184, 2139, 1462, 4756, 2174, 313, 888, 4887, 3559, 2831, 5968, 6442, 7666, 3574, 4966, 4189, 947, 3155, 7368, 4723, 1557, 2086, 363, 5806, 3572, 13, 4259, 6606, 4410, 5626, 5893, 6078, 6037, 5493, 1614, 2983, 3533, 573, 5441, 2704, 5104, 2571, 7579, 6946, 1020, 5896, 7373, 2460, 4154, 2533, 7503, 3345, 2672, 3296, 5711, 2422, 4541, 1042, 1571, 3444, 5447, 3105, 5548, 6128, 1425, 5042, 4662, 2465, 3326, 4488, 6829, 3, 2489, 2350, 1721, 3521, 6436, 4751, 7817, 5363, 2639, 3809, 7536, 3622, 5534, 1750, 4187, 3876, 6502, 7511, 6028, 1390, 5397, 694, 2324, 4222, 5438, 5185, 5072, 2745, 765, 6704, 6153, 1924, 5511, 2542, 7796, 6607, 1631, 1207, 200, 378, 7981, 3892, 5007, 6961, 6295, 596, 3730, 3395, 7259, 5159, 4715, 1592, 5884, 5704, 3145, 4049, 3273, 1998, 1208, 5374, 5633, 45, 7894, 7048, 6308, 8020, 873, 6376, 3482, 1792, 1440, 6587, 5700, 4243, 3805, 411, 4566, 2041, 6949, 994, 3739, 1092, 6565, 3806, 5468, 4351, 4578, 4877, 2599, 7812, 3625, 5018, 6676, 5892, 4135, 3495, 6804, 7362, 3652, 1303, 6092, 7057, 3888, 3686, 2123, 6158, 2025, 6880, 5223, 2271, 7828, 6370, 4270, 3969, 5134, 1959, 2249, 3603, 634, 5845, 2340, 1920, 2225, 2751, 2619, 4424, 660, 7926, 1235, 1894, 3137, 5684, 1251, 5786, 1752, 526, 3398, 3339, 2710, 4445, 3816, 3406, 510, 1694, 6823, 3441, 3190, 6306, 4784, 5697, 160, 7018, 6271, 4716, 3116, 3907, 48, 2881, 2446, 6172, 3194, 6991, 6846, 3432, 4409, 6123, 6018, 4473, 7489, 4941, 1806, 3999, 1797, 2235, 3570, 7701, 237, 3185, 2753, 5479, 5563, 6538, 3312, 5932, 7821, 6885, 3828, 1045, 5097, 7983, 220, 3227, 4848, 4623, 5431, 222, 687, 5265, 3511, 1111, 7101, 3782, 1488, 7284, 2131, 7370, 2681, 1733, 3724, 2677, 2764, 6235, 7154, 2279, 6160, 6815, 3453, 2066, 6839, 670, 3852, 158, 6136, 7860, 426, 2866, 1836, 5325, 562, 6399, 5339, 329, 6178, 254, 7294, 1633, 6874, 166, 5089, 1248, 1954, 1034, 3879, 5484, 937, 4620, 1785, 7350, 5730, 2099, 6282, 3021, 1374, 4963, 4974, 7667, 7307, 7898, 6371, 6711, 1341, 2548, 7468, 4740, 210, 2555, 7206, 7801, 3074, 3249, 5857, 1624, 622, 4850, 5657, 7263, 5138, 1989, 834, 7377, 6328, 2470, 6965, 5605, 4918, 6598, 7798, 6522, 4636, 6411, 336, 2844, 4364, 7932, 5419, 3035, 564, 7604, 5304, 2795, 103, 6959, 7214, 6736, 4015, 864, 3551, 2967, 5206, 6788, 3766, 5794, 1253, 3567, 1442, 6011, 4274, 5328, 2212, 5045, 6620, 4408, 6345, 3960, 3808, 3568, 6764, 5988, 4853, 2198, 2640, 6977, 2011, 6805, 709, 2284, 3692, 1997, 6147, 7276, 4668, 4999, 5473, 7693, 2755, 235, 7304, 6971, 2662, 1489, 3994, 1737, 2906, 6535, 2116, 2788, 2290, 4883, 7469, 2263, 4553, 83, 4232, 1565, 7835, 1977, 5898, 7496, 7857, 4548, 6210, 1968, 7085, 3900, 5291, 5831, 4020, 3671, 6495, 141, 762, 2410, 1815, 7177, 5667, 1993, 2508, 5439, 4764, 3023, 7344, 4534, 4349, 2816, 3485, 6110, 7715, 2709, 2882, 5757, 3717, 2219, 2511, 7905, 1888, 988, 5908, 1577, 7961, 979, 6085, 4389, 6243, 5653, 1516, 7732, 1772, 6050, 3966, 2265, 5935, 4829, 6227, 4297, 4888, 2318, 823, 6820, 1590, 2426, 1863, 2956, 7483, 2476, 115, 5800, 7170, 1036, 2247, 372, 446, 4533, 2393, 5713, 7118, 5225, 6166, 4021, 840, 100, 4702, 2329, 3845, 3921, 3608, 2791, 1510, 420, 2068, 3913, 934, 6734, 535, 3282, 4028, 606, 4726, 5156, 5623, 439, 1242, 1222, 6644, 4610, 7359, 697, 2033, 970, 4571, 6262, 3409, 7780, 7002, 6477, 8021, 1848, 6354, 4280, 7205, 3690, 3626, 2435, 4821, 3512, 2501, 4658, 5087, 493, 4994, 6062, 812, 6248, 1702, 5124, 7957, 2167, 5410, 665, 1286, 1964, 1423, 4521, 614, 1282, 21, 3346, 6870, 5647, 4864, 3849, 2385, 267, 1896, 2360, 5791, 2316, 5758, 3719, 583, 7561, 1912, 6854, 6453, 6481, 5120, 4831, 5416, 6585, 1620, 7291, 940, 4461, 1841, 5305, 1220, 2176, 6771, 1165, 7987, 488, 1359, 6493, 7843, 7650, 6135, 6752, 7365, 2364, 3597, 1018, 3839, 5641, 2491, 5732, 3297, 2230, 4099, 4423, 4045, 3586, 658, 4899, 7509, 3539, 6017, 7029, 8037, 2051, 211, 748, 7782, 5523, 4712, 4809, 169, 6264, 5507, 6729, 2207, 7763, 7128, 6251, 6195, 1435, 3854, 4251, 5337, 7348, 8060, 1486, 4795, 7185, 5200, 6670, 6895, 747, 3850, 2850, 7380, 2730, 2630, 5489, 856, 1317, 2701, 7479, 5682, 4058, 2361, 5427, 3280, 6664, 7830, 4506, 300, 3725, 721, 2576, 2067, 2648, 949, 7079, 3311, 4215, 9, 5387, 4444, 3784, 3385, 444, 1536, 4247, 2963, 5100, 6196, 4083, 5123, 3621, 6225, 422, 7562, 7992, 4499, 1073, 2359, 7593, 5720, 3970, 7280, 236, 5161, 4987, 6547, 1960, 5814, 1297, 2545, 4512, 112, 4524, 3342, 763, 7325, 929, 3780, 962, 5306, 1261, 4082, 5870, 2390, 4168, 5778, 2239, 3403, 3952, 3868, 1996, 3741, 4515, 1184, 3142, 1561, 4910, 4163, 6113, 1118, 571, 7000, 6329, 6471, 3399, 2784, 6452, 4159, 2188, 6845, 2317, 5947, 2445, 4808, 4750, 7450, 4011, 1217, 3658, 4412, 3967, 2827, 2723, 7710, 6249, 4451, 3090, 7313, 2636, 1545, 6916, 1956, 4684, 7229, 1913, 6353, 3365, 357, 2606, 6100, 7759, 5777, 3123, 3162, 5436, 6491, 5341, 1246, 4057, 303, 6915, 4114, 4834, 2719, 822, 3606, 816, 4308, 3743, 125, 5918, 1180, 3358, 7352, 1264, 612, 3846, 6402, 7820, 2773, 5105, 5674, 3256, 7804, 657, 2691, 5524, 4371, 8003, 2594, 7245, 5887, 6221, 3997, 4432, 294, 5058, 560, 1923, 5170, 5606, 2354, 6929, 6117, 740, 3555, 7741, 6228, 5191, 5766, 6933, 3634, 7703, 5685, 2453, 7183, 376, 2657, 7072, 459, 2403, 2936, 3070, 3528, 1192, 2000, 7274, 3375, 7071, 5585, 6492, 1475, 1392, 1434, 646, 4992, 7467, 5076, 5596, 1972, 4076, 4777, 1172, 1902, 3777, 6914, 2080, 3764, 2091, 5462, 7612, 3811, 2356, 5550, 4477, 1294, 605, 3618, 2830, 4813, 2450, 7868, 3475, 5655, 2048, 3742, 2474, 7323, 3151, 3958, 7643, 1943, 3124, 4685, 8040, 4708, 2423, 5728, 2418, 179, 5392, 7622, 2248, 66, 6544, 5618, 6375, 6623, 401, 4967, 6104, 4069, 2344, 6873, 7790, 4973, 2886, 1794, 5215, 7411, 5086, 2053, 5552, 6189, 5905, 6287, 5400, 5577, 1120, 5146, 795, 7917, 5294, 322, 2530, 3611, 273, 4747, 7998, 5999, 1076, 738, 2417, 2676, 7194, 6675, 1438, 1644, 1082, 6443, 7136, 2997, 4348, 4110, 2232, 1347, 2105, 3947, 7470, 6115, 2774, 943, 3836, 7529, 1153, 7127, 7123, 5540, 7176, 6586, 3255, 4565, 2996, 739, 3232, 114, 7575, 4395, 1012, 6898, 3019, 7327, 7026, 6773, 2147, 7933, 3121, 5230, 3043, 887, 5528, 1915, 3862, 205, 5075, 4599, 2687, 4997, 1813, 6816, 517, 5205, 3803, 5743, 2475, 5318, 3344, 955, 1145, 371, 304, 2493, 4035, 951, 796, 6574, 4403, 7160, 3183, 7885, 3039, 5492, 6087, 5705, 4425, 3433, 4811, 6080, 5952, 1265, 7223, 5365, 811, 4008, 5043, 3343, 2291, 268, 5654, 7064, 1779, 3632, 3642, 1934, 2971, 813, 5617, 3009, 4460, 5282, 2938, 7740, 3261, 2260, 1554, 1000, 6413, 750, 5429, 7008, 5256, 8095, 4891, 174, 7713, 7543, 1995, 1031, 4625, 1681, 7872, 6268, 4540, 1697, 4803, 1769, 1908, 7573, 7301, 4882, 23, 7250, 1185, 1064, 6368, 6659, 1429, 900, 5415, 6781, 1079, 121, 2934, 7774, 4823, 2652, 129, 1427, 2173, 429, 1038, 6076, 3448, 4309, 931, 6108, 7911, 3901, 3672, 7928, 4204, 4863, 893, 3702, 4127, 1814, 5038, 8034, 5957, 7339, 4271, 7874, 3752, 5270, 255, 498, 3923, 3290, 3492, 5620, 884, 4016, 5835, 3633, 602, 661, 2638, 4983, 1215, 538, 1033, 2252, 5114, 5186, 6759, 4492, 5833, 2663, 3120, 4893, 4346, 2415, 6373, 4141, 4959, 3524, 6859, 5748, 7115, 6599, 5332, 4516, 6468, 1761, 3523, 3699, 1871, 3389, 2776, 3715, 3266, 3407, 5976, 778, 2560, 3496, 6218, 5448, 2088, 3066, 1250, 7402, 3885, 549, 6754, 699, 6688, 3537, 791, 6099, 6052, 3052, 1065, 4557, 491, 4804, 4600, 4601, 2700, 5488, 1001, 6626, 2896, 5450, 6146, 3464, 5906, 421, 6520, 7075, 2559, 2880, 7578, 4734, 4156, 1742, 1267, 5379, 3950, 1837, 886, 2868, 4556, 3011, 941, 7446, 4703, 1852, 3515, 4595, 5093, 5026, 5529, 5264, 4560, 215, 7866, 5388, 5680, 2190, 6557, 1477, 2238, 5756, 2531, 2783, 2875, 50, 6760, 1173, 4639, 5384, 3283, 570, 1162, 6070, 6638, 251, 751, 6112, 4345, 1762, 3081, 3439, 6240, 2792, 7657, 3031, 2552, 5911, 6555, 4649, 4884, 695, 430, 1274, 6126, 5061, 407, 5521, 668, 2229, 3629, 7679, 3473, 7431, 6876, 6712, 3392, 2237, 1765, 4197, 932, 7474, 7356, 908, 2320, 5555, 5558, 4858, 7677, 4316, 5465, 2526, 6920, 7189, 3237, 4909, 448, 62, 1674, 2469, 1730, 1124, 2093, 2371, 6396, 7644, 63, 4074, 3527, 1439, 1058, 3114, 4362, 5764, 6478, 4098, 4577, 5472, 2901, 590, 3252, 346, 3573, 153, 7049, 637, 2564, 7093, 3516, 4697, 3313, 5812, 5244]


class PointCloudEmbeddingSequenceDataset(BaseDataset):
    
    def __init__(self, root, split, use_normals=False, verbose=False):
        super().__init__(root, split, use_normals=use_normals, verbose=verbose)
        
    def __getitem__(self, idx):
        point_cloud = o3d.io.read_point_cloud(self.pc[idx])
        if self.use_normals: 
            normals = np.asarray(point_cloud.normals)
            point_cloud = torch.tensor(np.hstack((point_cloud.points, normals)), dtype = torch.float32) # [N, C + 3]
        else:
            point_cloud = torch.tensor(np.asarray(point_cloud.points), dtype = torch.float32) # [N, C]
        sample_idx = random.sample(list(range(point_cloud.shape[0])), N_POINTS)
        # point_cloud = point_cloud[sample_idx] # IGNORE_INDICES
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

class PCExtrusionSegmentationDataset(BaseDataset):
    
    def __init__(self, root, split, use_normals=False, verbose=False):
        super().__init__(root, split, use_normals=use_normals, verbose=verbose)
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
        sample_idx = random.sample(list(range(point_cloud.shape[0])), N_POINTS)
        point_cloud = point_cloud[sample_idx] # IGNORE_INDICES
        
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
    def __init__(self, root, split, cfg, verbose=False):
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
        point_cloud = self.adjust_pointcloud_to_fixed_size(point_cloud, N_POINTS)
        point_cloud = torch.tensor(point_cloud, dtype = torch.float32) # [N, C]

        with h5py.File(self.labels[idx], "r") as f:
            extr_id = f['extrusion_id'][()]
            sequence = f['sequence'][:]

        extr_id = torch.tensor(extr_id, dtype=torch.long)
        sequence = torch.tensor(sequence, dtype=torch.long)

        pad_len = self.cfg.max_total_len - sequence.shape[0]
        cad_vec = np.concatenate([sequence, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
        
        return {'pc': point_cloud,
               'extrusion_id': extr_id,
               'sequence': cad_vec}

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

    def get_pc_path(self, idx):
        return self.pc[idx] 

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
import os
import sys
import argparse
import importlib
import time
import csv
import h5py

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
import numpy as np

from code.dataset import PointCloudEmbeddingDataset
from code.metrics import RegressionRunningScore
from code.utils import Logger

from models.DeepCAD.config.configAE import ConfigAE
from models.DeepCAD.trainer.trainerAE import TrainerAE


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser(
        'Test the ability of PointNet++ to encode point clouds into the same latent space as DeepCAD '
        'does with CAD Sequences.'
    )
    parser.add_argument('--data_root', 
                        type=str, 
                        default='data', 
                        help='data directory relative to root directory')
    parser.add_argument('--batch_size', type=int, default=48, help='batch size')
    parser.add_argument('--verbose', action='store_true', default=False, help='output per batch metrics')
    parser.add_argument('--exp_name', type=str, required=True, help='name of the experiment')
    parser.add_argument('--model_path', type=str, required=True, help='path to the trained model')
    parser.add_argument('--phase', type=str, choices=['train', 'validation', 'test'], 
                        default='train', help='which set to infer')
    return parser.parse_args()

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def main(args):
    print("### START ###\n")
    DATA_DIR = os.path.abspath(args.data_root)
    root_dir = os.path.dirname(DATA_DIR)
    model_path = os.path.abspath(args.model_path)
    latent_dim = 256

    # Load pretrained PointNet++
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert(os.path.exists(model_path)), f"Model path {model_path} does not exist."
    saved_model = torch.load(model_path, map_location=torch.device(device), weights_only=True)
    model_dir = os.path.dirname(os.path.abspath(model_path))

    # Create save directory for results
    results_dir = os.path.join(model_dir, "results", args.exp_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    h5_file = os.path.join(results_dir, 'z.h5')

    # Logging
    monitor = Logger(results_dir, 'pipeline')
    monitor.log_and_print(f"PointNet++: {model_path}\n")
    config = saved_model['config']
    monitor.log_and_print("### PN++ PARAMETERS ###")
    for key, value in config.items():
        monitor.log_and_print(f"{key}: {value}")

    # Create PointNet++
    sys.path.append(os.path.join(root_dir, 'models','Pointnet_Pointnet2_pytorch', 'models'))
    model = importlib.import_module('pointnet2_cls_ssg')
    classifier = model.get_model(latent_dim, normal_channel=False)
    criterion = model.get_loss_mse()
    classifier.apply(inplace_relu)

    # Load PointNet++ state dict
    state_dict = saved_model['model_state_dict']
    # Check if model has been saved wrapped in nn.DataParallel
    if 'module.' in next(iter(state_dict)):
        monitor.log_and_print("Model was saved wrapped in nn.DataParallel.\nRemoving 'module.' from state dict.")
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}#
    classifier.load_state_dict(state_dict)
    monitor.log_and_print(f'Loaded state dict from {model_path}.')

    # Cuda
    monitor.log_and_print(f"Using device: {device}\n")
    monitor.log_and_print(f"Number of devices: {torch.cuda.device_count()}")
    batch_size = args.batch_size
    if torch.cuda.device_count() > 1:
        monitor.log_and_print(f"Using {torch.cuda.device_count()} GPUs.\n")#
        classifier = nn.DataParallel(classifier)
        batch_size *= torch.cuda.device_count()
        monitor.log_and_print(f"Batch size multiplied with number of devices {torch.cuda.device_count()}, current batch size: {batch_size}")
    classifier = classifier.to(device)
    criterion = criterion.to(device)

    # Data
    num_workers = 0 if device.type == 'cpu' else 8
    dataset = PointCloudEmbeddingDataset(DATA_DIR, args.phase)
    dataloader = DataLoader(dataset, batch_size = batch_size, num_workers = num_workers, shuffle = False)
    num_samples = len(dataset) # FIXME When infering from a directory adapt this dynamically
    
    # Inference
    classifier.eval()
    monitor.log_and_print("### START INFERENCE ###")

    with h5py.File(h5_file, 'w') as hf:
        z_dataset = hf.create_dataset('latent_rep', 
                                shape=(num_samples, latent_dim), 
                                dtype=np.float32)
        start_idx = 0

        with torch.no_grad():
            for i, (pc, latent_rep) in enumerate(dataloader):
                pc, latent_rep = pc.to(device), latent_rep.to(device)
                pc = pc.transpose(2, 1)
                pred, _ = classifier(pc)
                loss = criterion(pred,latent_rep)

                if args.verbose:
                    print(f"Batch {i + 1}/{len(dataloader)}: "
                        f"Loss: {loss.cpu().item():.8f}", flush=True)
                
                end_idx = start_idx + batch_size
                z_dataset[start_idx:end_idx] = pred.cpu().numpy()
                start_idx = end_idx

                if i == 2:
                    break
    monitor.log_and_print("### DONE PC->z ###")

    # Load DeepCAD model
    cfg = ConfigAE('test')
    tr_agent = TrainerAE(cfg)
    


    

if __name__ == '__main__':
    args = parse_args()
    main(args)


# TODO  Script should be able to infer train/val/test set
#       But also one should be able to specify a dir with pointclouds inside
#       Maybe there needs to be some differentiation if CAD sequence targets
#       are available or not

#       When using the DeepCAD model, make sure to check the latent representations
#       if they are non zero
#       Integrate command line arguments in configAE.py

# TODO  README

# TODO  CHANGELOG DeepCAD
#       Adapted imports to new repo structure
#       Manage command line arguments in configAE.py
#       Removed tensorboard dependency
#       Fixed abstract method by including ABC dependency
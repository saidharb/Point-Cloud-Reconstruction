import os
import sys
import argparse
import importlib
import time
import csv

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
import wandb

from code.dataset import PointCloudEmbeddingDataset
from code.metrics import RegressionRunningScore
from code.utils import Logger

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
    parser.add_argument('--exp_name', type=str, required=True, help='name of the experiment')
    parser.add_argument('--model_path', type=str, required=True, help='path to the trained model')
    return parser.parse_args()

def main(args):
    print("### START ###\n")
    DATA_DIR = os.path.abspath(args.data_root)

    # Load pretrained PointNet++
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert(os.path.exists(args.model_path)), f"Model path {args.model_path} does not exist."
    saved_model = torch.load(args.model_path, map_location=torch.device(device), weights_only=True)
    model_dir = os.path.dirname(os.path.abspath(args.model_path))

    # Create save directory for results
    results_dir = os.path.join(model_dir, "results", args.exp_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Logging
    monitor = Logger(results_dir, 'pipeline')
    monitor.log_and_print("### NEW INFERENCE ###\n")
    monitor.log_and_print(f"PointNet++: {os.path.abspath(args.model_path)}\n")
    config = saved_model['config']
    monitor.log_and_print("### PointNet++ Parameters ###\n")
    for key, value in config.items():
        monitor.log_and_print(f"{key}: {value}")

if __name__ == '__main__':
    args = parse_args()
    main(args)
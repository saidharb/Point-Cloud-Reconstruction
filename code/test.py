import os
import sys
import argparse
import importlib
from datetime import datetime
import time
import csv

import torch
from torch.utils.data import DataLoader
import wandb

from dataset import PointCloudEmbeddingDataset
from models.Pointnet_Pointnet2_pytorch import provider
from metrics import RegressionRunningScore
from utils import SaveBestModel, EarlyStopping, Logger

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
    parser.add_argument('--model_path', type=str, required=True, help='path to the trained model')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size')
    return parser.parse_args()

def main(args):

    # Find data directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if os.getcwd() != root_dir:
        DATA_DIR = os.path.join(root_dir, args.data_root)
    else:
        DATA_DIR = os.path.abspath(args.data_root)

    # Find model directory
    model_dir = os.path.dirname(os.path.abspath(args.model_path))
    print(model_dir)

    # Logging
    monitor = Logger(model_dir, 'test')

    date_and_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    monitor.log_and_print(f"### TEST STARTED ###"
          f"\n{date_and_time}\n")


if __name__ == '__main__':
    args = parse_args()
    main(args)

# TODO: - print model configuration when testing -> save in logfile
# - create csv file test_metrics with test metrics in respective directory

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
import torch.nn as nn
import torch.nn.functional as F

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

class get_loss_mse(nn.Module):
    def __init__(self):
        super(get_loss_mse, self).__init__()

    def forward(self, pred, target):
        total_loss = F.mse_loss(pred, target)

        return total_loss

def main(args):

    print("### TEST STARTED ###")

    # Find data directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if os.getcwd() != root_dir:
        DATA_DIR = os.path.join(root_dir, args.data_root)
    else:
        DATA_DIR = os.path.abspath(args.data_root)

    # Find model directory
    assert(os.path.exists(args.model_path)), f"Model path {args.model_path} does not exist."
    model_dir = os.path.dirname(os.path.abspath(args.model_path))

    # Logging
    monitor = Logger(model_dir, 'test')

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = torch.load(args.model_path, map_location=torch.device(device), weights_only=True)
    
    config = classifier['config']
    monitor.log_and_print("### Parameters ###\n")
    for key, value in config.items():
        monitor.log_and_print(f"{key}: {value}")

    criterion = get_loss_mse()
    # next: load model, move best.pt somewhere where it makes more sense and I can see in VSCode


if __name__ == '__main__':
    args = parse_args()
    main(args)

# TODO: - print model configuration when testing -> save in logfile
# - create csv file test_metrics with test metrics in respective directory

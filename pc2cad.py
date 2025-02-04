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
    return parser.parse_args()

def main(args):
    print("### START ###\n")
    DATA_DIR = os.path.abspath(args.data_root)
    print(DATA_DIR)

if __name__ == '__main__':
    args = parse_args()
    main(args)
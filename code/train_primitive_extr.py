import os
import sys
import argparse
import importlib
from datetime import datetime
import time
import csv

import torch
from torch.utils.data import DataLoader
import torch.nn as nn 
import wandb
import numpy as np

from code.dataset import PCExtrusionSegmentationDataset
from code.metrics import ClassificationRunningScore
from code.utils import EarlyStoppingExtrusionSeg, Logger, SaveBestModelExtrusionSeg, LearningRateStepSchedulerExtrSeg
from code.LRSchedulers import CosineAnnealWarmRestart, StepLR

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser(
        'Train PointNet++ to segment point clouds into their extrusions.'
    )
    parser.add_argument('--data_root', 
                    type=str, 
                    default='data', 
                    help='data directory relative to root directory')
    parser.add_argument('--output_dir', type=str, required=True, help='name of output directory in trained_models')
    return parser.parse_args()

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def main(args):
    date_and_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    start_time = time.time()

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if os.getcwd() != root_dir:
        DATA_DIR = os.path.join(root_dir, args.data_root)
    else:
        DATA_DIR = os.path.abspath(args.data_root)

        # Find experiment directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.abspath(os.path.join(script_dir, "..", "models", "trained_models", args.output_dir))
    continue_training = False
    if not os.path.exists(save_dir):
        print(f"### NEW TRAINING STARTED ###"
          f"\n{date_and_time}\n", flush=True)
        os.makedirs(save_dir)
        print(f"Created model save directory at: {os.path.abspath(save_dir)}\n", flush=True)
    else:
        print(f"### CONTINUING TRAINING ###"
          f"\n{date_and_time}\n", flush=True)
        continue_training = True
        
    # Logging
    monitor = Logger(save_dir, 'train')
    if continue_training:
        monitor.log_and_print("### CONTINUING TRAINING ###")
    else:
        monitor.log_and_print("### NEW TRAINING STARTED ###")

if __name__ == '__main__':
    args = parse_args()
    main(args)
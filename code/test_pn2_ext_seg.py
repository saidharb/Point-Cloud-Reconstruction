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

from dataset import PCExtrusionSegmentationDataset
from metrics import ClassificationRunningScore
from utils import EarlyStoppingExtrusionSeg, Logger, LearningRateStepScheduler, SaveBestModelExtrusionSeg
from LRSchedulers import CosineAnnealWarmRestart, StepLR

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser(
        'Test the ability of PointNet++ to segment point clouds into their extrusions.'
    )
    parser.add_argument('--data_root', 
                        type=str, 
                        default='data', 
                        help='data directory relative to root directory')
    parser.add_argument('--verbose', action='store_true', default=False, help='output per batch metrics')
    parser.add_argument('--model_path', type=str, required=True, help='path to the trained model')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size')
    parser.add_argument('--gpu', action='store_true', default=False, help="Use multiple GPU's for testing.")
    return parser.parse_args()

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def main(args):
    
    print("### TEST STARTED ###\n")
    start_time = time.time()

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
    saved_model = torch.load(args.model_path, map_location=torch.device(device), weights_only=True)
    
    config = saved_model['config']
    monitor.log_and_print("### Parameters ###\n")
    for key, value in config.items():
        monitor.log_and_print(f"{key}: {value}")

    sys.path.append(os.path.join(root_dir, 'models','Pointnet_Pointnet2_pytorch', 'models'))
    model_name = 'pointnet2_sem_seg_msg'
    model = importlib.import_module(model_name)

    num_classes = 10 # max number of extrusions in dataset
    classifier = model.get_model(num_classes)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    state_dict = saved_model['model_state_dict']
    classifier.load_state_dict(state_dict)
    monitor.log_and_print(f'\nLoaded state dict from {os.path.abspath(args.model_path)}.')

    monitor.log_and_print(f"Using device: {device}\n")
    monitor.log_and_print(f"Number of devices: {torch.cuda.device_count()}")#
    batch_size = args.batch_size
    if args.gpu and torch.cuda.device_count() > 1:
        monitor.log_and_print(f"Using {torch.cuda.device_count()} GPUs.\n")#
        classifier = nn.DataParallel(classifier)
        batch_size *= torch.cuda.device_count()
        monitor.log_and_print(f"Batch size multiplied with number of devices {torch.cuda.device_count()}, current batch size: {batch_size}")
    classifier = classifier.to(device)
    criterion = criterion.to(device)

    # Load data
    num_workers = 0 if device.type == 'cpu' else 8
    print("Num. workers: ", num_workers, flush=True)
    test_dataset = PCExtrusionSegmentationDataset(DATA_DIR, 'test', use_normals=False, verbose=True)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, num_workers = num_workers, shuffle = False) # multiprocessing_context=multiprocessing.get_context("spawn")
    monitor.log(f"Test set: {len(test_dataloader)}")

    scores_test = ClassificationRunningScore(num_classes, model_dir, cont=False, phase='test')
    monitor.log_and_print("### Test starts ###\n")

    



if __name__ == '__main__':
    args = parse_args()
    main(args)
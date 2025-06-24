import os
import sys
import argparse
import importlib
from datetime import datetime
import time
import csv
import multiprocessing

import torch
from torch.utils.data import DataLoader
import torch.nn as nn 
import wandb

from dataset import PointCloudEmbeddingDataset
from models.Pointnet_Pointnet2_pytorch import provider
from metrics import RegressionRunningScore
from utils import SaveBestModel, EarlyStopping, Logger, LearningRateStepScheduler
from LRSchedulers import CosineAnnealWarmRestart, StepLR

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
    parser.add_argument('--batch_size', type=int, default=24, help='batch size')
    parser.add_argument('--gpu', action='store_true', default=False, 
                        help="Use multiple GPU's for training.")
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
        print(f"Created model save directory at: {os.path.abspath(save_dir)}\n")
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

    # Print parameters
    monitor.log_and_print("### Parameters ###\n")
    for key, value in vars(args).items():
        monitor.log_and_print(f"{key}: {value}")
    print("\n--- DONE ---\n", flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
    sys.path.append(os.path.join(root_dir, 'models','Pointnet_Pointnet2_pytorch', 'models'))
    model_name = 'pointnet2_part_seg_msg'
    model = importlib.import_module(model_name)

    num_classes = 10 # max number of extrusions in dataset
    classifier = model.get_model(num_classes, normal_channel=False)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    if continue_training:
        monitor.log_and_print(f"### Load pretrained {model_name} model ###\n")
        model_path = os.path.join(save_dir, 'last.pth')
        saved_model = torch.load(model_path, map_location=torch.device(device), weights_only=True)
        state_dict = saved_model['model_state_dict']
        first_epoch = saved_model['config']['final_epoch'] 
        classifier.load_state_dict(state_dict)
        classifier = classifier.to(device)
        criterion = criterion.to(device)
        monitor.log_and_print(f'\nLoaded state dict from {model_path}.')
    else:
        monitor.log_and_print(f"### Load new {model_name} model ###\n")

    ## Cuda
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
    print("--- DONE ---\n", flush=True)


if __name__ == '__main__':
    args = parse_args()
    main(args)
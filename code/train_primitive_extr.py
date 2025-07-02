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

from code.dataset import PCExtrusionSequenceDataset
from code.metrics import ClassificationRunningScore
from code.utils import EarlyStoppingExtrusionSeg, Logger, SaveBestModelExtrusionSeg, LearningRateStepSchedulerExtrSeg
from code.LRSchedulers import CosineAnnealWarmRestart, StepLR
from code.pn2_deepcad import Config
from models.DeepCAD.trainer.loss import CADLoss

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
    parser.add_argument('--learning_rate', type=float, default=0.001, help="initial learning rate")
    parser.add_argument('--max_epochs', type=int, default=50, help='maximum number of epochs')
    parser.add_argument('--early_stopping', 
                        type=int, 
                        default=20, 
                        help="abort training after this amount of epochs with no validation loss decrease")
    parser.add_argument('--save_interval', type=int, default=20, help='save interval for models')
    parser.add_argument('--lr_type', type=str, choices=['step', 'cosine', 'step_adv'], default='step', 
                        help="Learning rate type: step for a simple step learning rate scheduler, "
                        "step_adv for reducing learning rate on val_loss plateau or cosine for cosine "
                        "annealing with warm restarts")
    parser.add_argument('--wandb', action='store_true', default=False, help='enable WandB tracking')
    parser.add_argument('--name', type=str, default="test_run", help="name of WandB run")
    parser.add_argument('--lr_patience', type=int, default=15, help="patience in epochs for learning rate decay")
    parser.add_argument('--verbose', action='store_true', default=False, help='output per batch metrics')
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

    # Print parameters
    monitor.log_and_print("### Parameters ###\n")
    for key, value in vars(args).items():
        monitor.log_and_print(f"{key}: {value}")
    print("\n--- DONE ---\n", flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg = Config()
    model_name = "pn2_deepcad"
    model = importlib.import_module(model_name)
    classifier = model.get_pn2_deepcad_model(cfg, normal_channel=False)
    criterion = CADLoss(cfg)
    classifier.apply(inplace_relu)

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

    config = {
        'learning_rate': args.learning_rate,
        'batch_size': batch_size,
        'max_epochs': args.max_epochs,
        'optimizer': 'Adam',
        'model_type': model_name,
        'save_interval': args.save_interval,
        'early_stopping': args.early_stopping,
        'start_time': date_and_time,
        'lr_type': args.lr_type,
        'gpu': args.gpu,
    }
    if continue_training:
        model_path = os.path.join(save_dir, 'last.pth')
        saved_model = torch.load(model_path, map_location=torch.device(device), weights_only=True)
        config = saved_model['config']

    if args.wandb:
        print("### WANDB ###\n", flush=True)
        if os.getenv("WANDB_API_KEY"):
            print("Logging into WandB...\n", flush=True)
            wandb.login(key=os.getenv("WANDB_API_KEY"))

            run_id_file = os.path.join(save_dir, "wandb_run_id.txt")
            if os.path.exists(run_id_file):
                with open(run_id_file, "r") as f:
                    run_id = f.read().strip()
                print(f"Resuming WandB run with ID: {run_id}\n", flush=True)
                wandb.init(project='Master Thesis',
                        id=run_id,
                        resume="allow",
                        config=config)
            else:
                run = wandb.init(project='Master Thesis',
                                name=args.name,
                                config=config)
                run_id = run.id
                with open(run_id_file, "w") as f:
                    f.write(run_id)
                print(f"New WandB run started with ID: {run_id}\n", flush=True)
        else:
            print("No WandB API key provided, WandB is disabled.\n", flush=True)

    # Load data
    num_workers = 0 if device.type == 'cpu' else 8
    print("Num. workers: ", num_workers, flush=True)
    train_dataset = PCExtrusionSequenceDataset(DATA_DIR, 'train', verbose=True)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, num_workers = num_workers, shuffle = True) # multiprocessing_context=multiprocessing.get_context("spawn")
    val_dataset = PCExtrusionSequenceDataset(DATA_DIR, 'validation', verbose=True)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, num_workers = num_workers, shuffle = False) # multiprocessing_context=multiprocessing.get_context("spawn")
    monitor.log(f"Train Dataset: {len(train_dataset)}, Validation Dataset: {len(val_dataset)}")
    monitor.log(f"Train Dataloader: {len(train_dataloader)}, Validation Dataloader: {len(val_dataloader)}")

if __name__ == '__main__':
    args = parse_args()
    main(args)
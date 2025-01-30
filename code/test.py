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

from dataset import PointCloudEmbeddingDataset
from metrics import RegressionRunningScore
from utils import Logger

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
    parser.add_argument('--verbose', action='store_true', default=False, help='output per batch metrics')
    parser.add_argument('--model_path', type=str, required=True, help='path to the trained model')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size')
    parser.add_argument('--wandb', action='store_true', default=False, help='enable WandB tracking')
    return parser.parse_args()
    
def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def save_test_metrics(*lists, save_path = ""):
    headers = ['mse', 'rmse', 'mae']
    with open(os.path.join(os.path.dirname(save_path), 'test_metrics_' + os.path.basename(save_path) + '.csv'), mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for row in zip(*lists):
            writer.writerow(row)

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

    # Create model
    sys.path.append(os.path.join(root_dir, 'models','Pointnet_Pointnet2_pytorch', 'models'))
    model = importlib.import_module('pointnet2_cls_ssg')
    classifier = model.get_model(256, normal_channel=False)
    criterion = model.get_loss_mse()
    classifier.apply(inplace_relu)

    # Load saved model
    state_dict = saved_model['model_state_dict']
    # Check if model has been saved wrapped in nn.DataParallel (only in fourth official run),
    # after that I fixed it
    if 'module.' in next(iter(state_dict)):#
        monitor.log_and_print("Model was saved wrapped in nn.DataParallel.\nRemoving 'module.' from state dict.")
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}#
    classifier.load_state_dict(state_dict)
    monitor.log_and_print(f'\nLoaded state dict from {os.path.abspath(args.model_path)}.')

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

    if args.wandb:
        monitor.log_and_print("### WANDB ###\n")
        if os.getenv("WANDB_API_KEY"):
            monitor.log_and_print("Logging into WandB...\n")
            wandb.login(key=os.getenv("WANDB_API_KEY"))

            run_id_file = os.path.join(model_dir, "wandb_run_id.txt")
            if os.path.exists(run_id_file):
                with open(run_id_file, "r") as f:
                    run_id = f.read().strip()
                monitor.log_and_print(f"Resuming WandB run with ID: {run_id}\n")
                wandb.init(project='Master Thesis',
                        id=run_id,
                        resume="allow",
                        config=config)
            table = wandb.Table(columns=["test_mse", "test_rmse", "test_mae"])
        else:
            monitor.log_and_print("No WandB API key provided, WandB is disabled.\n")
    

    # Data
    num_workers = 0 if device.type == 'cpu' else 8
    test_dataset = PointCloudEmbeddingDataset(DATA_DIR, 'test')
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, num_workers = num_workers, shuffle = False)
    monitor.log(f"Length test set: {len(test_dataloader)}\n")
    monitor.log_and_print(f"Using device: {device}\n")

    # Metrics
    scores_test = RegressionRunningScore(len(test_dataloader), model_dir)

    # Testing
    classifier.eval()
    with torch.no_grad():
        for i, (pc, latent_rep) in enumerate(test_dataloader):
            pc, latent_rep = pc.to(device), latent_rep.to(device)
            pc = pc.transpose(2, 1)
            pred, _ = classifier(pc)
            loss_test = criterion(pred,latent_rep)

            scores_test.update(loss_test.detach(), pred, latent_rep)

            if args.verbose:
                print(f"Test Batch {i + 1}/{len(test_dataloader)}: "
                    f"Loss: {loss_test.cpu().item():.8f} --- "
                    f"RMSE: {scores_test.get_batch_rmse(loss_test):.8f} --- "
                    f"MAE: {scores_test.get_batch_mae(pred, latent_rep):.8f}", flush=True)

            # if i == 10:
            #     break

    scores_test.epoch_finished()
    monitor.log_and_print("### RESULTS ###")
    monitor.log_and_print(f"Test --- "
                          f"Avg. Loss/MSE: {scores_test.get_epoch_mse(0):.8f} --- "
                          f"Avg. RMSE: {scores_test.get_epoch_rmse(0):.8f} --- "
                          f"Avg. MAE: {scores_test.get_epoch_mae(0):.8f}")
    if args.wandb:
        table.add_data(scores_test.get_epoch_mse(0), scores_test.get_epoch_rmse(0), scores_test.get_epoch_mae(0))
        wandb.log({"test_results": table})
        wandb.finish()
    save_test_metrics(*scores_test.get_metrics_list(), 
                     save_path = args.model_path)
    duration = round((time.time() - start_time) / 60.0, 2)
    monitor.log_and_print(f"Test time in minutes: {duration}")



if __name__ == '__main__':
    args = parse_args()
    main(args)

# TODO
# eventuell noch in test_metrics.csv die beste trainings epoche metrics einfügen (wandb)
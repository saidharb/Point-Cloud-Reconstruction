import os
import sys
import argparse
import importlib
import time
import csv

import torch
from torch.utils.data import DataLoader

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
    return parser.parse_args()
    
def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def save_test_metrics(*lists, save_path = ""):
    headers = ['mse', 'rmse', 'mae']
    with open(os.path.join(save_path, 'test_metrics.csv'), mode='w', newline='', encoding='utf-8') as file:
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
    classifier.load_state_dict(state_dict)
    classifier = classifier.to(device)
    criterion = criterion.to(device)
    monitor.log_and_print(f'\nLoaded state dict from {os.path.abspath(args.model_path)}.')

    # Data
    num_workers = 0 if device.type == 'cpu' else 4
    test_dataset = PointCloudEmbeddingDataset(DATA_DIR, 'test')
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, num_workers = num_workers, shuffle = False)
    monitor.log(f"Length test set: {len(test_dataloader)}\n")

    # Metrics
    scores_test = RegressionRunningScore(len(test_dataloader))

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
    save_test_metrics(*scores_test.get_metrics_list(), 
                     save_path = model_dir)
    duration = round((time.time() - start_time) / 60.0, 2)
    monitor.log_and_print(f"Test time in minutes: {duration}")



if __name__ == '__main__':
    args = parse_args()
    main(args)

# TODO: - print model configuration when testing -> save in logfile
# - create csv file test_metrics with test metrics in respective directory

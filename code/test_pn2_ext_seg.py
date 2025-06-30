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

def save_test_metrics(*lists, save_dir):
    tp, fp, fn, class_iou, class_acc, miou, acc, mean_acc, loss, = lists
    csv_path = os.path.join(save_dir, 'test_metrics.csv')
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "mIoU", "acc", "mean_acc", "loss"
        ])
        for row in zip(
                        miou, acc, mean_acc, loss):
            writer.writerow(row)
    
    np.savez(os.path.join(save_dir, 'test_metrics.npz'),
             tp=np.array(tp), 
             fp=np.array(fp), 
             fn=np.array(fn), 
             class_iou=np.array(class_iou), 
             class_acc=np.array(class_acc))

def main(args):
    
    print("### TEST STARTED ###\n")

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

    loss_train_sum = 0.0
    classifier.eval()
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            pc = data['pc']
            label = data['label']
            pc = pc.transpose(2, 1) # [B, C, N]
            pc, label = pc.to(device), label.to(device)
            seg_pred, trans_feat = classifier(pc)

            seg_pred = seg_pred.contiguous().view(-1, num_classes)
            label = label.view(-1, 1).squeeze()

            loss_test = criterion(seg_pred, label, trans_feat, weight=None)
            scores_test.update(seg_pred, label)

            loss_train_sum += loss_test
            if args.verbose:
                print(f"Batch {i}/{len(test_dataloader) - 1}: "
                      f"Loss/Sum: {loss_test.item():<.4f}/{loss_train_sum.item():<.4f} | "
                      f"mAcc.: {scores_test.get_mean_class_accuracy():<.4f} | "
                      f"mIoU: {scores_test.get_mIoU():<.4f} ", flush=True)

    scores_test.epoch_finished(loss_train_sum.item() / len(test_dataloader))
    monitor.log_and_print("### RESULTS ###")
    for a in scores_test.get_metrics_list():
        monitor.log_and_print(a)

    save_test_metrics(*scores_test.get_metrics_list(), save_dir=model_dir)


   

    



if __name__ == '__main__':
    args = parse_args()
    main(args)
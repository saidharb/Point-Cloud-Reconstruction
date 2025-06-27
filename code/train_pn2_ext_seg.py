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

def save_metrics(*lists, save_path, epoch):
    lr_history, tp_train, fp_train, fn_train, class_iou_train, class_acc_train, \
    miou_train, acc_train, mean_acc_train, loss_train, \
    tp_val, fp_val, fn_val, class_iou_val, class_acc_val, \
    miou_val, acc_val, mean_acc_val, loss_val = lists

    epoch = list(range(1, epoch + 2))  # +1 for zero-based index, +1 for last epoch

    np.savez(os.path.join(save_path, "train_metrics.npz"),
             tp = np.array(tp_train),
             fp = np.array(fp_train),
             fn = np.array(fn_train),
             class_iou = np.array(class_iou_train),
             class_acc = np.array(class_acc_train))
    
    np.savez(os.path.join(save_path, "val_metrics.npz"),
             tp = np.array(tp_val),
             fp = np.array(fp_val),
             fn = np.array(fn_val),
             class_iou = np.array(class_iou_val),
             class_acc = np.array(class_acc_val))
    
    csv_path = os.path.join(save_path, "mean_metrics.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "lr",
            "train_mIoU", "train_acc", "train_mean_acc",
            "val_mIoU", "val_acc", "val_mean_acc",
            "train_loss", "val_loss"
        ])

        for row in zip(epoch, lr_history,
                        miou_train, acc_train, mean_acc_train,
                        miou_val, acc_val, mean_acc_val,
                        loss_train, loss_val):
            writer.writerow(row)

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
    model_name = 'pointnet2_sem_seg_msg'
    model = importlib.import_module(model_name)

    num_classes = 10 # max number of extrusions in dataset
    classifier = model.get_model(num_classes)
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

    first_epoch = 0
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
        classifier.apply(weights_init)

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
    print("Num. workers: ", num_workers)
    train_dataset = PCExtrusionSegmentationDataset(DATA_DIR, 'train', use_normals=False, verbose=True)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, num_workers = num_workers, shuffle = True) # multiprocessing_context=multiprocessing.get_context("spawn")
    val_dataset = PCExtrusionSegmentationDataset(DATA_DIR, 'validation', use_normals=False, verbose=True)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, num_workers = num_workers, shuffle = False) # multiprocessing_context=multiprocessing.get_context("spawn")
    monitor.log(f"Train set: {len(train_dataloader)}, Validation set: {len(val_dataloader)}")

    ## Optimizer
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4
        )
    
    if args.lr_type == 'step_adv':
        scheduler = LearningRateStepScheduler(optimizer, 
                                              0.1, 
                                              args.lr_patience, 
                                              monitor, 
                                              save_dir, 
                                              cont=continue_training)
    elif args.lr_type == 'cosine':
        scheduler = CosineAnnealWarmRestart(optimizer, 
                                            monitor, 
                                            save_dir, 
                                            T_0=20, 
                                            T_mult=1.5, 
                                            factor = 0.8, 
                                            min_lr=1e-7, 
                                            cont=continue_training)
    elif args.lr_type == 'step':
        scheduler = StepLR(optimizer,
                           monitor,
                           save_dir,
                           args.lr_patience,
                           0.1,
                           cont=continue_training)
        
    scores_train = ClassificationRunningScore(num_classes, save_dir, continue_training, phase='train')
    scores_val = ClassificationRunningScore(num_classes, save_dir, continue_training, phase='validation')

    best_model_tracker = SaveBestModelExtrusionSeg(config, save_dir, monitor, cont = continue_training)
    early_stopping = EarlyStoppingExtrusionSeg(config, monitor, save_dir, cont = continue_training)

    # Training
    monitor.log_and_print("### Training starts ###\n")
    for epoch in range(first_epoch, args.max_epochs):
        classifier.train()
        print(f"Epoch {epoch + 1}/{args.max_epochs}", flush=True)
        epoch_start_time = time.time()
        loss_train_sum = 0

        for i, data in enumerate(train_dataloader):
            pc = data['pc']
            label = data['label']
            
            optimizer.zero_grad()
            pc = pc.transpose(2, 1) # [B, C, N]
            pc, label = pc.to(device), label.to(device)
            seg_pred, trans_feat = classifier(pc)

            seg_pred = seg_pred.contiguous().view(-1, num_classes)
            label = label.view(-1, 1).squeeze()

            loss_train = criterion(seg_pred, label, trans_feat, weight=None)

            loss_train.backward()
            optimizer.step()

            scores_train.update(seg_pred, label)

            loss_train_sum += loss_train

            if args.verbose:
                print(f"Batch {i}/{len(train_dataloader) - 1}: "
                      f"Loss/Sum: {loss_train.item():<.4f}/{loss_train_sum.item():<.4f} | "
                      f"mAcc.: {scores_train.get_mean_class_accuracy():<.4f} | "
                      f"mIoU: {scores_train.get_mIoU():<.4f} ")

            # if i == 1:
            #     break

        print(f"Train Epoch {epoch + 1}: "
                f"Avg. Loss: {loss_train_sum.item()/len(train_dataloader):<.4f} | "
                f"Acc.: {scores_train.get_accuracy():<.4f} | "
                f"mAcc.: {scores_train.get_mean_class_accuracy():<.4f} | "
                f"mIoU: {scores_train.get_mIoU():<.4f} ")
        scores_train.epoch_finished(loss_train_sum.item()/len(train_dataloader))

        # Evaluation
        classifier.eval()
        loss_val_sum = 0

        with torch.no_grad():
            for i, data in enumerate(val_dataloader):
                pc = data['pc']
                label = data['label']
                pc = pc.transpose(2, 1)
                pc, label = pc.to(device), label.to(device)
                seg_pred, trans_feat = classifier(pc)

                seg_pred = seg_pred.contiguous().view(-1, num_classes)
                label = label.view(-1, 1).squeeze()

                loss_val = criterion(seg_pred, label, trans_feat, weight=None)

                scores_val.update(seg_pred, label)

                loss_val_sum += loss_val

                if args.verbose:
                    print(f"Batch {i}/{len(val_dataloader) - 1}: "
                        f"Loss/Sum: {loss_val.item():<.4f}/{loss_val_sum.item():<.4f} | "
                        f"mAcc.: {scores_val.get_mean_class_accuracy():<.4f} | "
                        f"mIoU: {scores_val.get_mIoU():<.4f} ")

                # if i == 1:
                #     break

        print(f"Val Epoch {epoch + 1}: "
                f"Avg. Loss: {loss_val_sum.item()/len(val_dataloader):<.4f} | "
                f"Acc.: {scores_val.get_accuracy():<.4f} | "
                f"mAcc.: {scores_val.get_mean_class_accuracy():<.4f} | "
                f"mIoU: {scores_val.get_mIoU():<.4f} ")
        scores_val.epoch_finished(loss_val_sum.item()/len(val_dataloader))
        epoch_duration = (time.time() - epoch_start_time) / 60.0

        current_lr = scheduler.get_current_learning_rate()
        scheduler.update(loss_val_sum.item() / len(val_dataloader))

        if args.wandb:
            if os.getenv("WANDB_API_KEY"):
                wandb.log({'epochs': epoch, 
                        'learning_rate': current_lr,
                        'train_loss': scores_train.get_epoch_loss(epoch),
                        'train_miou': scores_train.get_epoch_miou(epoch),
                        'train_acc': scores_train.get_epoch_acc(epoch),
                        'train_mean_acc': scores_train.get_epoch_mean_acc(epoch),
                        'val_loss': scores_val.get_epoch_loss(epoch),
                        'val_miou': scores_val.get_epoch_miou(epoch),
                        'val_acc': scores_val.get_epoch_acc(epoch),
                        'val_mean_acc': scores_val.get_epoch_mean_acc(epoch),
                        'time': epoch_duration})
        
        best_model_tracker.update(loss_val_sum.item() / len(val_dataloader), epoch, classifier)

        save_metrics(scheduler.get_lr_history(), 
                *scores_train.get_metrics_list(), 
                *scores_val.get_metrics_list(),
                save_path = save_dir,
                epoch = epoch)
        
        if early_stopping.update(loss_val_sum.item() / len(val_dataloader)):
            break

        print("", flush=True)
    
    minutes, seconds = divmod(time.time() - start_time, 60)
    monitor.log_and_print(f"Training time: {int(minutes)}:{int(seconds):02} minutes.\n"
                          f"--- DONE ---\n")

if __name__ == '__main__':
    args = parse_args()
    main(args)

# TODO:
# README
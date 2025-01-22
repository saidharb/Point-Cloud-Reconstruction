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

from dataset import PointCloudEmbeddingDataset
from models.Pointnet_Pointnet2_pytorch import provider
from metrics import RegressionRunningScore
from utils import SaveBestModel, EarlyStopping, Logger

# torch.manual_seed(42)
        
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser(
        'Train PointNet++ to encode point clouds into the same latent space as DeepCAD '
        'does with CAD Sequences.'
    )
    parser.add_argument('--data_root', 
                        type=str, 
                        default='data', 
                        help='data directory relative to root directory')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size')
    parser.add_argument('--max_epochs', type=int, default=50, help='maximum number of epochs')
    parser.add_argument('--save_interval', type=int, default=20, help='save interval for models')
    parser.add_argument('--learning_rate', type=float, default=0.001, help="initial learning rate")
    parser.add_argument('--early_stopping', 
                        type=int, 
                        default=20, 
                        help="abort training after this amount of epochs with no validation loss decrease")
    parser.add_argument('--verbose', action='store_true', default=False, help='output per batch metrics')
    parser.add_argument('--wandb', action='store_true', default=False, help='enable WandB tracking')
    parser.add_argument('--name', type=str, default="test_run", help="name of WandB run")
    parser.add_argument('--lr_patience', type=int, default=15, help="patience in epochs for learning rate decay")
    return parser.parse_args()

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def save_metrics(*lists, save_path = "", epoch = 1):
    epochs = list(range(1, epoch + 2))
    headers = ['epoch', 'learning_rate', 'train_mse', 'train_rmse', 'train_mae', 'val_mse', 'val_rmse', 'val_mae']
    with open(os.path.join(save_path, 'metrics.csv'), mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for row in zip(epochs, *lists):
            writer.writerow(row)

def main(args):

    data_and_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"### NEW TRAINING STARTED ###"
          f"\n{data_and_time}\n", flush=True)
    start_time = time.time()

    # Find data directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if os.getcwd() != root_dir:
        DATA_DIR = os.path.join(root_dir, args.data_root)
    else:
        DATA_DIR = os.path.abspath(args.data_root)

    # Find experiment directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.abspath(os.path.join(script_dir, "..", "models", "trained_models", data_and_time))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created model save directory at: {os.path.abspath(save_dir)}\n")
    # Logging
    monitor = Logger(save_dir)
    
    # Print parameters
    monitor.log_and_print("### Parameters ###\n")
    for key, value in vars(args).items():
        monitor.log_and_print(f"{key}: {value}")
    print("\n--- DONE ---\n", flush=True)

    config = {
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'max_epochs': args.max_epochs,
        'optimizer': 'Adam',
        'model_type': 'pointnet2_cls_ssg',
        'save_interval': args.save_interval,
        'early_stopping': args.early_stopping,
        'start_time': data_and_time
    }
    
    if args.wandb:
        print("### WANDB ###\n", flush=True)
        if os.getenv("WANDB_API_KEY"):
            print("Logging into WandB...\n", flush=True)
            wandb.login(key=os.getenv("WANDB_API_KEY"))
            wandb.init(project = 'Master Thesis',
                        name = args.name,
                        config = config)
        else:
            print("No WandB API key provided, WandB is disabled.\n", flush=True)

    # Load model
    print("### Load PointNet++ ssg model ###\n", flush=True)
    sys.path.append(os.path.join(root_dir, 'models','Pointnet_Pointnet2_pytorch', 'models'))
    model = importlib.import_module('pointnet2_cls_ssg')
    classifier = model.get_model(256, normal_channel=False)
    criterion = model.get_loss_mse()
    classifier.apply(inplace_relu)
    
    ## Cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    monitor.log_and_print(f"Using device: {device}\n")
    classifier = classifier.to(device)
    criterion = criterion.to(device)
    print("--- DONE ---\n", flush=True)

    # Load data
    num_workers = 0 if device.type == 'cpu' else 4
    train_dataset = PointCloudEmbeddingDataset(DATA_DIR, 'train')
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, num_workers = num_workers, shuffle = True)
    val_dataset = PointCloudEmbeddingDataset(DATA_DIR, 'validation')
    val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, num_workers = num_workers, shuffle = False)
    monitor.log(f"Train set: {len(train_dataloader)}, Validation set: {len(val_dataloader)}")

    ## Optimizer
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4
        )

    last_lr = args.learning_rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           mode = 'min', 
                                                           factor = 0.5, 
                                                           patience = args.lr_patience)

    scores_train = RegressionRunningScore(len(train_dataloader))
    scores_val = RegressionRunningScore(len(val_dataloader))

    best_model_tracker = SaveBestModel(config, save_dir, monitor)
    early_stopping = EarlyStopping(config, monitor)

    learning_rates = []
    
    # Training
    monitor.log_and_print("### Training starts ###\n")
    for epoch in range(0, args.max_epochs):
        classifier = classifier.train()
        print(f"Epoch {epoch + 1}/{args.max_epochs}", flush=True)

        for i, (pc, latent_rep) in enumerate(train_dataloader):
            optimizer.zero_grad()

            pc = pc.data.numpy()
            pc = provider.random_point_dropout(pc)
            pc[:, :, 0:3] = provider.random_scale_point_cloud(pc[:, :, 0:3])
            pc[:, :, 0:3] = provider.shift_point_cloud(pc[:, :, 0:3])
            pc = torch.Tensor(pc)
            pc = pc.transpose(2, 1) # [B, C, N]

            pc, latent_rep = pc.to(device), latent_rep.to(device)
            pred, _ = classifier(pc)
            loss_train = criterion(pred, latent_rep)

            # Metrics
            scores_train.update(loss_train.detach(), pred, latent_rep)

            loss_train.backward()
            optimizer.step()

            if args.verbose:
                print(f"Batch {i + 1}/{len(train_dataloader)}: "
                    f"Loss: {loss_train.cpu().item():.8f} --- "
                    f"RMSE: {scores_train.get_batch_rmse(loss_train):.8f} --- "
                    f"MAE: {scores_train.get_batch_mae(pred, latent_rep):.8f}", flush=True)

            # if i == 2:
            #     break

        scores_train.epoch_finished()

        print(f"{'Training':<10} --- "
              f"Avg. Loss/MSE: {scores_train.get_epoch_mse(epoch):.8f} --- "
              f"Avg. RMSE: {scores_train.get_epoch_rmse(epoch):.8f} "
              f"--- Avg. MAE: {scores_train.get_epoch_mae(epoch):.8f}", flush=True)
        scores_train.reset()

        # Evaluation
        classifier.eval()
        with torch.no_grad():
            for j, (pc, latent_rep) in enumerate(val_dataloader):
                pc, latent_rep = pc.to(device), latent_rep.to(device)
                pc = pc.transpose(2, 1)
                pred, _ = classifier(pc)
                loss_val = criterion(pred,latent_rep)

                # Metrics
                scores_val.update(loss_val.detach(), pred, latent_rep)
                
                if args.verbose:
                    print(f"Val Batch {j + 1}/{len(val_dataloader)}: "
                        f"Loss: {loss_val.cpu().item():.8f} --- "
                        f"RMSE: {scores_val.get_batch_rmse(loss_val):.8f} --- "
                        f"MAE: {scores_val.get_batch_mae(pred, latent_rep):.8f}", flush=True)

                # if j == 2:
                #     break
        
        scores_val.epoch_finished()
        print(f"Validation --- "
              f"Avg. Loss/MSE: {scores_val.get_epoch_mse(epoch):.8f} --- "
              f"Avg. RMSE: {scores_val.get_epoch_rmse(epoch):.8f} --- "
              f"Avg. MAE: {scores_val.get_epoch_mae(epoch):.8f}", flush=True)
        scores_val.reset()

        current_lr = scheduler.get_last_lr()[0]
        learning_rates.append(current_lr)
        if current_lr != last_lr:
            monitor.log_and_print(f"Learning rate was adjusted from {last_lr} to {current_lr} in this epoch.")
            last_lr = current_lr

        if args.wandb:
            if os.getenv("WANDB_API_KEY"):
                wandb.log({'epochs': epoch, 
                        'learning_rate': current_lr,
                        'train_loss': scores_train.get_epoch_mse(epoch),
                        'train_rmse': scores_train.get_epoch_rmse(epoch),
                        'train_mae': scores_train.get_epoch_mae(epoch),
                        'val_loss': scores_val.get_epoch_mse(epoch),
                        'val_rmse': scores_val.get_epoch_rmse(epoch),
                        'val_mae': scores_val.get_epoch_mae(epoch)})
        
        monitor.log(f"Epoch {epoch+1}/{args.max_epochs} --- "
                    f"Train - MSE: {scores_train.get_epoch_mse(epoch):.8f} RMSE: {scores_train.get_epoch_rmse(epoch):.8f} MAE: {scores_train.get_epoch_mae(epoch):.8f} --- "
                    f"Val - MSE: {scores_val.get_epoch_mse(epoch):.8f} RMSE: {scores_val.get_epoch_rmse(epoch):.8f} MAE: {scores_val.get_epoch_mae(epoch):.8f}")
        best_model_tracker.update(scores_val.get_epoch_mse(epoch), epoch, classifier)

        save_metrics(learning_rates, 
                     *scores_train.get_metrics_list(), 
                     *scores_val.get_metrics_list(),
                     save_path = save_dir,
                     epoch = epoch)

        if early_stopping.update(scores_val.get_epoch_mse(epoch)):
            break

        scheduler.step(scores_val.get_epoch_mse(epoch))
        print("", flush=True)
    
    minutes, seconds = divmod(time.time() - start_time, 60)
    monitor.log_and_print(f"Training time: {int(minutes)}:{int(seconds):02} minutes.\n"
                          f"--- DONE ---\n")

if __name__ == '__main__':
    args = parse_args()
    main(args)

# NEXT:
# - show time per epoch
# increase gpu usage?
# maybe log how much data is on disk?
# - Tune learning rate hyperparameter (look at convergence of loss for that)
# - write test.py (with log file at same save dir as model)
# - maybe use msg model?
# maybe train DeepCAD myself?
# Write pipeline PC->PN++->z->DeepCAD->step

# README
# - describe req.txt
# On cluster I had to install open3d using conda-forge and it is 0.18.0 and not 0.19.0
# Therefore I removed it from the req.txt

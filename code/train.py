import os
import sys
import argparse
import importlib

import torch
from torch.utils.data import DataLoader

from dataset import PointCloudEmbeddingDataset
from models.Pointnet_Pointnet2_pytorch import provider
from metrics import RegressionRunningScore
from utils import SaveBestModel

torch.manual_seed(42)
        
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

    return parser.parse_args()

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def main(args):
    # Find data directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if os.getcwd() != root_dir:
        DATA_DIR = os.path.join(root_dir, args.data_root)
    else:
        DATA_DIR = os.path.abspath(args.data_root)

    # Print parameters
    print("### Parameters ###\n", flush=True)
    for key, value in vars(args).items():
        print(f"{key}: {value}", flush=True)
    print("\n--- DONE ---\n", flush=True)

    config = {
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'max_epochs': args.max_epochs,
        'optimizer': 'Adam',
        'model_type': 'pointnet2_cls_ssg',
        'save_interval': args.save_interval
    }

    # Load data
    train_dataset = PointCloudEmbeddingDataset(DATA_DIR, 'train')
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = False)
    val_dataset = PointCloudEmbeddingDataset(DATA_DIR, 'validation')
    val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = False)

    # Load model
    print("### Load PointNet++ ssg model ###\n", flush=True)
    sys.path.append(os.path.join(root_dir, 'models','Pointnet_Pointnet2_pytorch', 'models'))
    model = importlib.import_module('pointnet2_cls_ssg')
    classifier = model.get_model(256, normal_channel=False)
    criterion = model.get_loss_mse()
    classifier.apply(inplace_relu)
    
    ## Cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n", flush=True)
    classifier = classifier.to(device)
    criterion = criterion.to(device)
    print("--- DONE ---\n", flush=True)

    ## Optimizer
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4
        )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    scores_train = RegressionRunningScore(len(train_dataloader))
    scores_val = RegressionRunningScore(len(val_dataloader))

    best_model_tracker = SaveBestModel(config)
    
    # Training
    print("### Training starts ###\n", flush=True)
    for epoch in range(0, args.max_epochs):
        classifier = classifier.train()
        print(f"Training Epoch {epoch + 1}/{args.max_epochs}")

        for i, (pc, latent_rep) in enumerate(train_dataloader):
            optimizer.zero_grad()

            # pc = pc.data.numpy()
            # pc = provider.random_point_dropout(pc)
            # pc[:, :, 0:3] = provider.random_scale_point_cloud(pc[:, :, 0:3])
            # pc[:, :, 0:3] = provider.shift_point_cloud(pc[:, :, 0:3])
            pc = torch.Tensor(pc)
            pc = pc.transpose(2, 1) # [B, C, N]

            pc, latent_rep = pc.to(device), latent_rep.to(device)
            pred, _ = classifier(pc)
            loss_train = criterion(pred, latent_rep)

            # Metrics
            scores_train.update(loss_train.detach(), pred, latent_rep)

            loss_train.backward()
            optimizer.step()

            print(f"Batch {i + 1}/{len(train_dataloader)}: "
                  f"Loss: {loss_train.cpu().item():.8f} --- "
                  f"RMSE: {scores_train.get_batch_rmse(loss_train):.8f} --- "
                  f"MAE: {scores_train.get_batch_mae(pred, latent_rep):.8f}")

            if i == 2:
                break

        scores_train.epoch_finished()
        print(f"Training epoch {epoch + 1} finished --- "
              f"Avg. Loss/MSE: {scores_train.get_epoch_mse(epoch):.8f} --- "
              f"Avg. RMSE: {scores_train.get_epoch_rmse(epoch):.8f} "
              f"--- Avg. MAE: {scores_train.get_epoch_mae(epoch):.8f}\n")
        scores_train.reset()

        # Evaluation
        classifier.eval()

        with torch.no_grad():
            print(f"Validation Epoch {epoch + 1}/{args.max_epochs}")

            for j, (pc, latent_rep) in enumerate(val_dataloader):
                pc, latent_rep = pc.to(device), latent_rep.to(device)
                pc = pc.transpose(2, 1)
                pred, _ = classifier(pc)
                loss_val = criterion(pred,latent_rep)

                # Metrics
                scores_val.update(loss_val.detach(), pred, latent_rep)
                
                print(f"Val Batch {j + 1}/{len(val_dataloader)}: "
                      f"Loss: {loss_val.cpu().item():.8f} --- "
                      f"RMSE: {scores_val.get_batch_rmse(loss_val):.8f} --- "
                      f"MAE: {scores_val.get_batch_mae(pred, latent_rep):.8f}")

                if j == 2:
                    break
        
        scores_val.epoch_finished()
        print(f"Validation epoch {epoch + 1} finished --- "
              f"Avg. Loss/MSE: {scores_val.get_epoch_mse(epoch):.8f} --- "
              f"Avg. RMSE: {scores_val.get_epoch_rmse(epoch):.8f} --- "
              f"Avg. MAE: {scores_val.get_epoch_mae(epoch):.8f}\n")
        scores_val.reset()
        
        best_model_tracker.update(scores_val.get_epoch_mse(epoch), epoch, classifier)


        print("")
        
        scheduler.step()






    
    print("--- DONE ---\n")

if __name__ == '__main__':
    args = parse_args()
    main(args)

    

# For README:
# Execute train.py either from root or from ./code
# Give the data directory always relative to root (makes it easier)
# To run the script execute before from root: export PYTHONPATH=$(pwd):$PYTHONPATH 
# no gpu necessary, adapts dynamically

# TODO: 

# logging (wandb - metrics, learning rate)
# create logging file, which is saved where the model is saved! (trainiing time,)
# (, save path)
# train time
# validation
# output during training
# early stopping
# Remove seed
# Remove breaks
# Turn on augmentation again
# write test.py (with log file at same save dir as model)
# Turn on dataset check again
# Add verbose for batch printing
# Adapt learning rate
# turn on trainloader shuffle


# CHANGELOG pointnet 16.01.: removed softmax

# NEXT
# Karpathy workflow

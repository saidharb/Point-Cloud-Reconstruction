import os
import sys
import argparse
import importlib

import torch
from torch.utils.data import DataLoader

from dataset import PointCloudEmbeddingDataset
provider_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", 'Pointnet_Pointnet2_pytorch'))
print(provider_path)
sys.path.append(provider_path)
import provider

        
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser(
        'Train PointNet++ to encode point clouds into the same latent space as DeepCAD '
        'does with CAD Sequences.'
    )
    parser.add_argument('--data_root', type=str, default='data', help='data directory relative to root directory')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--max_epoch', type=int, default=50, help='maximum number of epochs for training')
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
    print(f"Data directory: {DATA_DIR}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)
    print("", flush=True)

    # Load data
    train_dataset = PointCloudEmbeddingDataset(DATA_DIR, 'test') # CHANGE to train
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)

    # Load model
    print("### Load PointNet++ ssg model ###\n", flush=True)
    sys.path.append(os.path.join(root_dir, 'models','Pointnet_Pointnet2_pytorch', 'models'))
    model = importlib.import_module('pointnet2_cls_ssg')
    classifier = model.get_model(256, normal_channel=False)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)
    
    ## Cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n", flush=True)
    classifier = classifier.to(device)
    criterion = criterion.to(device)
    print("### DONE ### \n", flush=True)

    ## Optimizer
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4
        )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    
    # Training
    print("### Training starts ###", flush=True)
    classifier = classifier.train()
    for epoch in range(0, args.max_epoch):
        for i, (pc, latent_rep) in enumerate(train_dataloader):
            optimizer.zero_grad()

            pc = pc.data.numpy()
            pc = provider.random_point_dropout(pc)
            pc[:, :, 0:3] = provider.random_scale_point_cloud(pc[:, :, 0:3])
            pc[:, :, 0:3] = provider.shift_point_cloud(pc[:, :, 0:3])
            pc = torch.Tensor(pc)
            pc = pc.transpose(2, 1)

            pc, latent_rep = pc.to(device), latent_rep.to(device)
            break


if __name__ == '__main__':
    args = parse_args()
    main(args)

    

# For README:
# Execute train.py either from root or from ./code
# Give the data directory always relative to root (makes it easier)

# TODO: Adapt learning rate
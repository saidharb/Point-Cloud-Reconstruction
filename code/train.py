import os

import argparse
from torch.utils.data import DataLoader

from dataset import PointCloudEmbeddingDataset
        
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser(
        'Train PointNet++ to encode point clouds into the same latent space as DeepCAD '
        'does with CAD Sequences.'
    )
    # for the data_root argument: 
    # I want the user to give the data directory from the root directory
    # It would be kind of laborious to construct the relative file path to data from this script
    parser.add_argument('--data_root', type=str, default='data', help='data directory relative to root directory')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    return parser.parse_args()

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

    train_dataset = PointCloudEmbeddingDataset(DATA_DIR, 'test')
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)

if __name__ == '__main__':
    args = parse_args()
    main(args)

    

# For README:
# Execute train.py either from root or from ./code
# Give the data directory always relative to root (makes it easier)
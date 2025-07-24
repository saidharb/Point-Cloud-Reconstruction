import os
import sys
import argparse
import importlib
import h5py

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
import numpy as np

from code.dataset import PointCloudEmbeddingSequenceDataset
from code.utils import Logger

from models.DeepCAD.config.configAE import ConfigAE
from models.DeepCAD.trainer.trainerAE import TrainerAE
from models.DeepCAD.cadlib.macro import EOS_IDX

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
    parser.add_argument('--batch_size', type=int, default=48, help='batch size')
    parser.add_argument('--verbose', action='store_true', default=False, help='output per batch metrics')
    parser.add_argument('--exp_name', type=str, required=True, 
                        help='will be saved in "path/to/model/results/exp_name')
    parser.add_argument('--model_path', type=str, required=True, help='path to the trained PointNet++ model')
    parser.add_argument('--save', action='store_true', default=False,
                        help='save predicted latent representations and cad-sequences')
    parser.add_argument('--phase', type=str, choices=['train', 'validation', 'test'], 
                        default='train', help='which set to infer')
    return parser.parse_args()

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def main(args):
    print("### START ###\n")
    DATA_DIR = os.path.abspath(args.data_root)
    root_dir = os.path.dirname(DATA_DIR)
    model_path = os.path.abspath(args.model_path)
    latent_dim = 256

    # Load pretrained PointNet++
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert(os.path.exists(model_path)), f"Model path {model_path} does not exist."
    saved_model = torch.load(model_path, map_location=torch.device(device), weights_only=True)
    model_dir = os.path.dirname(os.path.abspath(model_path))

    # Create save directory for results
    results_dir = os.path.join(model_dir, "results", args.exp_name)
    h5_dir = os.path.join(results_dir, "output")
    if not os.path.exists(h5_dir):
        os.makedirs(h5_dir)

    # Logging
    monitor = Logger(results_dir, 'pipeline')
    monitor.log_and_print(f"PointNet++: {model_path}\n")
    config = saved_model['config']
    monitor.log_and_print("### PN++ PARAMETERS ###")
    for key, value in config.items():
        monitor.log_and_print(f"{key}: {value}")

    # Create PointNet++
    sys.path.append(os.path.join(root_dir, 'models','Pointnet_Pointnet2_pytorch', 'models'))
    model = importlib.import_module(config['model_type'])
    if 'architecture' in config:
        if config['architecture'] == 'own':
            classifier = model.get_model(256, normal_channel=False)
        elif config['architecture'] == "copy_author":
            classifier = model.get_model_copy_author(256, normal_channel=False)
        elif config['architecture'] == "tanh":
            classifier = model.get_model_tanh(256, normal_channel=False)
    else:
        classifier = model.get_model(256, normal_channel=False)
    criterion = model.get_loss_mse()
    classifier.apply(inplace_relu)

    # Load PointNet++ state dict
    state_dict = saved_model['model_state_dict']
    # Check if model has been saved wrapped in nn.DataParallel
    if 'module.' in next(iter(state_dict)):
        monitor.log_and_print("Model was saved wrapped in nn.DataParallel.\nRemoving 'module.' from state dict.")
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}#
    classifier.load_state_dict(state_dict)
    monitor.log_and_print(f'Loaded state dict from {model_path}.')

    # Cuda
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

    # Data
    num_workers = 0 if device.type == 'cpu' else 8
    dataset = PointCloudEmbeddingSequenceDataset(DATA_DIR, args.phase)
    dataloader = DataLoader(dataset, batch_size = batch_size, num_workers = num_workers, shuffle = False)
    num_samples = len(dataloader) # FIXME When infering from a directory adapt this dynamically
    
    # Load DeepCAD model
    cfg = ConfigAE('test', parse=False) # Creates config data and model and log dirs if they don't exist
    tr_agent = TrainerAE(cfg) # Initializes CADTransformer, CADLoss, Adam and LRScheduler
    tr_agent.load_ckpt(cfg.ckpt)
    
    # Inference
    classifier.eval()
    tr_agent.net.eval()
    monitor.log_and_print("### START INFERENCE ###")
    mse_running_loss = 0.0
    cmd_running_loss = 0.0
    args_running_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            pc, cad_seq, latent_rep, id = data["pc"], data["tgt_vec"], data["z"], data["id"]
            
            # PC -> z
            pc, latent_rep = pc.to(device), latent_rep.to(device)
            pc = pc.transpose(2, 1)
            pred, _ = classifier(pc)
            loss = criterion(pred,latent_rep)
            
            
            pred = pred.unsqueeze(1) # CRITICAL: shape = (B,1,256), NOT (1,B,256) -> unsqueeze(1 not 0)
            output = tr_agent.decode(pred)
            
            tgt_commands = cad_seq[:, :, 0]
            tgt_args = cad_seq[:, :, 1:]

            output["tgt_commands"] = tgt_commands.to(device)
            output["tgt_args"] = tgt_args.to(device)
            
            # in the original deepcad repo they extract
            # the commmands and params(args) in the get_item method of CADDataset
            # There the batch dimension is not applied yet, this is why they access only
            # two dims instead of 3. Here outside of get item i have to access 3 

            loss_dict = tr_agent.loss_func(output)
            batch_out_vec = tr_agent.logits2vec(output)

            cmd_loss = loss_dict['loss_cmd'].detach().cpu().item()
            args_loss = loss_dict['loss_args'].detach().cpu().item()
            mse_running_loss += loss.detach().cpu().item()
            cmd_running_loss += cmd_loss
            args_running_loss += args_loss

            if args.verbose:
                print(f"Batch {i + 1}/{len(dataloader)}: "
                        f"MSE-Loss: {loss.cpu().item():8.5f}", 
                        f"Commands-Loss: {cmd_loss:8.5f}", 
                        f"Arguments-Loss: {args_loss:8.5f}",
                        flush=True)

            if args.save:
                for j in range(cad_seq.shape[0]):
                    out_vec = batch_out_vec[j]
                    seq_len = tgt_commands[j].tolist().index(EOS_IDX)
                    save_path = os.path.join(h5_dir, f'{id[j]}_vec.h5')
                    with h5py.File(save_path, 'w') as fp:
                        fp.create_dataset('out_vec', data=out_vec[:seq_len], dtype=np.int32)
                        fp.create_dataset('gt_vec', data=cad_seq[j][:seq_len], dtype=np.int32)

            # if i == 10:
            #     break
    monitor.log_and_print(f"Avg. MSE-Loss: {mse_running_loss/num_samples:8.8f} " # FIXME When not infering sets, change this
                          f"Avg. Command-Loss: {cmd_running_loss/num_samples:8.5f} " 
                          f"Avg. Argument-Loss: {args_running_loss/num_samples:8.5f}")
    
    monitor.log_and_print("### DONE ###")

if __name__ == '__main__':
    args = parse_args()
    main(args)


# TODO  

#       differentiation if CAD sequence targets are available or not -> new pc testing

#       export2step

#       infer the whole train, val and testset with the best model from the cluster!

#       Update req.txt conda

# TODO  
#       denormalizing in jupyter notebook
#       from vector vec2solid
# DONE
#       Integrate command line arguments in configAE.py
#       Loss theoretisch verstehen
#       Collect inference metrics (Avg. MSE in PC->z, CADLoss z->CAD)
#       STIMMEN EIGENTLICH PC UND LATENT ÜBEREIN?? -> Ja
#       Script should be able to infer train/val/test set
#       But also one should be able to specify a dir with pointclouds inside
#       Maybe there needs to be some 

#       Refactor pc_to_cad_pipeline notebook
import os
import argparse
import importlib
import csv

import torch
from torch.utils.data import DataLoader
import torch.nn as nn 
import numpy as np

from code.dataset import PointCloudEmbeddingSequenceDataset
from code.metrics import PrimitiveExtrusionRunningScore
from code.utils import Logger
from code.pn2_deepcad import Config
from models.DeepCAD.trainer.loss import CADLoss
from models.DeepCAD.cadlib.macro import CMD_ARGS_MASK, ALL_COMMANDS, EOS_IDX, SOL_IDX, EXT_IDX, ARC_IDX, N_ARGS

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser(
        'Test PointNet++ to parse primitives of point clouds into cad sequences.'
    )
    parser.add_argument('--data_root', 
                    type=str, 
                    default='data', 
                    help='data directory relative to root directory')
    parser.add_argument('--model_path', type=str, required=True, help='path to the trained model')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size')
    parser.add_argument('--gpu', action='store_true', default=False, 
                        help="Use multiple GPU's for training.")
    parser.add_argument('--verbose', action='store_true', default=False, help='output per batch metrics')
    return parser.parse_args()

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def logits2vec(outputs, device, refill_pad=True, to_numpy=True):
    """network outputs (logits) to final CAD vector"""
    out_command = torch.argmax(torch.softmax(outputs['command_logits'], dim=-1), dim=-1)  # (N, S)
    out_args = torch.argmax(torch.softmax(outputs['args_logits'], dim=-1), dim=-1) - 1  # (N, S, N_ARGS)
    if refill_pad: # fill all unused element to -1
        mask = ~torch.tensor(CMD_ARGS_MASK).bool().to(device)[out_command.long()]
        out_args[mask] = -1

    out_cad_vec = torch.cat([out_command.unsqueeze(-1), out_args], dim=-1)
    if to_numpy:
        out_cad_vec = out_cad_vec.detach().cpu().numpy()
    return out_cad_vec

def calculate_ACC(results):

    TOLERANCE = 3

    # overall accuracy
    avg_cmd_acc = [] # ACC_cmd
    avg_param_acc = [] # ACC_param
    
    # accuracy w.r.t. each command type
    each_cmd_cnt = np.zeros((len(ALL_COMMANDS),))
    each_cmd_acc = np.zeros((len(ALL_COMMANDS),))

    # accuracy w.r.t each parameter
    args_mask = CMD_ARGS_MASK.astype(np.float32)
    N_ARGS = args_mask.shape[1]
    each_param_cnt = np.zeros([*args_mask.shape])
    each_param_acc = np.zeros([*args_mask.shape])

    B = results["tgt_commands"].shape[0]

    for i in range(B): # for each sample in the batch
        seq_length = list(results["tgt_commands"][i]).index(EOS_IDX)
        out_cmd = results["pred"][i,:seq_length,0]
        gt_cmd = results["tgt_commands"][i, :seq_length].numpy()
    
        out_param = results["pred"][i,:seq_length,1:]
        gt_param = results["tgt_args"][i, :seq_length].numpy()

        cmd_acc = (out_cmd == gt_cmd).astype(np.int32)
        param_acc = []
              
        for j in range(len(gt_cmd)):
            cmd = gt_cmd[j]
            each_cmd_cnt[cmd] += 1
            each_cmd_acc[cmd] += cmd_acc[j]
            if cmd in [SOL_IDX, EOS_IDX]:
                continue
        
            if out_cmd[j] == gt_cmd[j]: # NOTE: only account param acc for correct cmd
                tole_acc = (np.abs(out_param[j] - gt_param[j]) < TOLERANCE).astype(np.int32)
                
                # filter param that do not need tolerance (i.e. requires strictly equal)
                if cmd == EXT_IDX:
                    tole_acc[-2:] = (out_param[j] == gt_param[j]).astype(np.int32)[-2:]
                elif cmd == ARC_IDX:
                    tole_acc[3] = (out_param[j] == gt_param[j]).astype(np.int32)[3]

                valid_param_acc = tole_acc[args_mask[cmd].astype(bool)].tolist()
                param_acc.extend(valid_param_acc)
                each_param_cnt[cmd, np.arange(N_ARGS)] += 1
                each_param_acc[cmd, np.arange(N_ARGS)] += tole_acc

        if len(param_acc) == 0: # No cmd was correct, therefore no param is recorded
            param_acc = 0       # Therefore the param accuarcy for this sample is 0
        else:
            param_acc = np.mean(param_acc)
        
        avg_param_acc.append(param_acc)
        cmd_acc = np.mean(cmd_acc)
        avg_cmd_acc.append(cmd_acc)

    # acc of each command type
    each_cmd_acc = each_cmd_acc / (each_cmd_cnt + 1e-6)

    # acc of each parameter type
    each_param_acc = each_param_acc * args_mask
    each_param_cnt = each_param_cnt * args_mask
    each_param_acc = each_param_acc / (each_param_cnt + 1e-6)

    return {"each_cmd_acc": each_cmd_acc, 
            "each_cmd_cnt": each_cmd_cnt,
            "each_param_acc": each_param_acc,
            "each_param_cnt": each_param_cnt}

def main(args):
    print("### TEST STARTED ###\n")

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if os.getcwd() != root_dir:
        DATA_DIR = os.path.join(root_dir, args.data_root)
    else:
        DATA_DIR = os.path.abspath(args.data_root)

        # Find model directory
    assert(os.path.exists(args.model_path)), f"Model path {args.model_path} does not exist."
    model_dir = os.path.dirname(os.path.abspath(args.model_path))

    monitor = Logger(model_dir, 'test')

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    saved_model = torch.load(args.model_path, map_location=torch.device(device), weights_only=True)

    config = saved_model['config']
    monitor.log_and_print("### Parameters ###\n")
    for key, value in config.items():
        monitor.log_and_print(f"{key}: {value}")

    cfg = Config()
    model_name = "pn2_deepcad"
    model = importlib.import_module(model_name)
    classifier = model.get_pn2_deepcad_model(cfg, normal_channel=False)
    criterion = CADLoss(cfg)
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

    test_dataset = PointCloudEmbeddingSequenceDataset(DATA_DIR, 'test', use_normals=False)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, num_workers = num_workers, shuffle = False)
    monitor.log_and_print(f"Test Dataloader: {len(test_dataset)}, Test Dataloader: {len(test_dataloader)}")

    scores_test = PrimitiveExtrusionRunningScore(len(ALL_COMMANDS), N_ARGS, model_dir, 'test', cont=False)
    monitor.log_and_print("### Test starts ###\n")

    classifier.eval()
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            pc = data['pc']
            sequence = data['tgt_vec']

            pc = pc.transpose(2, 1)
            pc, sequence = pc.to(device), sequence.to(device)
            output = classifier(pc)
            tgt_commands = sequence[:, :, 0]
            tgt_args = sequence[:, :, 1:]

            output["tgt_commands"] = tgt_commands.to(device)
            output["tgt_args"] = tgt_args.to(device)

            loss_dict = criterion(output)

            cmd_loss = loss_dict['loss_cmd'].detach().cpu().item()
            args_loss = loss_dict['loss_args'].detach().cpu().item()

            batch_out_vec = logits2vec(output, device)

            metrics = calculate_ACC({"tgt_commands": tgt_commands,
                                    "tgt_args": tgt_args,
                                    "pred": batch_out_vec})
            
            scores_test.update(metrics, cmd_loss, args_loss, pc.shape[0])
            if args.verbose:
                print(f"Batch {i + 1}/{len(test_dataloader)}: "
                        f"Commands-Loss: {cmd_loss:8.5f}", 
                        f"Arguments-Loss: {args_loss:8.5f}",
                        flush=True)
            if i == 2:
                 break

    scores_test.epoch_finished()
    for a in scores_test.get_metrics_list():
        monitor.log_and_print(a)

    save_test_metrics(*scores_test.get_metrics_list(), save_path=model_dir)

def save_test_metrics(*lists, save_path):
    test_epoch_avg_cmd_acc, test_epoch_avg_param_acc, \
    test_epoch_cmd_loss, test_epoch_param_loss, test_epoch_per_cmd_acc, \
    test_epoch_per_param_acc = lists

    np.savez(os.path.join(save_path, "test_metrics.npz"),
             epoch_per_cmd_acc_test = np.array(test_epoch_per_cmd_acc),
             epoch_per_param_acc_test = np.array(test_epoch_per_param_acc))
    
    csv_path = os.path.join(save_path, "test_metrics.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "test_avg_cmd_acc", "test_avg_param_cc", "test_cmd_loss",
            "test_param_loss"
        ])

        for row in zip(test_epoch_avg_cmd_acc, test_epoch_avg_param_acc, test_epoch_cmd_loss,
                        test_epoch_param_loss):
            writer.writerow(row)

if __name__ == '__main__':
    args = parse_args()
    main(args)

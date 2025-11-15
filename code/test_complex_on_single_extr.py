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
from models.DeepCAD.cadlib.visualize import vec2CADsolid, CADsolid2pc
from models.DeepCAD.utils import read_ply
from scipy.spatial import cKDTree as KDTree
import random
import pickle
import pandas as pd

# Testing MEM on simple models

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
  #  parser.add_argument('--batch_size', type=int, default=24, help='batch size')
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

    # acc of each parameter type
    each_param_acc = each_param_acc * args_mask
    each_param_cnt = each_param_cnt * args_mask

    return {"each_cmd_acc": each_cmd_acc, # per cmd number of correct 
            "each_cmd_cnt": each_cmd_cnt, # per cmd count
            "each_param_acc": each_param_acc, # per param number of correct
            "each_param_cnt": each_param_cnt} # per param count

def normalize_pc(points):
    scale = np.max(np.abs(points))
    points = points / scale
    return points

def chamfer_dist(gt_points, gen_points, offset=0, scale=1):
    gen_points = gen_points / scale - offset

    # one direction
    gen_points_kd_tree = KDTree(gen_points)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer + gen_to_gt_chamfer

def calculate_CD(vec_pred, gt_pc_path, data_id):
    pred_seq_len = vec_pred[0,:,0].tolist().index(EOS_IDX)
    vec_pred = vec_pred.squeeze()[:pred_seq_len]

    try:
        shape = vec2CADsolid(vec_pred) # out vec only contains until target seq length
    except Exception as e:
        return float('nan') # Create CAD failed
    
    try:
        out_pc = CADsolid2pc(shape, 2000, data_id) # 2000 is the number of sampled points
    except Exception as e:
        return float('nan') # Create PC failed

    if np.max(np.abs(out_pc)) > 2: # normalize out-of-bound data
        out_pc = normalize_pc(out_pc)

    gt_pc = read_ply(gt_pc_path)
    sample_idx = random.sample(list(range(gt_pc.shape[0])), 2000)
    gt_pc = gt_pc[sample_idx]

    cd = chamfer_dist(gt_pc, out_pc)
    return cd

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
    batch_size = 1
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
    cd_list = []
    missing_gt_pc_counter = 0
    monitor.log_and_print("### Test starts ###\n")

    # Instantiate pandas dataframe
    rows = []
    cols = [
        "id", "set_id",
        "cmd_acc", "param_acc", "cd", "tgt_commands", "pred_commands", "tgt_args",
        "pred_args", "seq_len", "pred_seq_len",
        "cmd_count", "cmd_correct", "cmd_count_total", "cmd_correct_total",
        "per_cmd_param_count", "per_cmd_param_correct", "param_count_total", "param_correct_total",
        "cmd_loss", "param_loss", "total_loss",
        ]

    num_samples_single_extr = 0
    num_samples_multiple_extr = 0
    classifier.eval()
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            pc = data['pc']
            sequence = data['tgt_vec']
            id = data['id'][0]

            # Only Infer single sample extrusions
            counter = 0
            for cmd in sequence[0]:
                if cmd[0].item() == EXT_IDX:
                    counter += 1
                if cmd[0].item() == EOS_IDX:
                    break
            if counter > 1:
                num_samples_multiple_extr += 1
                continue
            else:
                num_samples_single_extr += 1

            pc = pc.transpose(2, 1)
            pc, sequence = pc.to(device), sequence.to(device)
            output = classifier(pc)

            tgt_commands = sequence[:, :, 0]
            tgt_args = sequence[:, :, 1:]

            seq_length = list(tgt_commands[0]).index(EOS_IDX)

            output["tgt_commands"] = tgt_commands.to(device)
            output["tgt_args"] = tgt_args.to(device)

            loss_dict = criterion(output)

            cmd_loss = loss_dict['loss_cmd'].detach().cpu().item()
            args_loss = loss_dict['loss_args'].detach().cpu().item()

            batch_out_vec = logits2vec(output, device)
            pred_commands = batch_out_vec[:, :, 0]
            pred_args = batch_out_vec[:, :, 1:]
            pred_seq_length = list(pred_commands[0]).index(EOS_IDX)

            metrics = calculate_ACC({"tgt_commands": tgt_commands,
                                    "tgt_args": tgt_args,
                                    "pred": batch_out_vec})
            
            gt_pc_path = os.path.join(DATA_DIR, "pc_from_vec", id[:4], id + ".ply")
            if not os.path.exists(gt_pc_path):
                missing_gt_pc_counter += 1
                cd = float('nan')
            else:
                cd = calculate_CD(batch_out_vec, gt_pc_path, id)
                cd_list.append(cd)

            sample_cmd_acc = np.sum(metrics["each_cmd_acc"]) / np.sum(metrics["each_cmd_cnt"] + 1e-6)
            sample_param_acc = np.sum(metrics["each_param_acc"]) / np.sum(metrics["each_param_cnt"] + 1e-6)

            COMMAND_NAMES = ['L', 'A', 'C', 'EOS', 'S', 'E']
            tgt_args_list = []
            pred_args_list = []
            for k, cmd in enumerate(tgt_commands.squeeze()[:seq_length].tolist()):
                if not k == seq_length:
                    tgt_args_list.append(f"{COMMAND_NAMES[cmd]}")
                params = tgt_args.squeeze()[k]
                selected_args = params[torch.tensor(CMD_ARGS_MASK[cmd]).bool()]
                tgt_args_list.extend(selected_args.tolist())
            for k, cmd in enumerate(pred_commands.squeeze()[:seq_length].tolist()):
                if not k == pred_seq_length:
                    pred_args_list.append(f"{COMMAND_NAMES[cmd]}")
                params = pred_args.squeeze()[k]
                selected_args = params[torch.tensor(CMD_ARGS_MASK[cmd]).bool()]
                pred_args_list.extend(selected_args.tolist())

            row = {
                "id": id,                   # str
                "set_id": i,           # int
                "cmd_acc": sample_cmd_acc,         # float
                "param_acc": sample_param_acc,     # float
                "cd": cd,                   # float
                "tgt_commands": tgt_commands.squeeze()[:seq_length].tolist(), # torch.Tensor (60,)
                "pred_commands": pred_commands.squeeze()[:pred_seq_length].tolist(),
                "tgt_args": tgt_args_list,         # torch.Tensor (60, 16)
                "pred_args": pred_args_list,
                "seq_len": seq_length,        # int
                "pred_seq_len": pred_seq_length, # int
                "cmd_count": metrics["each_cmd_cnt"].astype(int),     # torch.Tensor (6,)
                "cmd_correct": metrics["each_cmd_acc"].astype(int), # torch.Tensor (6,)
                "cmd_count_total": np.sum(metrics["each_cmd_cnt"]).astype(int), # int
                "cmd_correct_total": np.sum(metrics["each_cmd_acc"]).astype(int), # int
                "per_cmd_param_count": metrics["each_param_cnt"].astype(int), # torch.Tensor (6×16)
                "per_cmd_param_correct": metrics["each_param_acc"].astype(int), # torch.Tensor (6×16)
                "param_count_total": np.sum(metrics["each_param_cnt"]).astype(int), # int
                "param_correct_total": np.sum(metrics["each_param_acc"]).astype(int), # int
                "cmd_loss": cmd_loss,              # float
                "param_loss": args_loss,          # float
                "total_loss": cmd_loss + args_loss, # float
            }
            rows.append(row)
            
            scores_test.update(metrics, cmd_loss, args_loss, pc.shape[0])
            if args.verbose:
                print(f"Batch {i + 1}/{len(test_dataloader)}: "
                        f"Commands-Loss: {cmd_loss:8.5f}", 
                        f"Arguments-Loss: {args_loss:8.5f}",
                        flush=True)
            # if i == 10:
            #      break

    scores_test.epoch_finished()
    for a in scores_test.get_metrics_list():
        monitor.log_and_print(a)
    print(f"Missing GT PC: {missing_gt_pc_counter} out of {len(test_dataset)} samples", flush=True)
    print(f"Single Extrusion Samples: {num_samples_single_extr}, Multiple Extrusion Samples: {num_samples_multiple_extr}", flush=True)

    df = pd.DataFrame(rows, columns=cols)
    df.to_pickle(os.path.join(model_dir, "test_sample_results_single_extr.pkl"))

def save_test_metrics(*lists, cd_list, save_path):
    test_epoch_avg_cmd_acc, test_epoch_avg_param_acc, \
    test_epoch_cmd_loss, test_epoch_param_loss, test_epoch_per_cmd_acc, \
    test_epoch_per_param_acc = lists

    # with open(os.path.join(save_path, "test_per_sl_metrics.pkl"), "wb") as f:
    #     pickle.dump({
    #         "cd" : cd_dict,
    #         "cmd_acc": cmd_dict,
    #         "param_acc": param_dict
    #     }, f)

    np.save(os.path.join(save_path, "test_cd.npy"), np.array(cd_list))

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

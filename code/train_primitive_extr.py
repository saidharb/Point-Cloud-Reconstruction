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

from code.dataset import PCExtrusionSequenceDataset
from code.metrics import PrimitiveExtrusionRunningScore
from code.utils import EarlyStoppingPrimitiveExtrusion, Logger, SaveBestModelPrimitiveExtrusion, LearningRateStepSchedulerPrimitiveExtrusion
from code.LRSchedulers import CosineAnnealWarmRestart, StepLR
from code.pn2_deepcad import Config
from models.DeepCAD.trainer.loss import CADLoss
from models.DeepCAD.cadlib.macro import CMD_ARGS_MASK, ALL_COMMANDS, EOS_IDX, SOL_IDX, EXT_IDX, ARC_IDX, N_ARGS

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
        print(f"Created model save directory at: {os.path.abspath(save_dir)}\n", flush=True)
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

    cfg = Config()
    model_name = "pn2_deepcad"
    model = importlib.import_module(model_name)
    classifier = model.get_pn2_deepcad_model(cfg, normal_channel=False)
    criterion = CADLoss(cfg)
    classifier.apply(inplace_relu)

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

        ## Cuda
    monitor.log_and_print(f"Using device: {device}\n")
    monitor.log_and_print(f"Number of devices: {torch.cuda.device_count()}")#
    batch_size = args.batch_size
    if args.gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1: 
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
    print("Num. workers: ", num_workers, flush=True)
    train_dataset = PCExtrusionSequenceDataset(DATA_DIR, 'train', cfg, verbose=True)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, num_workers = num_workers, shuffle = True) # multiprocessing_context=multiprocessing.get_context("spawn")
    val_dataset = PCExtrusionSequenceDataset(DATA_DIR, 'validation', cfg, verbose=True)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, num_workers = num_workers, shuffle = False) # multiprocessing_context=multiprocessing.get_context("spawn")
    monitor.log_and_print(f"Train Dataset: {len(train_dataset)}, Validation Dataset: {len(val_dataset)}")
    monitor.log_and_print(f"Train Dataloader: {len(train_dataloader)}, Validation Dataloader: {len(val_dataloader)}")

        ## Optimizer
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4
        )
    
    if args.lr_type == 'step_adv': 
        scheduler = LearningRateStepSchedulerPrimitiveExtrusion(optimizer, 
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
        
    scores_train = PrimitiveExtrusionRunningScore(len(ALL_COMMANDS), N_ARGS, save_dir, 'train', cont=continue_training)
    scores_val = PrimitiveExtrusionRunningScore(len(ALL_COMMANDS), N_ARGS, save_dir, 'validation', cont=continue_training)

    best_model_tracker = SaveBestModelPrimitiveExtrusion(config, save_dir, monitor, cont = continue_training)
    early_stopping = EarlyStoppingPrimitiveExtrusion(config, monitor, save_dir, cont = continue_training)

    monitor.log_and_print("### Training starts ###\n")
    
    for epoch in range(first_epoch, args.max_epochs):
        classifier.train()
        print(f"Epoch {epoch + 1}/{args.max_epochs}", flush=True)
        epoch_start_time = time.time()
        for i, data in enumerate(train_dataloader):
            pc = data['pc']
            sequence = data['sequence']
            extrusion_id = data['extrusion_id']

            optimizer.zero_grad()
            pc = pc.transpose(2, 1) # [B, C, N]
            pc, sequence = pc.to(device), sequence.to(device)
            output = classifier(pc)

            tgt_commands = sequence[:, :, 0]
            tgt_args = sequence[:, :, 1:]

            output["tgt_commands"] = tgt_commands.to(device)
            output["tgt_args"] = tgt_args.to(device)

            loss_dict = criterion(output)
            loss = sum(loss_dict.values())
            loss.backward()
            optimizer.step()

            cmd_loss = loss_dict['loss_cmd'].detach().cpu().item()
            args_loss = loss_dict['loss_args'].detach().cpu().item()

            batch_out_vec = logits2vec(output, device)

            metrics = calculate_ACC({"tgt_commands": tgt_commands.cpu(),
                                     "tgt_args": tgt_args.cpu(),
                                     "pred": batch_out_vec}) # already np.array
            
            scores_train.update(metrics, cmd_loss, args_loss, pc.shape[0])

            if args.verbose:
                print(f"Batch {i + 1}/{len(train_dataloader)}: "
                        f"Commands-Loss: {cmd_loss:8.5f}", 
                        f"Arguments-Loss: {args_loss:8.5f}",
                        flush=True)
          #  if i == 3:
           #     break

        mean_cmd_acc, mean_param_acc = scores_train.get_mean_accuracy()
        avg_cmd_loss, avg_args_loss = scores_train.get_avg_loss()
        monitor.log_and_print(f"Train Epoch {epoch + 1}: Avg. Command-Loss: {avg_cmd_loss:8.5f} " 
                          f"Avg. Argument-Loss: {avg_args_loss:8.5f} "
                          f"Avg. Command-Accuracy: {mean_cmd_acc:8.5f} "
                          f"Avg. Parameter-Accuracy: {mean_param_acc:8.5f}")
        scores_train.epoch_finished()

        classifier.eval()
        with torch.no_grad():
            for i, data in enumerate(val_dataloader):
                pc = data['pc']
                sequence = data['sequence']
                extrusion_id = data['extrusion_id']

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

                metrics = calculate_ACC({"tgt_commands": tgt_commands.cpu(),
                                        "tgt_args": tgt_args.cpu(),
                                        "pred": batch_out_vec})
                
                scores_val.update(metrics, cmd_loss, args_loss, pc.shape[0])
                if args.verbose:
                    print(f"Batch {i + 1}/{len(val_dataloader)}: "
                            f"Commands-Loss: {cmd_loss:8.5f}", 
                            f"Arguments-Loss: {args_loss:8.5f}",
                            flush=True)
         #       if i == 3:
          #          break

        mean_cmd_acc, mean_param_acc = scores_val.get_mean_accuracy()
        avg_cmd_loss, avg_args_loss = scores_val.get_avg_loss()
        monitor.log_and_print(f"Val Epoch {epoch + 1}: Avg. Command-Loss: {avg_cmd_loss:8.5f} " 
                          f"Avg. Argument-Loss: {avg_args_loss:8.5f} "
                          f"Avg. Command-Accuracy: {mean_cmd_acc:8.5f} "
                          f"Avg. Parameter-Accuracy: {mean_param_acc:8.5f}")
        scores_val.epoch_finished()
        epoch_duration = (time.time() - epoch_start_time) / 60.0

        current_lr = scheduler.get_current_learning_rate()
        scheduler.update(scores_val.get_epoch_avg_cmd_acc(epoch), scores_val.get_epoch_avg_arg_acc(epoch))

        if args.wandb:
            if os.getenv("WANDB_API_KEY"):
                wandb.log({'epochs': epoch, 
                        'learning_rate': current_lr,
                        'train_cmd_loss': scores_train.get_epoch_cmd_loss(epoch),
                        'train_arg_loss': scores_train.get_epoch_arg_loss(epoch),
                        'train_avg_cmd_acc': scores_train.get_epoch_avg_cmd_acc(epoch),
                        'train_avg_arg_acc': scores_train.get_epoch_avg_arg_acc(epoch),
                        'val_cmd_loss': scores_val.get_epoch_cmd_loss(epoch),
                        'val_arg_loss': scores_val.get_epoch_arg_loss(epoch),
                        'val_avg_cmd_acc': scores_val.get_epoch_avg_cmd_acc(epoch),
                        'val_avg_arg_acc': scores_val.get_epoch_avg_arg_acc(epoch),
                        'time': epoch_duration})
        
        best_model_tracker.update(scores_val.get_epoch_avg_cmd_acc(epoch), scores_val.get_epoch_avg_arg_acc(epoch), epoch, classifier)
        save_metrics(scheduler.get_lr_history(), 
                *scores_train.get_metrics_list(), 
                *scores_val.get_metrics_list(),
                save_path = save_dir,
                epoch = epoch)
        
        if early_stopping.update(scores_val.get_epoch_avg_cmd_acc(epoch), scores_val.get_epoch_avg_arg_acc(epoch)):
            break

        print("", flush=True)

    minutes, seconds = divmod(time.time() - start_time, 60)
    monitor.log_and_print(f"Training time: {int(minutes)}:{int(seconds):02} minutes.\n"
                          f"--- DONE ---\n")

def save_metrics(*lists, save_path, epoch):
    lr_history, train_epoch_avg_cmd_acc, train_epoch_avg_param_acc, \
    train_epoch_cmd_loss, train_epoch_param_loss, train_epoch_per_cmd_acc, \
    train_epoch_per_param_acc, val_epoch_avg_cmd_acc, val_epoch_avg_param_acc, \
    val_epoch_cmd_loss, val_epoch_param_loss, val_epoch_per_cmd_acc, \
    val_epoch_per_param_acc= lists

    epoch = list(range(1, epoch + 2))

    np.savez(os.path.join(save_path, "train_metrics.npz"),
             epoch_per_cmd_acc_train = np.array(train_epoch_per_cmd_acc),
             epoch_per_param_acc_train = np.array(train_epoch_per_param_acc))
    
    np.savez(os.path.join(save_path, "val_metrics.npz"),
            epoch_per_cmd_acc_val = np.array(val_epoch_per_cmd_acc),
            epoch_per_param_acc_val = np.array(val_epoch_per_param_acc))
    
    csv_path = os.path.join(save_path, "mean_metrics.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "lr",
            "train_avg_cmd_acc", "train_avg_param_cc", "train_cmd_loss",
            "train_param_loss", "val_avg_cmd_acc", "val_avg_param_cc", "val_cmd_loss",
            "val_param_loss",
        ])

        for row in zip(epoch, lr_history,
                        train_epoch_avg_cmd_acc, train_epoch_avg_param_acc, train_epoch_cmd_loss,
                        train_epoch_param_loss, val_epoch_avg_cmd_acc, val_epoch_avg_param_acc,
                        val_epoch_cmd_loss, val_epoch_param_loss):
            writer.writerow(row)
    
    
if __name__ == '__main__':
    args = parse_args()
    main(args)

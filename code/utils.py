import os
import time
import logging
import math
import json

import torch
import torch.nn as nn 
import pandas as pd

class SaveBestModel():
    """Based on val loss (MSE)"""

    def __init__(self, config, save_dir, logger, cont = False):
        self.best_val_loss = float('inf')
        self.save_dir = save_dir
        self.best_model_path = os.path.join(self.save_dir, "best.pth")
        self.current_epoch = 0
        self.best_epoch = 0
        self.config = config
        self.logger = logger
        self.save_interval = config['save_interval']
        self.start_time = time.time()

        # Load previous metrics if continuing training
        if cont:
            df = pd.read_csv(os.path.join(save_dir, 'metrics.csv'))
            data_dict = df.to_dict(orient = 'list')
            self.best_val_loss = min(data_dict['val_mse'])
            self.current_epoch = len(data_dict['epoch'])
            self.best_epoch = data_dict['val_mse'].index(min(data_dict['val_mse']))
            self.start_time = time.time() - config['training_time_min'] * 60.0 

    def create_checkpoint(self, model):
        if isinstance(model, nn.DataParallel):
            checkpoint = {
                'model_state_dict': model.module.state_dict(),
                'config': self.config
            }
        else:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'config': self.config
            }     
        return checkpoint 

    def update(self, val_loss, epoch, model):
        
        checkpoint_bool = (self.current_epoch + 1) %self.save_interval == 0 
        if checkpoint_bool:
            checkpoint_path = os.path.abspath(os.path.join(self.save_dir, f'ckpt_{self.current_epoch + 1}.pth'))
            self.logger.log_and_print(f"Saving checkpoint model every {self.save_interval} epochs to: "
                                      f"{checkpoint_path}")
            self.config['final_epoch'] = self.current_epoch + 1
            self.config['training_time_min'] = round((time.time() - self.start_time) / 60.0, 2)
            checkpoint = self.create_checkpoint(model)
            torch.save(checkpoint, checkpoint_path)

        
        best_model_bool = val_loss < self.best_val_loss
        if best_model_bool:
            self.logger.log_and_print(f"New best model found with validation MSE: {val_loss:.8f} --- "
                                      f"Improvement to previous best in epoch"
                                      f" {self.best_epoch + 1}: {(self.best_val_loss - val_loss):.8f}")
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.logger.log_and_print(f"Saving model to: {os.path.abspath(self.best_model_path)}")
            self.config['final_epoch'] = self.current_epoch + 1
            self.config['training_time_min'] = round((time.time() - self.start_time) / 60.0, 2)
            checkpoint = self.create_checkpoint(model)
            torch.save(checkpoint, self.best_model_path)

        last_path = os.path.abspath(os.path.join(self.save_dir, f'last.pth'))
        self.config['final_epoch'] = self.current_epoch + 1
        self.config['training_time_min'] = round((time.time() - self.start_time) / 60.0, 2)
        checkpoint = self.create_checkpoint(model)
        torch.save(checkpoint, last_path)

        self.current_epoch += 1

class SaveBestModelExtrusionSeg():
    """based on mIoU"""

    def __init__(self, config, save_dir, logger, cont = False):
        self.best_miou = 0
        self.save_dir = save_dir
        self.best_model_path = os.path.join(self.save_dir, "best.pth")
        self.current_epoch = 0
        self.best_epoch = 0
        self.config = config
        self.logger = logger
        self.save_interval = config['save_interval']
        self.start_time = time.time()

        # Load previous metrics if continuing training
        if cont:
            df = pd.read_csv(os.path.join(save_dir, 'mean_metrics.csv'))
            data_dict = df.to_dict(orient = 'list')
            self.best_miou = max(data_dict['val_mIoU'])
            self.current_epoch = len(data_dict['epoch'])
            self.best_epoch = data_dict['val_mIoU'].index(max(data_dict['val_mIoU']))
            self.start_time = time.time() - config['training_time_min'] * 60.0

    def create_checkpoint(self, model):
        if isinstance(model, nn.DataParallel):
            checkpoint = {
                'model_state_dict': model.module.state_dict(),
                'config': self.config
            }
        else:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'config': self.config
            }     
        return checkpoint 

    def update(self, metric, epoch, model):
        
        checkpoint_bool = (self.current_epoch + 1) % self.save_interval == 0
        if checkpoint_bool:
            checkpoint_path = os.path.abspath(os.path.join(self.save_dir, f'ckpt_{self.current_epoch + 1}.pth'))
            self.logger.log_and_print(f"Epoch {epoch}: Saving checkpoint model every {self.save_interval} epochs to: "
                                      f"{checkpoint_path}")
            self.config['final_epoch'] = self.current_epoch + 1
            self.config['training_time_min'] = round((time.time() - self.start_time) / 60.0, 2)
            checkpoint = self.create_checkpoint(model)
            torch.save(checkpoint, checkpoint_path)

        
        best_model_bool = metric > self.best_miou
        if best_model_bool:
            self.logger.log_and_print(f"New best model found with validation mIoU: {metric:.8f} --- "
                                      f"Improvement to previous best in epoch"
                                      f" {self.best_epoch + 1}: {(metric - self.best_miou):.8f}")
            self.best_miou = metric
            self.best_epoch = epoch
            self.logger.log_and_print(f"Epoch {epoch}: Saving model to: {os.path.abspath(self.best_model_path)}")
            self.config['final_epoch'] = self.current_epoch + 1
            self.config['training_time_min'] = round((time.time() - self.start_time) / 60.0, 2)
            checkpoint = self.create_checkpoint(model)
            torch.save(checkpoint, self.best_model_path)

        last_path = os.path.abspath(os.path.join(self.save_dir, f'last.pth'))
        self.config['final_epoch'] = self.current_epoch + 1
        self.config['training_time_min'] = round((time.time() - self.start_time) / 60.0, 2)
        checkpoint = self.create_checkpoint(model)
        torch.save(checkpoint, last_path)

        self.current_epoch += 1

class SaveBestModelPrimitiveExtrusion():
    """based on mean command or parameter accuracy"""

    def __init__(self, config, save_dir, logger, cont = False):
        self.best_score = 0.0

        self.save_dir = save_dir
        self.best_model_path = os.path.join(self.save_dir, "best.pth")
        self.current_epoch = 0
        self.best_epoch = 0
        self.config = config
        self.logger = logger
        self.save_interval = config['save_interval']
        self.start_time = time.time()

        # Load previous metrics if continuing training
        if cont:
            df = pd.read_csv(os.path.join(save_dir, 'mean_metrics.csv'))
            data_dict = df.to_dict(orient = 'list')
            avg_cmd_acc = data_dict['val_avg_cmd_acc']
            avg_param_acc = data_dict['val_avg_param_cc']
            weighted_sum = [0.5 * a + 0.5 * b for a, b in zip(avg_cmd_acc, avg_param_acc)]
            self.best_score = max(weighted_sum)
            self.current_epoch = len(weighted_sum)
            self.best_epoch = weighted_sum.index(max(weighted_sum))
            self.start_time = time.time() - config['training_time_min'] * 60.0

    def create_checkpoint(self, model):
        if isinstance(model, nn.DataParallel):
            checkpoint = {
                'model_state_dict': model.module.state_dict(),
                'config': self.config
            }
        else:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'config': self.config
            }     
        return checkpoint 

    def update(self, mean_cmd_acc, mean_param_acc, epoch, model):
        new_score = 0.5 * mean_cmd_acc + 0.5 * mean_param_acc
        checkpoint_bool = (self.current_epoch + 1) % self.save_interval == 0
        if checkpoint_bool:
            checkpoint_path = os.path.abspath(os.path.join(self.save_dir, f'ckpt_{self.current_epoch + 1}.pth'))
            self.logger.log_and_print(f"Epoch {epoch}: Saving checkpoint model every {self.save_interval} epochs to: "
                                      f"{checkpoint_path}")
            self.config['final_epoch'] = self.current_epoch + 1
            self.config['training_time_min'] = round((time.time() - self.start_time) / 60.0, 2)
            checkpoint = self.create_checkpoint(model)
            torch.save(checkpoint, checkpoint_path)

        
        best_model_bool = new_score > self.best_score
        if best_model_bool:
            self.logger.log_and_print(f"New best model found with avg val cmd acc: {mean_cmd_acc:.8f}, avg val param acc {mean_param_acc:.8f}--- "
                                      f"Improvement to previous best in epoch"
                                      f" {self.best_epoch + 1}: {(new_score - self.best_score):.8f}")
            self.best_score = new_score
            self.best_epoch = epoch
            self.logger.log_and_print(f"Epoch {epoch}: Saving model to: {os.path.abspath(self.best_model_path)}")
            self.config['final_epoch'] = self.current_epoch + 1
            self.config['training_time_min'] = round((time.time() - self.start_time) / 60.0, 2)
            checkpoint = self.create_checkpoint(model)
            torch.save(checkpoint, self.best_model_path)

        last_path = os.path.abspath(os.path.join(self.save_dir, f'last.pth'))
        self.config['final_epoch'] = self.current_epoch + 1
        self.config['training_time_min'] = round((time.time() - self.start_time) / 60.0, 2)
        checkpoint = self.create_checkpoint(model)
        torch.save(checkpoint, last_path)

        self.current_epoch += 1

class EarlyStopping():

    def __init__(self, config, logger, save_dir, cont = False):
        self.max_epochs = config['early_stopping']
        self.running_epoch = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.logger = logger
        if cont:
            df = pd.read_csv(os.path.join(save_dir, 'metrics.csv'))
            data_dict = df.to_dict(orient = 'list')
            self.epoch = len(data_dict['epoch'])
            self.best_loss = min(data_dict['val_mse'])
            self.best_epoch = data_dict['val_mse'].index(min(data_dict['val_mse']))
            self.running_epoch = self.epoch - self.best_epoch


    def update(self, loss):
        if loss < self.best_loss:
            self.best_epoch = self.epoch
            self.running_epoch = 0
            self.best_loss = loss
        else:
            self.running_epoch += 1
        self.epoch += 1
        if self.running_epoch == self.max_epochs:
            self.logger.log_and_print(f"Early stopping: Validation loss did not decrease for {self.max_epochs} epochs "
                                      f"from {self.best_loss:.8f} since epoch {self.best_epoch + 1}.")
            return True
        return False
    
class EarlyStoppingExtrusionSeg():

    def __init__(self, config, logger, save_dir, cont = False):
        self.max_epochs = config['early_stopping']
        self.running_epoch = 0
        self.epoch = 0
        self.best_miou = 0
        self.best_epoch = 0
        self.logger = logger
        if cont:
            df = pd.read_csv(os.path.join(save_dir, 'mean_metrics.csv'))
            data_dict = df.to_dict(orient = 'list')
            self.epoch = len(data_dict['epoch'])
            self.best_miou = max(data_dict['val_mIoU'])
            self.best_epoch = data_dict['val_mIoU'].index(max(data_dict['val_mIoU']))
            self.running_epoch = self.epoch - self.best_epoch


    def update(self, metric):
        if metric > self.best_miou:
            self.best_epoch = self.epoch
            self.running_epoch = 0
            self.best_miou = metric
        else:
            self.running_epoch += 1
        self.epoch += 1
        if self.running_epoch == self.max_epochs:
            self.logger.log_and_print(f"Epoch {self.epoch}: Early stopping: Validation mIoU did not increase for {self.max_epochs} epochs "
                                      f"from {self.best_miou:.8f} since epoch {self.best_epoch + 1}.")
            return True
        return False
    
    
class EarlyStoppingPrimitiveExtrusion():

    def __init__(self, config, logger, save_dir, cont = False):
        self.max_epochs = config['early_stopping']
        self.running_epoch = 0
        self.epoch = 0

        self.best_score = 0.0

        self.best_epoch = 0
        self.logger = logger

        if cont:
            df = pd.read_csv(os.path.join(save_dir, 'mean_metrics.csv'))
            data_dict = df.to_dict(orient = 'list')
            avg_cmd_acc = data_dict['val_avg_cmd_acc']
            avg_param_acc = data_dict['val_avg_param_cc']
            weighted_sum = [0.5 * a + 0.5 * b for a, b in zip(avg_cmd_acc, avg_param_acc)]
            self.best_score = max(weighted_sum)
            self.best_epoch = weighted_sum.index(max(weighted_sum))



    def update(self, mean_cmd_acc, mean_param_acc):
        new_score = 0.5 * mean_cmd_acc + 0.5 * mean_param_acc
        if new_score > self.best_score:
            self.best_epoch = self.epoch
            self.running_epoch = 0
            self.best_score = new_score
        else:
            self.running_epoch += 1
        self.epoch += 1
        if self.running_epoch == self.max_epochs:
            self.logger.log_and_print(f"Epoch {self.epoch}: Early stopping: Validation score did not increase for {self.max_epochs} epochs "
                                      f"from {self.best_score:.8f} since epoch {self.best_epoch + 1}.")
            return True
        return False
    
class Logger():
    def __init__(self, save_dir, phase):
        self.logger = logging.getLogger(phase)
        self.logger.setLevel(logging.INFO)
        log_name = phase
        file_handler = logging.FileHandler(os.path.abspath(os.path.join(save_dir, log_name + ".log")))
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(file_handler)
        self.log_and_print(f"Save directory: {save_dir}")

    def log_and_print(self, information):
        print(information, flush=True)
        self.logger.info(information)

    def log(self, information):
        self.logger.info(information)

class LearningRateStepScheduler():

    def __init__(self, optimizer, factor, patience, logger, save_dir, cont = False):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.running_patience = patience
        self.best_loss = float('inf')
        self.lr_history = []
        self.logger = logger
        self.min_lr = 1e-6
        if cont:
            df = pd.read_csv(os.path.join(save_dir, 'metrics.csv'))
            data_dict = df.to_dict(orient = 'list')
            self.lr_history = data_dict['learning_rate']
            self.set_new_learning_rate(historic_lr = self.lr_history[-1])

    def get_current_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']
    
    def set_new_learning_rate(self, historic_lr = None):
        new_lr = self.compute_lr()
        if historic_lr:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = historic_lr
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

    def compute_lr(self):
        lr = self.get_current_learning_rate() * self.factor
        return lr

    def get_lr_history(self):
        return self.lr_history

    def update(self, loss):
        self.lr_history.append(self.get_current_learning_rate())

        if loss >= self.best_loss:
            self.running_patience -= 1
        else:
            self.running_patience = self.patience
            self.best_loss = loss

        if self.running_patience == 0:
            if self.get_current_learning_rate() * self.factor > self.min_lr:
                self.logger.log_and_print(f"No decrease in validation loss since {self.patience} epochs.\n"
                            f"Reducing learning rate from {self.get_current_learning_rate()} to "
                            f"{self.get_current_learning_rate() * self.factor}.")
                self.set_new_learning_rate()
                self.running_patience = self.patience
            else:
                self.logger.log_and_print(f"Reducing learning rate {self.get_current_learning_rate()} "
                                          f"by a factor of {self.factor} results in a learning rate of "
                                          f"{self.get_current_learning_rate() * self.factor} which is "
                                          "below the minimum learning rate of {self.min_lr}.")

class LearningRateStepSchedulerExtrSeg():

    def __init__(self, optimizer, factor, patience, logger, save_dir, cont = False):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.running_patience = patience
        self.best_miou = 0
        self.lr_history = []
        self.logger = logger
        self.min_lr = 1e-6
        if cont:
            df = pd.read_csv(os.path.join(save_dir, 'mean_metrics.csv'))
            data_dict = df.to_dict(orient = 'list')
            self.lr_history = data_dict['lr']
            self.set_new_learning_rate(historic_lr = self.lr_history[-1])
            self.best_miou = max(data_dict['val_mIoU'])

    def get_current_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']
    
    def set_new_learning_rate(self, historic_lr = None):
        new_lr = self.compute_lr()
        if historic_lr:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = historic_lr
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

    def compute_lr(self):
        lr = self.get_current_learning_rate() * self.factor
        return lr

    def get_lr_history(self):
        return self.lr_history

    def update(self, metric):
        self.lr_history.append(self.get_current_learning_rate())

        if metric <= self.best_miou:
            self.running_patience -= 1
        else:
            self.running_patience = self.patience
            self.best_miou = metric

        if self.running_patience == 0:
            if self.get_current_learning_rate() * self.factor > self.min_lr:
                self.logger.log_and_print(f"No increase in val mIous since {self.patience} epochs.\n"
                            f"Reducing learning rate from {self.get_current_learning_rate()} to "
                            f"{self.get_current_learning_rate() * self.factor}.")
                self.set_new_learning_rate()
                self.running_patience = self.patience
            else:
                self.logger.log_and_print(f"Reducing learning rate {self.get_current_learning_rate()} "
                                          f"by a factor of {self.factor} results in a learning rate of "
                                          f"{self.get_current_learning_rate() * self.factor} which is "
                                          "below the minimum learning rate of {self.min_lr}.")
                
class LearningRateStepSchedulerPrimitiveExtrusion():

    def __init__(self, optimizer, factor, patience, logger, save_dir, cont = False):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.running_patience = patience
        self.best_score = 0
        self.lr_history = []
        self.logger = logger
        self.min_lr = 1e-6
        if cont:
            df = pd.read_csv(os.path.join(save_dir, 'mean_metrics.csv'))
            data_dict = df.to_dict(orient = 'list')
            self.lr_history = data_dict['lr']
            self.set_new_learning_rate(historic_lr = self.lr_history[-1])
            avg_cmd_acc = data_dict['val_avg_cmd_acc']
            avg_param_acc = data_dict['val_avg_param_cc']
            weighted_sum = [0.5 * a + 0.5 * b for a, b in zip(avg_cmd_acc, avg_param_acc)]
            self.best_score = max(weighted_sum)

    def get_current_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']
    
    def set_new_learning_rate(self, historic_lr = None):
        new_lr = self.compute_lr()
        if historic_lr:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = historic_lr
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

    def compute_lr(self):
        lr = self.get_current_learning_rate() * self.factor
        return lr

    def get_lr_history(self):
        return self.lr_history

    def update(self, mean_cmd_acc, mean_param_acc):
        self.lr_history.append(self.get_current_learning_rate())
        new_score = 0.5 * mean_cmd_acc + 0.5 * mean_param_acc

        if new_score <= self.best_score:
            self.running_patience -= 1
        else:
            self.running_patience = self.patience
            self.best_score = new_score

        if self.running_patience == 0:
            if self.get_current_learning_rate() * self.factor > self.min_lr:
                self.logger.log_and_print(f"No increase in val score since {self.patience} epochs.\n"
                            f"Reducing learning rate from {self.get_current_learning_rate()} to "
                            f"{self.get_current_learning_rate() * self.factor}.")
                self.set_new_learning_rate()
                self.running_patience = self.patience
            else:
                self.logger.log_and_print(f"Reducing learning rate {self.get_current_learning_rate()} "
                                          f"by a factor of {self.factor} results in a learning rate of "
                                          f"{self.get_current_learning_rate() * self.factor} which is "
                                          "below the minimum learning rate of {self.min_lr}.")


# Note: the step-based scheduler above still uses the original configuration format;
# it is kept as-is because the published runs were trained with it.

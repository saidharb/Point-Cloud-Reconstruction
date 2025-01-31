import os
import time
import logging
import math

import torch
import torch.nn as nn 
import pandas as pd

class SaveBestModel():

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
            self.best_val_loss = max(data_dict['val_mse'])
            self.current_epoch = len(data_dict['epoch'])
            self.best_epoch = data_dict['val_mse'].index(max(data_dict['val_mse']))
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
        
        checkpoint_bool = (self.current_epoch + 1) %self.save_interval == 0 and not val_loss < self.best_val_loss
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
            self.best_loss = max(data_dict['val_mse'])
            self.best_epoch = data_dict['val_mse'].index(max(data_dict['val_mse']))
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
    
class Logger():
    def __init__(self, save_dir, phase):
        self.logger = logging.getLogger("Training")
        self.logger.setLevel(logging.INFO)
        log_name = "train" if phase == 'train' else 'test'
        file_handler = logging.FileHandler(os.path.abspath(os.path.join(save_dir, log_name + ".log")))
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(file_handler)
        if phase == 'test':
            self.logger.info(f"### TEST STARTED ###\n")
        self.log_and_print(f"Save directory: {save_dir}\n")

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
                
                # FIX: when abbruch geht mit der richtigen weiter?
        

class CosineAnnealWarmRestart():
    def __init__(self, optimizer, logger, save_dir, T_0=20, T_mult = 10, initial_lr=0.001, lr_min = 0.0, factor = 1.0, cont = False):
        self.optimizer = optimizer
        self.T_0 = T_0 # Num epochs first cycle
        self.T_mult = T_mult # Factor to increase cycle length after each restart
        self.lr_min = lr_min 
        self.T_cur = 0 # Epoch within current cycle
        self.initial_lr = initial_lr
        self.logger = logger
        self.factor = factor # factor that decreases max learning rate per restart
        self.lr_history = []
        self.lr_history.append(self.initial_lr)
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
        """Compute learning rate using the cosine formula."""
        lr = self.lr_min + (self.initial_lr - self.lr_min) * (1 + math.cos(math.pi * self.T_cur / self.T_0)) / 2
        return lr
    
    def get_lr_history(self):
        return self.lr_history

    def update(self, val_loss):
        self.lr_history.append(self.get_current_learning_rate())
        self.T_cur += 1
        if self.T_cur >= self.T_0:
            self.logger.log_and_print(f"Warm Restart of learning rate after {self.T_0} epochs "
                                      f"to {self.initial_lr * self.factor}.")
            self.T_cur = 0
            self.T_0 *= self.T_mult  # Increase cycle length if T_mult > 1
            self.initial_lr *= self.factor
            
        self.set_new_learning_rate()
    
# Mach es so, dass Step und Cosine beide subclasses von einer class sind
# Fine heraus was jede funktion in step macht und mach jede funktion in cosine dass sie das selbe macht
# somit musst du den Hauptcode kaum ändern
# die parent class damit es auch formal stimmt (schauen wo vereinfacht werden kann)

# FIX:


    


        
            


        

    
import os
from datetime import datetime
import time
import logging

import torch

class SaveBestModel():

    def __init__(self, config, save_dir):
        self.best_val_loss = float('inf')
        self.save_dir = save_dir
        self.best_model_path = os.path.join(self.save_dir, "best.pth")
        self.current_epoch = 0
        self.best_epoch = 0
        self.config = config
        self.save_interval = config['save_interval']
        self.start_time = time.time()

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"Created model save directory at: {os.path.abspath(self.save_dir)}\n", flush=True)


    def update(self, val_loss, epoch, model):

        if (self.current_epoch + 1) %self.save_interval == 0 and not val_loss < self.best_val_loss:
            checkpoint_path = os.path.abspath(os.path.join(self.save_dir, f'ckpt_{self.current_epoch + 1}.pth'))
            print(f"Saving checkpoint model every {self.save_interval} epochs to: "
                  f"{checkpoint_path}", flush=True)
            self.config['final_epoch'] = self.current_epoch + 1
            self.config['training_time_min'] = round((time.time() - self.start_time) / 60.0, 2)
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'config': self.config
            }
            torch.save(checkpoint, checkpoint_path)

        if val_loss < self.best_val_loss:
            print(f"New best model found with validation MSE: {val_loss:.8f} --- "
                  f"Improvement to previous best in epoch"
                  f" {epoch + 1}: {(self.best_val_loss - val_loss):.8f}", flush=True)
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            print(f"Saving model to: {os.path.abspath(self.best_model_path)}", flush=True)
            self.config['final_epoch'] = self.current_epoch + 1
            self.config['training_time_min'] = round((time.time() - self.start_time) / 60.0, 2)
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'config': self.config
            }
            torch.save(checkpoint, self.best_model_path)

        self.current_epoch += 1

class EarlyStopping():

    def __init__(self, config):
        self.max_epochs = config['early_stopping']
        self.epoch = 0
        self.best_loss = float('inf')
        self.best_epoch = 0

    def update(self, loss):
        if loss < self.best_loss:
            self.best_epoch = self.epoch
            self.epochs = 0
            self.best_loss = loss
        else:
            self.epochs += 1
    
        if self.epochs == self.max_epochs:
            print(f"Early stopping: Validation loss did not decrease for {self.max_epochs} epochs "
                  f"from {self.best_loss:.8f} since epoch {self.best_epoch + 1}.", flush=True)
            return True
        return False
    
class Logger():
    def __init__(self, save_dir):
        self.logger = logging.getLogger("Training")
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.abspath(os.path.join(save_dir, "info.log")))
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(file_handler)
        self.logger.info(f"### NEW TRAINING STARTED ###\n")
        self.logger.info(f"Save directory: {save_dir}\n")

    def log_and_print(self, information):
        print(information, flush=True)
        self.logger.info(information)

    def log(self, information):
        self.logger.info(information)


        
            


        

    
import os
import time
import logging

import torch

class SaveBestModel():

    def __init__(self, config, save_dir, logger):
        self.best_val_loss = float('inf')
        self.save_dir = save_dir
        self.best_model_path = os.path.join(self.save_dir, "best.pth")
        self.current_epoch = 0
        self.best_epoch = 0
        self.config = config
        self.logger = logger
        self.save_interval = config['save_interval']
        self.start_time = time.time()


    def update(self, val_loss, epoch, model):

        if (self.current_epoch + 1) %self.save_interval == 0 and not val_loss < self.best_val_loss:
            checkpoint_path = os.path.abspath(os.path.join(self.save_dir, f'ckpt_{self.current_epoch + 1}.pth'))
            self.logger.log_and_print(f"Saving checkpoint model every {self.save_interval} epochs to: "
                                      f"{checkpoint_path}")
            self.config['final_epoch'] = self.current_epoch + 1
            self.config['training_time_min'] = round((time.time() - self.start_time) / 60.0, 2)
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'config': self.config
            }
            torch.save(checkpoint, checkpoint_path)

        if val_loss < self.best_val_loss:
            self.logger.log_and_print(f"New best model found with validation MSE: {val_loss:.8f} --- "
                                      f"Improvement to previous best in epoch"
                                      f" {epoch + 1}: {(self.best_val_loss - val_loss):.8f}")
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.logger.log_and_print(f"Saving model to: {os.path.abspath(self.best_model_path)}")
            self.config['final_epoch'] = self.current_epoch + 1
            self.config['training_time_min'] = round((time.time() - self.start_time) / 60.0, 2)
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'config': self.config
            }
            torch.save(checkpoint, self.best_model_path)

        self.current_epoch += 1

class EarlyStopping():

    def __init__(self, config, logger):
        self.max_epochs = config['early_stopping']
        self.epoch = 0
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.logger = logger

    def update(self, loss):
        if loss < self.best_loss:
            self.best_epoch = self.epoch
            self.epochs = 0
            self.best_loss = loss
        else:
            self.epochs += 1
    
        if self.epochs == self.max_epochs:
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
        self.logger.info(f"### NEW TRAINING STARTED ###\n")
        self.logger.info(f"Save directory: {save_dir}\n")

    def log_and_print(self, information):
        print(information, flush=True)
        self.logger.info(information)

    def log(self, information):
        self.logger.info(information)


        
            


        

    
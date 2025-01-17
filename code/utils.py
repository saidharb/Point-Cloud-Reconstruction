import os
from datetime import datetime

import torch

class SaveBestModel():

    def __init__(self, save_interval):
        self.best_val_loss = float('inf')
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_and_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.save_dir = os.path.join(script_dir, "..", "models", "trained_models", data_and_time)
        self.best_model_path = os.path.join(self.save_dir, "best.pth")
        self.current_epoch = 0
        self.best_epoch = 0
        self.save_interval = save_interval

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"Created model save directory at: {os.path.abspath(self.save_dir)}\n")


    def update(self, val_loss, epoch, model):

        self.current_epoch += 1
        if self.current_epoch%self.save_interval == 0 and not val_loss < self.best_val_loss:
            checkpoint_path = os.path.abspath(os.path.join(self.save_dir, f'ckpt_{self.current_epoch}.pth'))
            print(f"Saving checkpoint model every {self.save_interval} epochs to: "
                  f"{checkpoint_path}")
            checkpoint_model_state_dict = model.state_dict()
            torch.save(checkpoint_model_state_dict, checkpoint_path)

        if val_loss < self.best_val_loss:
            print(f"New best model found with validation MSE: {val_loss:.8f} --- "
                  f"Improvement to previous best in epoch {epoch + 1}: {(self.best_val_loss - val_loss):.8f}")
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            print(f"Saving model to: {os.path.abspath(self.best_model_path)}")
            best_model_state_dict = model.state_dict()
            torch.save(best_model_state_dict, self.best_model_path)

        
            


        

    
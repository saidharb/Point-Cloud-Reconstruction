import os

import torch

class SaveBestModel():

    def __init__(self):
        self.best_val_loss = float('inf')
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(script_dir, "..", "models", "trained_models")
        self.best_model_path = os.path.join(self.save_dir, "best.pth")
        self.best_epoch = 0

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"Created model save directory at: {os.path.abspath(self.save_dir)}\n")


    def update(self, val_loss, epoch, model):
        if val_loss < self.best_val_loss:
            print(f"New best model found with validation MSE: {val_loss:.8f} --- "
                  f"Improvement to previous best in epoch {epoch + 1}: {(self.best_val_loss - val_loss):.8f}")
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            print(f"Saving model to: {os.path.abspath(self.best_model_path)}")
            best_model_state_dict = model.state_dict()
            torch.save(best_model_state_dict, self.best_model_path)

        

    
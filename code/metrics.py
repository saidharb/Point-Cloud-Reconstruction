import os

import torch
import pandas as pd

class RegressionRunningScore():
    def __init__(self, length, save_dir, phase = None, cont = False):
        self.length = length
        self.mse = []
        self.rmse = []
        self.mae = []
        self.running_mse = 0.0
        self.running_rmse = 0.0
        self.running_mae = 0.0
        if cont:
            df = pd.read_csv(os.path.join(save_dir, 'metrics.csv'))
            data_dict = df.to_dict(orient = 'list')
            if phase == 'train':
                self.mse = data_dict['train_mse']
                self.rmse = data_dict['train_rmse']
                self.mae = data_dict['train_mae']
            if phase == 'validation':
                self.mse = data_dict['val_mse']
                self.rmse = data_dict['val_rmse']
                self.mae = data_dict['val_mae']

    def update(self, mse, pred, target):
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(pred - target))
        self.running_mse += mse.cpu().item()
        self.running_rmse += rmse.cpu().item()
        self.running_mae += mae.cpu().item()

    def epoch_finished(self):
        self.mse.append(self.running_mse/self.length)
        self.rmse.append(self.running_rmse/self.length)
        self.mae.append(self.running_mae/self.length)

    def reset(self):
        self.running_mse = 0.0
        self.running_rmse = 0.0
        self.running_mae = 0.0

    def get_batch_rmse(self, batch_mse):
        batch_rmse = torch.sqrt(batch_mse)
        return batch_rmse.cpu().item()
    
    def get_batch_mae(self, pred, target):
        batch_mae = torch.mean(torch.abs(pred - target))
        return batch_mae.cpu().item()

    def get_metrics_list(self):
        return self.mse, self.rmse, self.mae
    
    def get_epoch_mse(self, epoch):
        return self.mse[epoch]
    
    def get_epoch_rmse(self, epoch):
        return self.rmse[epoch]
    
    def get_epoch_mae(self, epoch):
        return self.mae[epoch]
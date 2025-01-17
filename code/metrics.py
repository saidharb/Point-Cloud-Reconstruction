import torch

class RegressionRunningScore():
    def __init__(self, length):
        self.length = length
        self.rmse = []
        self.mae = []
        self.running_rmse = 0.0
        self.running_mae = 0.0

    def update(self, pred, target, mse):
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(pred - target))
        self.running_rmse += rmse.cpu().item()
        self.running_mae += mae.cpu().item()

    def reset(self):
        self.rmse.append(self.running_rmse/self.length)
        self.mae.append(self.running_mae/self.length)
        self.running_rmse = 0.0
        self.running_mae = 0.0

    def get_batch_rmse(self, batch_mse):
        batch_rmse = torch.sqrt(batch_mse)
        return batch_rmse.cpu().item()
    
    def get_batch_mae(self, pred, target):
        batch_mae = torch.mean(torch.abs(pred - target))
        return batch_mae.cpu().item()

    def get_metrics_list(self):
        return self.rmse, self.mae
    
    def get_epoch_rmse(self, epoch):
        return self.rmse[epoch]
    
    def get_epoch_mae(self, epoch):
        return self.mae[epoch]
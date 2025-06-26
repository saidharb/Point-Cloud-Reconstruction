import os

import torch
import pandas as pd
import numpy as np

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
    
    def get_best_model_metrics(self):
        best_epoch = self.mse.index(max(self.mse))
        return self.mse[best_epoch], self.rmse[best_epoch], self.mae[best_epoch]
    
class ClassificationRunningScore():
    def __init__(self, num_classes, save_dir, cont, phase):
        self.num_classes = num_classes

        # per class metrics
        self.tp_epoch_list = []
        self.fp_epoch_list = []
        self.fn_epoch_list = []
        self.class_iou_epoch_list = []
        self.class_acc_epoch_list = []

        # mean metrics
        self.miou_epoch_list = []
        self.acc_epoch_list = []
        self.mean_acc_epoch_list = []

        # Loss
        self.loss_epoch_list = []



        self.tp = np.zeros(num_classes, dtype=np.int64)
        self.fp = np.zeros(num_classes, dtype=np.int64)
        self.fn = np.zeros(num_classes, dtype=np.int64)

        if cont: 
            df = pd.read_csv(os.path.join(save_dir, 'mean_metrics.csv'))
            data_dict = df.to_dict(orient = 'list')
            if phase == 'train':
                data = np.load(os.path.join(save_dir, 'train_metrics.npz'))
                self.tp_epoch_list = list(data["tp"])
                self.fp_epoch_list = list(data["fp"])
                self.fn_epoch_list = list(data["fn"]) 
                self.class_iou_epoch_list = list(data["class_iou"])
                self.class_acc_epoch_list = list(data["class_acc"])
                self.miou_epoch_list = data_dict["train_mIoU"]
                self.acc_epoch_list = data_dict["train_acc"]
                self.mean_acc_epoch_list = data_dict["train_mean_acc"]
                self.loss_epoch_list = data_dict["train_loss"]
            if phase == 'validation':
                data = np.load(os.path.join(save_dir, 'val_metrics.npz'))
                self.tp_epoch_list = list(data["tp"])
                self.fp_epoch_list = list(data["fp"])
                self.fn_epoch_list = list(data["fn"])
                self.class_iou_epoch_list = list(data["class_iou"])
                self.class_acc_epoch_list = list(data["class_acc"])
                self.miou_epoch_list = data_dict["val_mIoU"]
                self.acc_epoch_list = data_dict["val_acc"]
                self.mean_acc_epoch_list = data_dict["val_mean_acc"]
                self.loss_epoch_list = data_dict["val_loss"]

    def update(self, pred, label):
        pred_choice = pred.cpu().data.max(1)[1].numpy()
        batch_label = label.view(-1, 1).squeeze().cpu().data.numpy()
        for c in range(self.num_classes):
            self.tp[c] += np.sum((pred_choice == c) & (batch_label == c))
            self.fp[c] += np.sum((pred_choice == c) & (batch_label != c))
            self.fn[c] += np.sum((pred_choice != c) & (batch_label == c))

    def epoch_finished(self, loss):
        self.tp_epoch_list.append(self.tp.copy())
        self.fp_epoch_list.append(self.fp.copy())
        self.fn_epoch_list.append(self.fn.copy())   
        self.loss_epoch_list.append(loss)
        self.class_iou_epoch_list.append(self.get_class_IoU())
        self.class_acc_epoch_list.append(self.get_class_accuracy())
        self.miou_epoch_list.append(self.get_mIoU())
        self.acc_epoch_list.append(self.get_accuracy())
        self.mean_acc_epoch_list.append(self.get_mean_class_accuracy())

        self.reset()

    def get_metrics_list(self):
        return self.tp_epoch_list, self.fp_epoch_list, self.fn_epoch_list, \
               self.class_iou_epoch_list, self.class_acc_epoch_list, \
               self.miou_epoch_list, self.acc_epoch_list, self.mean_acc_epoch_list, \
               self.loss_epoch_list

    def get_scores(self):
        return {'tp': self.tp, 'fp': self.fp, 'fn': self.fn}
    
    def get_class_IoU(self):
        iou = self.tp / (self.tp + self.fp + self.fn + 1e-8)
        return iou
    
    def get_mIoU(self):
        iou = self.tp / (self.tp + self.fp + self.fn + 1e-8)
        return np.mean(iou)
    
    def get_accuracy(self): 
        total_correct = np.sum(self.tp)
        total_points = np.sum(self.tp + self.fn)
        return total_correct / (total_points + 1e-8)

    def get_class_accuracy(self): 
        """Per-class accuracy: TP / (TP + FN)"""
        acc = self.tp / (self.tp + self.fn + 1e-8)
        return acc 

    def get_mean_class_accuracy(self): 
        """Mean of per-class accuracies"""
        acc = self.get_class_accuracy()
        return np.mean(acc)
    
    def reset(self):
        self.tp.fill(0)
        self.fp.fill(0)
        self.fn.fill(0)

    def get_epoch_miou(self, epoch):
        return self.miou_epoch_list[epoch]
    
    def get_epoch_acc(self, epoch):
        return self.acc_epoch_list[epoch]
    
    def get_epoch_mean_acc(self, epoch):
        return self.mean_acc_epoch_list[epoch]
    
    def get_epoch_loss(self, epoch):
        return self.loss_epoch_list[epoch]



        
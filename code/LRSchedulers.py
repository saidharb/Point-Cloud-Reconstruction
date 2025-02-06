import os
import time
import logging
import math
import json

import torch
import torch.nn as nn 
import pandas as pd

class BaseLRScheduler():
    def __init__(self, optimizer, logger, save_dir, min_lr, cont):
        self.optimizer = optimizer
        self.logger = logger
        self.save_dir = save_dir
        self.min_lr = min_lr
        self.cont = cont

    def get_current_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']
    
    def set_new_learning_rate(self):
        new_lr = self.compute_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def update(self, val_loss = None):
        self.lr_history.append(self.get_current_learning_rate())
        self.save_state()
        self.update_state()

    def get_lr_history(self):
        return self.lr_history
    
    def compute_lr(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def save_state(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def load_state(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def update_state(self):
        raise NotImplementedError("Subclasses must implement this method.")
    

class CosineAnnealWarmRestart(BaseLRScheduler):
    def __init__(self, optimizer, logger, save_dir, T_0=20, T_mult=1.5, factor=1.0, min_lr=0.0, cont=False):
        super().__init__(optimizer, logger, save_dir, min_lr, cont)

        self.T_0 = T_0 # Num epochs first cycle
        self.T_mult = T_mult # Factor to increase cycle length after each restart
        self.T_cur = 0 # Epoch within current cycle
        self.factor = factor # Factor that decreases initial learning rate per restart

        self.initial_lr = self.get_current_learning_rate()
        self.lr_history = []
        self.json_save_path = os.path.join(self.save_dir, "scheduler_state.json")

        if cont:
            self.load_state()
            self.update_state()
    
    def compute_lr(self):
        lr = self.min_lr + (self.initial_lr - self.min_lr) * (1 + math.cos(math.pi * self.T_cur / self.T_0)) / 2
        return lr
    
    def save_state(self):
        state = {
            "T_0": self.T_0,
            "T_cur": self.T_cur,
            "lr_history": self.lr_history,
            "initial_lr": self.initial_lr
        }
        with open(self.json_save_path, "w") as f:
            json.dump(state, f)

    def load_state(self):
        with open(self.json_save_path, "r") as f:
            state = json.load(f)
        self.T_0 = state.get("T_0", 0)
        self.T_cur = state.get("T_cur", [])
        self.lr_history = state.get("lr_history")
        self.initial_lr = state.get("initial_lr", 0.001)

    def update_state(self):
        self.T_cur += 1
        if self.T_cur >= self.T_0:
            self.logger.log_and_print(f"Warm Restart of learning rate after {self.T_0} epochs "
                                      f"to {self.initial_lr * self.factor}.")
            self.T_cur = 0
            self.T_0 *= self.T_mult
            self.initial_lr *= self.factor
        self.set_new_learning_rate()
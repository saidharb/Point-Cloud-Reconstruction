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

    def update(self):
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
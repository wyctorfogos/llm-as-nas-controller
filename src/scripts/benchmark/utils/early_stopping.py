import torch
import numpy as np
import os
import copy

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.

    Attributes:
        patience (int): How many epochs to wait without improvement before stopping.
        delta (float): Minimum change in the monitored quantity to qualify as improvement.
        verbose (bool): Whether to print messages about improvements and the counter.
        path (str): Where to save the model if save_to_disk=True.
        save_to_disk (bool): If True, saves a new 'best' model (state_dict) to disk whenever val_loss improves.
    """

    def __init__(self, 
                 patience=7, 
                 delta=0.0, 
                 verbose=False,
                 path='checkpoint.pt',
                 save_to_disk=False,
                 early_stopping_metric_name="val_loss"):
        """
        Args:
            patience (int): Number of epochs to wait without improvement. Default: 7
            delta (float): Minimum improvement threshold. Default: 0.0
            verbose (bool): Print messages about early stopping progress. Default: False
            path (str): Path to save the best model state_dict (if save_to_disk=True). Default: 'checkpoint.pt'
            save_to_disk (bool): Whether to save the best model to disk. Default: False
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.save_to_disk = save_to_disk
        self.early_stopping_metric_name = early_stopping_metric_name

        # Internal counters and tracking
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.best_model_wts = None

    def __call__(self, val_loss, val_bacc, model):
        if self.early_stopping_metric_name == "val_loss":
            score = -val_loss  # minimizar val_loss
            current = val_loss
            best_so_far = getattr(self, "val_loss_min", np.Inf)
        elif self.early_stopping_metric_name == "val_bacc":
            score = val_bacc   # maximizar val_bacc
            current = val_bacc
            best_so_far = getattr(self, "best_val_bacc", -np.Inf)
        else:
            raise ValueError(f"Unsupported early stopping metric: {self.early_stopping_metric_name}")

        if self.best_score is None:
            self.best_score = score
            self.best_model_wts = copy.deepcopy(model.state_dict())
            if self.early_stopping_metric_name == "val_loss":
                self.val_loss_min = val_loss
            else:
                self.best_val_bacc = val_bacc
            self._save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience} "
                    f"({self.early_stopping_metric_name}: {current:.6f} vs. best: {best_so_far:.6f})")
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.best_model_wts = copy.deepcopy(model.state_dict())
            if self.early_stopping_metric_name == "val_loss":
                self.val_loss_min = val_loss
            else:
                self.best_val_bacc = val_bacc
            self.counter = 0
            self._save_checkpoint(val_loss, model)

    def _save_checkpoint(self, val_loss, model):
        """
        Saves the best model checkpoint if save_to_disk=True. Also prints 
        a message if verbose=True.
        """
        
        if self.save_to_disk:
            # Caso não exista, a pasta será criada
            os.makedirs(self.path, exist_ok=True)
            # Salva o modelo
            torch.save(model.state_dict(), self.path+'best_model.pt')
            print(f"Saving best model...")
        if self.verbose:
            print(f"Validation loss decreased to {val_loss:.6f}.")

    def load_best_weights(self, model):
        """
        Loads the best weights found into the provided model instance.

        Args:
            model (torch.nn.Module): The model to load the best weights into.
        """
        model.load_state_dict(self.best_model_wts)
        print("Best model weights have been loaded!\n")
        return model

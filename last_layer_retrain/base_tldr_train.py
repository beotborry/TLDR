import wandb
import torch
import os
import time

from abc import ABC, abstractmethod
from copy import deepcopy

from evaluate.evaluator import Evaluator


class BaseTLDRTrain(ABC):
    """
    Abstract base class for InvariantTrain methods
    Provides support for worst group accuracy early stopping
    """
    def __init__(
        self, 
    ):
        """
        Initializes the model trainer.

        :param val_evaluator: Evaluator object for validation evaluation. Default is None.
        :type val_evaluator: Evaluator, optional
        :param verbose: Whether to print training progress. Default is False.
        :type verbose: bool
        """
        self._best_model = None 
        self._cb_best_model = None
        self._wg_acc_at_best_cb_acc = -1
        self._best_wg_acc = -1
        self._best_cb_acc = -1
        self._best_val_loss = 1000000
        self._best_proj_nmse = 1000000
        self._best_inv_proj_nmse = 1000000
        self._best_retained_acc = -1
        self.trainer = None
        self.mode = None
        
    def train(self, num_epochs, save_path=None):
        """
        Train for specified number of epochs (and do early stopping if val_evaluator given)
        """
        if save_path is not None and os.path.exists(save_path):
            self.trainer.model.load_state_dict(torch.load(save_path))
            print("Loaded model from", save_path)
            return

        for epoch in range(num_epochs):
            if self.mode == "proj":
                cycle_loss = True if epoch >= int(num_epochs * (5/6)) else False
                self.train_epoch(epoch, cycle_loss)
            else:
                self.train_epoch(epoch)
            if self.val_evaluator is not None:
                if self.mode == "erm" or self.mode == "rect":
                    self.val_evaluator.evaluate()
                else: self.val_evaluator.evaluate(cycle_loss)

                if self.mode == "erm" or self.mode == "rect":
                    if self.val_evaluator.worst_group_accuracy[1] > self._best_wg_acc:
                        self._best_wg_acc = self.val_evaluator.worst_group_accuracy[1]
                        self._best_model = deepcopy(self.trainer.model)
                        self._best_epoch = epoch
                    
                    if self.val_evaluator.cb_accuracy > self._best_cb_acc:
                        self._best_cb_acc = self.val_evaluator.cb_accuracy
                        self._cb_best_model = deepcopy(self.trainer.model)
                        self._cb_best_epoch = epoch
                        self._wg_acc_at_best_cb_acc = self.val_evaluator.worst_group_accuracy[1]
                    

                    if self.verbose:
                        print('Epoch {}: Val Worst-Group Accuracy: {}'.format(epoch, self.val_evaluator.worst_group_accuracy[1]))

                    wandb.log({
                        'Epoch': epoch,
                        f'{self.mode} Val Worst-Group Accuracy': self.val_evaluator.worst_group_accuracy[1],
                        f'{self.mode} Best Val Worst-Group Accuracy': self._best_wg_acc,
                        f"{self.mode} Val CB Accuracy": self.val_evaluator.cb_accuracy,
                        f"{self.mode} Best Val CB Accuracy": self._best_cb_acc,
                        f"{self.mode} Val Worst Group Accuracy at Best CB Accuracy": self._wg_acc_at_best_cb_acc,
                        'Mode': self.mode
                    }, 
                    step=epoch)

                    for key in self.val_evaluator.accuracies:
                        wandb.log({
                            f'{self.mode} Val {key} Accuracy': self.val_evaluator.accuracies[key],
                            'Mode': self.mode
                        },
                        step=epoch)

                elif self.mode == "proj":
                    if self.val_evaluator.retained_acc > self._best_retained_acc:
                        self._best_retained_acc = self.val_evaluator.retained_acc
                        self._best_model = deepcopy(self.trainer.model)
                        self._best_epoch = epoch

                    if self.val_evaluator.inv_proj_nmse < self._best_inv_proj_nmse:
                        self._best_inv_proj_nmse = self.val_evaluator.inv_proj_nmse

                    if self.verbose:
                        print('Epoch {}: Val Loss: {}'.format(epoch, self.val_evaluator.val_loss))
                        print('Epoch {}: Val Retained Acc: {}'.format(epoch, self.val_evaluator.retained_acc))
                        print('Best Val Retained Acc: {}'.format(self._best_retained_acc))
                        print('Epoch {}: Val Proj NMSE: {}'.format(epoch, self.val_evaluator.proj_nmse))
                        print('Epoch {}: Val Inv Proj NMSE: {}'.format(epoch, self.val_evaluator.inv_proj_nmse))
                        print('Best Val Inv Proj NMSE: {}'.format(self._best_inv_proj_nmse))
                        print('Epoch {}: Val Inv Proj Gap Norm: {}'.format(epoch, self.val_evaluator.inv_proj_gap_norm))

                    wandb.log({
                        'Epoch': epoch,
                        f'{self.mode} Val Loss': self.val_evaluator.val_loss,
                        f'{self.mode} Val Retained Acc': self.val_evaluator.retained_acc,
                        f'{self.mode} Best Val Retained Acc': self._best_retained_acc,
                        f'{self.mode} Val Proj NMSE': self.val_evaluator.proj_nmse,
                        f'{self.mode} Val Inv Proj NMSE': self.val_evaluator.inv_proj_nmse,
                        f'{self.mode} Best Val Inv Proj NMSE': self._best_inv_proj_nmse,
                        f'{self.mode} Val Inv Proj Gap Norm': self.val_evaluator.inv_proj_gap_norm,
                        f'{self.mode} Val Average Accuracy': self.val_evaluator.average_accuracy,
                        'Mode': self.mode
                    },
                    step=epoch)
            
        if save_path is not None:
            print("Saved last model to", save_path)
            torch.save(self.trainer.model.state_dict(), save_path)
                
    def train_epoch(self, epoch: int, cycle_loss=False):
        """
        Trains the model for a single epoch.

        :param epoch: The current epoch number.
        :type epoch: int
        """
        if self.mode == "proj":
            self.trainer.train_epoch(epoch, cycle_loss)
    
        else:
            self.trainer.train_epoch(epoch)

    def set_mode(self, mode):
        self.mode = mode
        self._best_model = None
        self._cb_best_model = None
        self._wg_acc_at_best_cb_acc = -1
        self._best_cb_acc = -1
        self._best_wg_acc = -1
        self._best_val_loss = 1000000
        self._best_retained_acc = -1
        self._best_proj_nmse = 1000000
        self._best_inv_proj_nmse = 1000000
        self._best_epoch = -1
        
    @property
    def best_model(self):
        """
        Property for accessing the best model.

        :return: The best model.
        :rtype: Any
        :raises NotImplementedError: If no val_evaluator is set to get worst group validation accuracy.
        """
        if self.val_evaluator is None:
            raise NotImplementedError("Cannot get best model if no val_evaluator set to \
                get worst group validation accuracy.")
        else:
            return self._best_model

    @property
    def cb_best_model(self):
        if self.val_evaluator is None:
            raise NotImplementedError(
                "Cannot get best model if no val_evaluator set to \
                get worst group validation accuracy."
            )
        else:
            return self._cb_best_model

    @property
    def best_cb_acc(self):
        if self.val_evaluator is None:
            raise NotImplementedError(
                "Cannot get worst group validation accuracy \
                no val_evaluator passed."
            )
        else:
            return self._best_cb_acc

    @property
    def best_wg_acc(self):
        """
        Property for accessing the best worst group validation accuracy.

        :return: The best worst group validation accuracy.
        :rtype: Any
        :raises NotImplementedError: If no val_evaluator is passed.
        """
        assert self.mode == "erm" or self.mode == "rect"
        if self.val_evaluator is None:
            raise NotImplementedError("Cannot get worst group validation accuracy \
                no val_evaluator passed.")
        else:
            return self._best_wg_acc

    @property
    def wg_acc_at_best_cb_acc(self):
        return self._wg_acc_at_best_cb_acc
        
    @property
    def best_val_loss(self):
        """
        Property for accessing the best validation loss.

        :return: The best validation loss.
        :rtype: Any
        :raises NotImplementedError: If no val_evaluator is passed.
        """
        assert self.mode == "proj"
        if self.val_evaluator is None:
            raise NotImplementedError("Cannot get best validation loss \
                no val_evaluator passed.")
        else:
            return self._best_val_loss
        
    @property
    def best_epoch(self):
        """
        Property for accessing the best epoch number.

        :return: The best epoch number.
        :rtype: Any
        :raises NotImplementedError: If no val_evaluator is passed.
        """
        if self.val_evaluator is None:
            raise NotImplementedError("Cannot get early stopping epoch \
                no val_evaluator passed.")
        else:
            return self._best_epoch

    @property
    def best_cb_epoch(self):
        if self.val_evaluator is None:
            raise NotImplementedError(
                "Cannot get early stopping epoch \
                no val_evaluator passed."
            )
        else:
            return self._cb_best_epoch
        
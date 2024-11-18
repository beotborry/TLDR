import random
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
import os
import wandb
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
from utils.random_seed import seed_randomness, seed_worker
from utils.custom_indices_sampler import CustomIndicesSampler
from datasets.embedding_dataset_w_label import EmbeddingDatasetWLabel

class Trainer:
    def __init__(
        self,
        trainset: Dataset,
        model: nn.Module,
        batch_size: int,
        optimizer: optim.Optimizer,
        lr_scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        max_grad_norm: Optional[float] = None,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        forward_pass: Optional[Callable[[Any], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
        sampler: Sampler = None,
        device: torch.device = torch.device("cpu"),
        verbose: bool = False,
        args=None,
        mode="erm",
    ) -> None:
        """
        Initializes an instance of the Trainer class.

        :param trainset: The training set.
        :type trainset: torch.utils.data.Dataset
        :param model: The PyTorch model to train.
        :type model: torch.nn.Module
        :param batch_size: The batch size to use during training.
        :type batch_size: int
        :param optimizer: The optimizer to use for training.
        :type optimizer: torch.optim.Optimizer
        :param criterion: The loss function to use during training. Default is nn.CrossEntropyLoss().
        :type criterion: torch.nn.Module, optional
        :param forward_pass: The forward pass function to use during training. Default is None.
        :type forward_pass: Callable[[Any], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], optional
        :param sampler: The sampler to use for creating batches. Default is None.
        :type sampler: torch.utils.data.Sampler, optional
        :param device: The device to use for computations. Default is torch.device("cpu").
        :type device: torch.device, optional
        :param verbose: Whether to print training progress. Default is False.
        :type verbose: bool, optional
        """
        seed_randomness(torch_module=torch, numpy_module=np, random_module=random)

        self.trainset = trainset
        self.model = model
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.max_grad_norm = max_grad_norm
        self.criterion = criterion
        self.batch_size = batch_size
        self.sampler = sampler
        self.verbose = verbose
        self.device = device
        self.args = args
        self.mode = mode
        
        if "OSE" in self.args.method:
            self.dropout = nn.Dropout(p=self.args.ose_ratio)
            
        if forward_pass is None:
            if "VNE" in self.args.method:
                def forward_pass(self, batch):
                    inputs, labels = batch
                    try:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                    except:
                        labels = labels.to(self.device)
                        
                    feature = self.model.backbone(inputs)
                    # print(feature.shape) # (B, 2048)
                    vne = self.get_vne(feature)
                    if "OSE" in self.args.method:
                        feature = self.dropout(feature)
                    outputs = self.model.classifier(feature)
                    loss = self.criterion(outputs, labels) + self.args.vne_alpha * vne
                    return loss, outputs, labels, vne
                
            elif "FRD" in self.args.method:
                def forward_pass(self, batch):
                    inputs, labels = batch
                    try:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                    except:
                        labels = labels.to(self.device)
                        
                    feature = self.model.backbone(inputs)
                    frd = self.get_frd(feature)
                    outputs = self.model.classifier(feature)
                    loss = self.criterion(outputs, labels) + self.args.frd_alpha * frd
                    return loss, outputs, labels, frd
                    
            else:
                def forward_pass(self, batch):
                    inputs, labels = batch
                    try:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                    except:
                        labels = labels.to(self.device)
                    outputs = self.model(inputs)

                    if self.args is not None and self.args.method == "AFR" and self.mode == "rect":
                        loss = self.criterion(outputs, labels, (self.model.classifier.weight, self.model.classifier.bias))
                    else:
                        loss = self.criterion(outputs, labels)
                    return loss, outputs, labels
                
            self.forward_pass = forward_pass
        else:
            self.forward_pass = forward_pass

        if self.batch_size != len(self.trainset):
            print("Shuffle:", (self.sampler is None and (self.args is not None and not (self.args.method == "AFR" and self.mode == "rect"))))
            self.trainloader = DataLoader(
                self.trainset,
                batch_size=self.batch_size,
                shuffle=(self.sampler is None and (self.args is not None and not (self.args.method == "AFR" and self.mode == "rect"))),
                sampler=self.sampler,
                num_workers=4,
                pin_memory=True,
                worker_init_fn=seed_worker,
            )
        else:
            if isinstance(self.trainset, EmbeddingDatasetWLabel):
                self.trainloader = [(self.trainset.embeddings, self.trainset.labels)]

    def train(self, num_epochs: int, model_save_path: str = None):
        """
        Trains for given number of epochs

        :param num_epochs: Number of epochs to train for
        :type num_epochs: int
        """
        if os.path.exists(model_save_path):
            self.model.load_state_dict(torch.load(model_save_path))
            print("Loaded model from", model_save_path)
            return

        for epoch in range(num_epochs):
            self.train_epoch(epoch)

        if model_save_path is not None:
            torch.save(self.model.state_dict(), model_save_path)
            print("Saved model to", model_save_path)
            
    def get_vne(self, H):
        Z = torch.nn.functional.normalize(H, dim=1)
        sing_val = torch.svd(Z / np.sqrt(Z.shape[0]))[1]
        eig_val = sing_val ** 2
        return - (eig_val * torch.log(eig_val)).nansum()
    
    def get_frd(self, H, crop_val = 1e-4):
        Z = torch.nn.functional.normalize(H, dim=1)
        sing_val = torch.svd(Z / np.sqrt(Z.shape[0]))[1]
        eig_val = sing_val ** 2
        return torch.sqrt(0.5 * (torch.log(eig_val[eig_val > crop_val]) ** 2).sum())

    def train_epoch(self, epoch: int) -> None:
        """
        Trains the PyTorch model for 1 epoch

        :param epoch: epoch number that is being trained (only used by logging)
        :type epoch: int
        """
        if self.args.model == "clip":
            self.model.train()
        else:
            if not self.args.backbone_freeze:
                self.model.train()
            else:
                self.model.backbone.eval()
                self.model.classifier.train()
            
        if self.args is not None and self.args.method == "TLDR" and self.sampler is not None and isinstance(self.sampler, CustomIndicesSampler):
            group_partition = self.trainset.group_partition
            group_balanced_indices = []
            min_size = min([len(group_partition[group_label]) for group_label in group_partition.keys()])
            print([len(group_partition[group_label]) for group_label in group_partition.keys()])
            for group_label in group_partition.keys():
                group_balanced_indices += random.sample(group_partition[group_label], min_size)

            self.trainloader.sampler.indices = group_balanced_indices
            print(len(self.trainloader.sampler.indices))

        vne_cache = []
        frd_cache = []
        with tqdm(self.trainloader, unit="batch", total=len(self.trainloader), disable=not self.verbose) as pbar:
            pbar.set_description(f"Epoch {epoch}")
            average_accuracy = 0.0
            for batch in pbar:
                if "VNE" in self.args.method:
                    loss, _, _, vne = self.forward_pass(self, batch)
                    vne_cache.append(vne)
                elif "FRD" in self.args.method:
                    loss, _, _, frd = self.forward_pass(self, batch)
                    frd_cache.append(frd)
                else:
                    loss, _, _ = self.forward_pass(self, batch)
                # accuracy = Trainer.compute_accuracy(outputs, labels)

                # backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                # if self.lr_scheduler is not None and isinstance(self.optimizer, optim.AdamW):
                #     self.lr_scheduler.step()
                self.optimizer.step()

                # pbar.set_postfix(loss=loss.item(), accuracy=f"{accuracy}%")
                # average_accuracy += accuracy

            # if self.lr_scheduler is not None and not isinstance(self.optimizer, optim.AdamW):
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            if "VNE" in self.args.method:
                wandb.log({"Train_VNE": torch.stack(vne_cache).mean()}, step=epoch)
            elif "FRD" in self.args.method:
                wandb.log({"Train_FRD": torch.stack(frd_cache).mean()}, step=epoch)
            return average_accuracy / len(pbar)

    @staticmethod
    def compute_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Computes the accuracy of the PyTorch model.

        :param outputs: The predicted outputs of the model.
        :type outputs: torch.Tensor
        :param labels: The ground truth labels.
        :type labels: torch.Tensor
        :return: The accuracy of the model.
        :rtype: float
        """
        if len(outputs.shape) == 1:
            outputs = outputs.unsqueeze(0)
        predicted = torch.argmax(outputs, dim=1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        return 100.0 * correct / total

    def get_trainset_outputs(self):
        """
        Gets output of model on trainset
        """
        with torch.no_grad():
            self.model.eval()
            eval_trainloader = DataLoader(
                dataset=self.trainset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=seed_worker,
            )
            with tqdm(eval_trainloader, unit="batch", total=len(self.trainloader), disable=not self.verbose) as pbar:
                outputs = []
                pbar.set_description("Getting Trainset Outputs")
                for input, _ in pbar:
                    outputs.append(self.model(input.to(self.device)))
                return torch.cat(outputs, dim=0)

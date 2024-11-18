import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from tqdm import tqdm

from utils.random_seed import seed_randomness

from sklearn.linear_model import LogisticRegression
from typing import Tuple
from utils.random_seed import seed_worker
from models.rect_model import RectModel
from datasets.embedding_dataset_w_label_w_group import EmbeddingDatasetWLabelWGroup
from datasets.embedding_dataset_w_label import EmbeddingDatasetWLabel
from models.spuco_model import SpuCoModel
from models.rect_model import RectModel

class TLDREvaluator:
    def __init__(
        self,
        testset: Dataset, 
        group_partition: Dict[Tuple[int, int], List[int]],
        group_weights: Dict[Tuple[int, int], float],
        batch_size: int,
        model: nn.Module,
        mode: str,
        classi_emb_dim: int,
        erm_model: nn.Module = None,
        device: torch.device = torch.device("cpu"),
        verbose: bool = False,
        preprocess_embed: str = "none",
        modality: str = "image",
        gap_regularize: bool = True,
        testset_convert: bool = True,
        clip_variants: str = "ViT-B/32",
    ):
        """
        Initializes an instance of the Evaluator class.

        :param testset: Dataset object containing the test set.
        :type testset: Dataset

        :param group_partition: Dictionary object mapping group keys to a list of indices corresponding to the test samples in that group.
        :type group_partition: Dict[Tuple[int, int], List[int]]

        :param group_weights: Dictionary object mapping group keys to their respective weights.
        :type group_weights: Dict[Tuple[int, int], float]

        :param batch_size: Batch size for DataLoader.
        :type batch_size: int

        :param model: PyTorch model to evaluate.
        :type model: nn.Module

        :param sklearn_linear_model: Tuple representing the coefficients and intercept of the linear model from sklearn. Default is None.
        :type sklearn_linear_model: Optional[Tuple[float, float, float, Optional[StandardScaler]]], optional

        :param device: Device to use for computations. Default is torch.device("cpu").
        :type device: torch.device, optional

        :param verbose: Whether to print evaluation results. Default is False.
        :type verbose: bool, optional
        """
          
        seed_randomness(torch_module=torch, numpy_module=np, random_module=random)

        self.testloaders = {}
        self.group_partition = group_partition
        self.group_weights = group_weights
        self.model = model
        self.erm_model = erm_model
        self.device = device
        self.clip_model, _ = clip.load(clip_variants, device=device)
        self.verbose = verbose
        self.accuracies = None
        self.mode = mode
        self.proj_testset = None
        self.preprocess_embed = preprocess_embed
        self.classi_emb_dim = classi_emb_dim
        self.modality = modality
        self.gap_regularize = gap_regularize
        self.inv_proj_gap_norm = -1
        self.gap_val = None
        self.testset_convert = testset_convert

        self.model.to(self.device)
        
        if not self.mode == "proj":
            self.n_classes = testset.num_classes
            
        # Create DataLoaders 

        if self.mode == "erm" or self.mode == "rect" or self.mode == "eval":
            # Group-Wise DataLoader
            for key in group_partition.keys():
                sampler = SubsetRandomSampler(group_partition[key])
                self.testloaders[key] = DataLoader(testset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True, shuffle=False, worker_init_fn=seed_worker)
            
            # self.testloader_all = DataLoader(testset, batch_size=len(testset), num_workers=4, pin_memory=True, shuffle=False, worker_init_fn=seed_worker)
            self.testloader_all = DataLoader(testset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=False, worker_init_fn=seed_worker)

            if self.mode == "rect":
                if self.testset_convert:
                    print("Converting to embedding dataset...")
                    if not isinstance(testset, EmbeddingDatasetWLabelWGroup) and not isinstance(testset, EmbeddingDatasetWLabel):
                        testset = self.convert_to_embedding_dataset(modality)
                        del self.testloader_all
                    if isinstance(testset, EmbeddingDatasetWLabelWGroup) or isinstance(testset, EmbeddingDatasetWLabel):
                        self.testloader_all = [(testset.embeddings, testset.labels)]
                    else:
                        self.testloader_all = DataLoader(testset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=False, worker_init_fn=seed_worker)
                else:
                    self.testloader_all = DataLoader(testset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=False, worker_init_fn=seed_worker)

        elif self.mode == "proj":
            self.img_testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=seed_worker)
    
    def convert_to_embedding_dataset(self, modality):
        if modality == "image":
            embs = []
            ls = []
            with torch.no_grad():
                for inputs, labels in tqdm(self.testloader_all):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    embs.append(self.model.classi_feature_extractor(inputs).detach().cpu().numpy())
                    ls.append(labels.detach().cpu().numpy())

            embs = torch.tensor(np.vstack(embs))
            ls = torch.tensor(np.hstack(ls))

            print("Embs: ", embs.shape, " Labels: ", ls.shape)

            return EmbeddingDatasetWLabelWGroup(embeddings=embs, labels=ls, group_partition=self.group_partition)
        
        elif modality == "text":
            embs = []
            ls = []
            with torch.no_grad():
                for data, labels in tqdm(self.testloader_all):
                    try:
                        emb = [(self.model.target_clip_embs_dict[target_word][text_target], self.model.spurious_clip_embs_dict[spurious_word][text_spurious]) for text_target, target_word, text_spurious, spurious_word in zip(data['text_target'], data['attributes']['target_name'], data['text_spurious'], data['attributes']['spurious_name'])]
                    except:
                        target_embs = self.model.clip_model.encode_text(clip.tokenize(data['text_target']).cuda()).type(torch.float32).detach().cpu().numpy()
                        spurious_embs = self.model.clip_model.encode_text(clip.tokenize(data['text_spurious']).cuda()).type(torch.float32).detach().cpu().numpy()
                        emb = [(target_embs[i], spurious_embs[i]) for i in range(len(target_embs))]
    
                    embs.append(emb)
                    ls.append(labels.detach().cpu().numpy())
            
            embs = torch.tensor(np.vstack(embs)).squeeze()
            ls = torch.tensor(np.hstack(ls))
        
            print("Embs: ", embs.shape, " Labels: ", ls.shape)
            return EmbeddingDatasetWLabelWGroup(embeddings=embs, labels=ls, group_partition=self.group_partition)

    def evaluate(self, cycle_loss=False):
        """
        Evaluates the PyTorch model on the test dataset and computes the accuracy for each group.
        """
        self.model.eval()

        if self.mode == "erm" or self.mode == "rect" or self.mode == "eval":
            self.accuracies = {}
            if self.mode == "rect":
                _, is_correct = self._evaluate_accuracy(self.testloader_all)
            
                for key in sorted(self.group_partition.keys()):
                    self.accuracies[key] = torch.mean(is_correct[self.group_partition[key]])
                    if self.verbose:
                        print(f"Group {key} Accuracy: {self.accuracies[key]}")
            else:
                for key in sorted(self.group_partition.keys()):
                    self.accuracies[key] = self._evaluate_accuracy(self.testloaders[key])[0]
                    if self.verbose:
                        print(f"Group {key} Accuracy: {self.accuracies[key]}")

            return self.accuracies
        
        elif self.mode == "proj":
            self.testloader = DataLoader(self.proj_testset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=seed_worker)
            self.val_loss = self._evaluate_val_loss(self.testloader, cycle_loss=cycle_loss)
            self.proj_nmse, self.inv_proj_nmse = self._evaluate_nmse(self.testloader)
            self.retained_acc = -1
            if self.gap_regularize:
                self.inv_proj_gap_norm = self._evaluate_inv_proj_gap_norm(self.gap_val)


    def _evaluate_inv_proj_gap_norm(self, gap_emb):
        total = 0
        with torch.no_grad():
            self.model.eval()
            gap_emb = gap_emb.type(torch.FloatTensor).to(self.device)
            if self.model.proj_model == 'linear':
                # gap_emb_projected = gap_emb @ self.model.inv_proj.weight.T       
                gap_emb_projected = gap_emb @ self.model.inv_proj[0].weight.T       
                gap_emb_projected_norm = gap_emb_projected.norm(dim=-1, keepdim=True) / self.classi_emb_dim
            else:
                if self.model.proj_activ == 'relu':
                    gap_emb_projected = gap_emb @ self.model.inv_proj[0].weight.T
                    gap_emb_projected_norm = gap_emb_projected.norm(dim=-1, keepdim=True) / self.model.inv_proj[0].weight.shape[0]
                else:
                    gap_emb_projected = None
                    for layer in self.model.inv_proj:
                        if isinstance(layer, nn.Linear):
                            if gap_emb_projected is None:
                                gap_emb_projected = gap_emb @ layer.weight.T
                            else: 
                                gap_emb_projected = gap_emb_projected @ layer.weight.T
                        else:
                            continue
                        
                    gap_emb_projected_norm = gap_emb_projected.norm(dim=-1, keepdim=True) / self.classi_emb_dim

            total += torch.sum(gap_emb_projected_norm)

        return total / len(gap_emb)

    def _evaluate_retained_acc(self, testloader: DataLoader):
        with torch.no_grad():
            correct_orig = 0
            correct_proj = 0
            total = len(testloader.dataset)
            self.erm_model.eval()
            for inputs, labels in testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Original Acc
                classi_emb = self.erm_model.backbone(inputs)
                classi_emb_norm = classi_emb.norm(dim=-1, keepdim=True)

                correct_orig += self.erm_model.classifier(classi_emb).argmax(dim=-1).eq(labels).sum().item()
                # Proj Acc
                proj_emb = self.clip_model.encode_image(inputs).type(torch.FloatTensor).to(self.device)
                proj_emb = self.model.inv_proj(proj_emb)
                
                pred = self.erm_model.classifier(proj_emb)
                
                correct_proj += pred.argmax(dim=-1).eq(labels).sum().item()

            
            print("Correct Proj: ", correct_proj / total)
            print("Correct Orig: ", correct_orig / total)
            return ((correct_proj / total) / (correct_orig / total)) * 100



    def _evaluate_nmse(self, testloader: DataLoader):
        total_proj_nmse = 0
        total_inv_proj_nmse = 0

        with torch.no_grad():
            for batch in testloader:
                classi_emb = batch[:, :self.classi_emb_dim].type(torch.FloatTensor).to(self.device)
                clip_emb = batch[:, self.classi_emb_dim:self.classi_emb_dim + self.clip_model.visual.output_dim].type(torch.FloatTensor).to(self.device)


                classi_projected, clip_projected, cycle_projected = self.model(classi_emb=classi_emb, 
                                                                            clip_emb=clip_emb,
                                                                            cycle_loss=False,)
                
                total_proj_nmse += torch.sum(torch.norm(clip_emb - classi_projected) ** 2 / torch.norm(clip_emb) ** 2)
                total_inv_proj_nmse += torch.sum(torch.norm(classi_emb - clip_projected) ** 2 / torch.norm(classi_emb) ** 2)

        return total_proj_nmse / len(testloader.dataset), total_inv_proj_nmse / len(testloader.dataset)

    def _evaluate_val_loss(self, testloader: DataLoader, cycle_loss=False):
        with torch.no_grad():
            total_loss = 0
            for batch in testloader:
                loss = self._get_proj_loss(batch, cycle_loss=cycle_loss)
                total_loss += loss
            return total_loss / len(testloader.dataset)
        
    def _get_proj_loss(self, batch, cycle_loss=False):
        with torch.no_grad():
            classi_emb = batch[:, :self.classi_emb_dim].type(torch.FloatTensor).to(self.device)
            clip_emb = batch[:, self.classi_emb_dim:self.classi_emb_dim + self.clip_model.visual.output_dim].type(torch.FloatTensor).to(self.device)

            classi_projected, clip_projected, cycle_projected = self.model(classi_emb=classi_emb, 
                                                                        clip_emb=clip_emb,
                                                                        cycle_loss=cycle_loss)
            
            loss = self._get_loss(classi_emb, clip_emb, classi_projected, clip_projected, cycle_projected)
            return loss.item()
        
    def _get_loss(self, classi_emb, clip_emb, classi_projected, clip_projected, cycle_projected):
        if cycle_projected is not None:
            return torch.nn.functional.mse_loss(
                classi_projected, clip_emb) + torch.nn.functional.mse_loss(
                clip_projected, classi_emb) + torch.nn.functional.mse_loss(
                cycle_projected, classi_emb)
        else:
            return torch.nn.functional.mse_loss(
                classi_projected, clip_emb) + torch.nn.functional.mse_loss(
                clip_projected, classi_emb)

    def _evaluate_accuracy(self, testloader: DataLoader):
        with torch.no_grad():
            correct = 0
            total = 0    
            is_correct = []
            for inputs, labels in testloader:
                try:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                except: labels = labels.to(self.device)
                
                if self.mode == "rect" or self.mode == "eval":
                    if isinstance(self.model, RectModel):
                        outputs = self.model(inputs, modality=self.modality)
                    else:
                        try:
                            outputs = self.model(inputs, modality=self.modality)
                        except:
                            outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)
                predicted = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                is_correct.append(predicted == labels)
            return correct / total, torch.cat(is_correct).type(torch.float32)
    
    def _evaluate_accuracy_sklearn_logreg(self, testloader: DataLoader):
        C, coef, intercept, scaler = self.sklearn_linear_model

        X_test, y_test = self._encode_testset(testloader)
        X_test = X_test.detach().cpu().numpy()
        y_test = y_test.detach().cpu().numpy()
        if scaler:
            X_test = scaler.transform(X_test)
        logreg = LogisticRegression(penalty='l1', C=C, solver="liblinear")
        # the fit is only needed to set up logreg
        logreg.fit(X_test[: self.n_classes], np.arange(self.n_classes))
        logreg.coef_ = coef
        logreg.intercept_ = intercept
        preds_test = logreg.predict(X_test)
        return (preds_test == y_test).mean()
    
    def _encode_testset(self, testloader):
        X_test = []
        y_test = []

        self.model.eval()
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                X_test.append(self.model.backbone(inputs))
                y_test.append(labels)
            return torch.cat(X_test), torch.cat(y_test)
        
    def evaluate_spurious_attribute_prediction(self):
        """
        Evaluates accuracy if the task was predicting the spurious attribute.
        """
        return self._evaluate_accuracy(self.spurious_dataloader)

    @property
    def worst_group_accuracy(self):
        """
        Returns the group with the lowest accuracy and its corresponding accuracy.

        :returns: A tuple containing the key of the worst-performing group and its corresponding accuracy.
        :rtype: tuple
        """
        if self.accuracies is None:
            print("Run evaluate() first")
            return None
        else:
            min_key = min(self.accuracies, key=self.accuracies.get)
            min_value = min(self.accuracies.values())
            return (min_key, min_value)
        
    @property
    def cb_accuracy(self):
        if isinstance(self.model, SpuCoModel):
            class_wise_accuracies = torch.zeros(self.model.classifier.out_features)
            class_wise_cnt = torch.zeros(self.model.classifier.out_features)
        elif isinstance(self.model, RectModel):
            class_wise_accuracies = torch.zeros(self.model.linear_layer.out_features)
            class_wise_cnt = torch.zeros(self.model.linear_layer.out_features)
        for key in self.group_partition.keys():
            label = key[0]
            try:
                class_wise_accuracies[label] += self.accuracies[key] * len(self.group_partition[key])
            except:
                class_wise_accuracies[label] += self.accuracies[key].item() * len(self.group_partition[key])
                
            class_wise_cnt[label] += len(self.group_partition[key])
            
        return (class_wise_accuracies / class_wise_cnt).mean().item()
    
    @property
    def average_accuracy(self):
        """
        Returns the weighted average accuracy across all groups.

        :returns: The weighted average accuracy across all groups.
        :rtype: float
        """
        if self.accuracies is None:
            print("Run evaluate() first")
            return None
        else:
            accuracy = 0
            for key in self.group_partition.keys():
                accuracy += self.group_weights[key] * self.accuracies[key]
            return accuracy
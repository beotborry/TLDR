import torch.nn as nn
import torch.optim as optim
import torch
import clip
import numpy as np
import os
import random
import wandb
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import Dataset
from evaluate.tldr_evaluator import TLDREvaluator
from last_layer_retrain.base_tldr_train import BaseTLDRTrain
from models.projector import Projector
from models.rect_model import RectModel
from utils.trainer import Trainer
from utils.random_seed import seed_randomness, seed_worker
from torch.utils.data import DataLoader
from utils.custom_indices_sampler import CustomIndicesSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets.text_dataset import TextDataset
from datasets.generate_text import TextDatasetGenerator

class TLDR(BaseTLDRTrain):
    def __init__(self, 
                model: nn.Module,
                proj_model: nn.Module,
                erm_dataset: Dataset,
                proj_dataset: Dataset,
                proj_val_dataset: Dataset,
                rect_val_dataset: Dataset,
                erm_val_evaluator: TLDREvaluator,
                erm_optimizer: optim.Optimizer,
                scheduler,
                classi_emb_dim: int,
                device,
                verbose: bool,
                args,
                proj_gap_dataset=None,
                proj_gap_val_dataset=None,
                ):
        
        seed_randomness(torch_module=torch, numpy_module=np, random_module=random)

        self.model = model
        self.proj_model = proj_model
        self.clip_model, _ = clip.load(args.clip_variants, device="cuda")
        self.clip_model.eval()
        self.clip_emb_dim = self.clip_model.visual.output_dim
        self.classi_emb_dim = classi_emb_dim

        self.args = args
        self.erm_save_path = args.erm_save_path
        self.proj_save_path = args.proj_save_path
        self.train_emb_save_path = args.train_emb_save_path
        self.val_emb_save_path = args.val_emb_save_path
 
        if not os.path.exists(os.path.join(args.log_dir, args.exp_name)):
            os.makedirs(os.path.join(args.log_dir, args.exp_name))
            
        self.rect_save_path = args.rect_save_path

        self.erm_dataset = erm_dataset
        self.proj_dataset = proj_dataset
        self.proj_val_dataset = proj_val_dataset
        self.proj_gap_dataset = proj_gap_dataset
        self.proj_gap_val_dataset = proj_gap_val_dataset

        self.rect_val_dataset = rect_val_dataset
    
        self.erm_val_evaluator = erm_val_evaluator

        self.scheduler = scheduler
        self.device = device
        self.verbose = verbose

        self.erm_optimizer = erm_optimizer

    def train_all(self):
        """
        Trains the model.
        """
        # 1. Train ERM
        self.set_mode("erm")
        self.set_trainer_erm()
        self.train(num_epochs=self.args.num_epochs, save_path=self.erm_save_path)

        self.erm_model = self.trainer.model
        self.feature_extractor = self.erm_model.backbone
        self.feature_extractor.eval()

        # 2. Train Proj

        if self.args.proj_method == "analytic":
            if os.path.exists(self.proj_save_path):
                print("Loading proj model from {}".format(self.proj_save_path))
                self.proj_model = Projector(classi_emb_dim=self.classi_emb_dim, clip_emb_dim=self.clip_emb_dim, use_relu=self.args.model=="resnet50")
                self.proj_model.load_state_dict(torch.load(self.proj_save_path))
                self.proj_model = self.proj_model.inv_proj
            else:
                self.proj_model = Projector(classi_emb_dim=self.classi_emb_dim, clip_emb_dim=self.clip_emb_dim, use_relu=self.args.model=="resnet50")

                self.embeddings = self.encode_dataset(self.proj_dataset, split="train")
                self.embeddings_val = self.encode_dataset(self.proj_val_dataset, split="val")

                if self.args.n_gap_estimates > 0:
                    self.proj_gap = self.get_modality_gap(self.proj_gap_dataset, split="train")
                    self.proj_gap_val = self.get_modality_gap(self.proj_gap_val_dataset, split="val")

                if self.args.use_mean_gap:
                    if self.args.n_gap_estimates > 1:
                        self.proj_gap = self.proj_gap.mean(dim=0).unsqueeze(0)
                        self.proj_gap_val = self.proj_gap_val.mean(dim=0).unsqueeze(0)
                    elif self.args.n_gap_estimates == 1:
                        self.proj_gap = self.proj_gap.unsqueeze(0)
                        self.proj_gap_val = self.proj_gap_val.unsqueeze(0)

                Y = self.embeddings[:, :self.classi_emb_dim]
                X = self.embeddings[:, self.classi_emb_dim:]
                if self.args.n_gap_estimates > 0:
                    G = self.proj_gap

                lamb = self.args.proj_weight_decay

                beta_hat = torch.from_numpy(np.linalg.inv(X.T @ X + lamb * torch.eye(self.clip_emb_dim))) @ X.T @ Y
                del self.embeddings

                XTX_inv = torch.from_numpy(np.linalg.inv(X.T @ X + lamb * torch.eye(self.clip_emb_dim)))

                if self.args.n_gap_estimates > 0:
                    PI_inv = torch.from_numpy(np.linalg.inv(G @ XTX_inv @ G.T))

                    optimal_W = beta_hat - XTX_inv @ G.T @ PI_inv @ G @ beta_hat
                    optimal_b = torch.mean(Y - X @ optimal_W, axis=0)
                elif self.args.n_gap_estimates == 0:
                    optimal_W = beta_hat
                    optimal_b = torch.mean(Y - X @ optimal_W, axis=0)

                self.proj_model.inv_proj[0].weight = nn.Parameter(optimal_W.T.float().to(self.device))
                self.proj_model.inv_proj[0].bias = nn.Parameter(optimal_b.float().to(self.device))

                torch.save(self.proj_model.state_dict(), self.proj_save_path)
                self.proj_model = self.proj_model.inv_proj

                if self.args.n_gap_estimates > 0:
                    wandb.log({
                        'Proj Val Inv Proj Gap Norm': torch.norm(self.proj_gap_val.float().to(self.device) @ self.proj_model[0].weight.T).item() / self.classi_emb_dim,
                    })

        if self.args.proj_only:
            wandb.finish(exit_code = 0)
            return
        
        # 3. Train Rect
        self.rect_train_one_model()

    def encode_dataset(self, dataset, split="train"):
        """
        Encode the dataset using the feature extractor.
        """
        if split == "train":
            emb_save_path = self.train_emb_save_path
        elif split == "val":
            emb_save_path = self.val_emb_save_path

        if os.path.exists(emb_save_path):
            print("Loading embeddings from {}".format(emb_save_path))
            return torch.load(emb_save_path)

        dataloader = DataLoader(
            dataset,
            batch_size=64,
            sampler=None,
            drop_last=False,
            num_workers=4,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )

        classi_features = []
        clip_features = []
        with torch.no_grad():
            for img, _ in tqdm(dataloader):
                img = img.cuda()

                classi_features.append(self.feature_extractor(img).squeeze().type(torch.float64).cpu())
                clip_feature = self.clip_model.encode_image(img).squeeze().type(torch.float64).cpu()

                if self.args.preprocess_embed == "clip_normalize":
                    clip_feature = F.normalize(clip_feature, dim=-1)
                if len(clip_feature.shape) == 1:
                    clip_feature = clip_feature.unsqueeze(0)
                    classi_features[-1] = classi_features[-1].unsqueeze(0)

                clip_features.append(clip_feature)

        classi_features = torch.cat(classi_features)
        clip_features = torch.cat(clip_features)

        print("Classi features shape: {}".format(classi_features.shape))
        print("Clip features shape: {}".format(clip_features.shape))

        # save embeddings
        torch.save(torch.cat([classi_features, clip_features], dim=1), emb_save_path)
        print("Saved embeddings to {}".format(emb_save_path))

        return torch.cat([classi_features, clip_features], dim=1)
    
    def get_modality_gap(self, dataset, split='train'):
        if split == 'train':
            save_path = os.path.join(self.args.log_dir, self.args.log_name + f"pe_{self.args.preprocess_embed}_train_gap_n_sample_{len(dataset)}_gap_ds_{self.args.gap_dataset}.pt")
        elif split == 'val':
            save_path = os.path.join(self.args.log_dir, self.args.log_name + f"pe_{self.args.preprocess_embed}_val_gap_n_sample_{len(dataset)}_gap_ds_{self.args.gap_dataset}.pt")

        if os.path.exists(save_path):
            print("Loading modality gap from {}".format(save_path))
            return torch.load(save_path)
        
        dataloader = DataLoader(dataset, batch_size=64, sampler=None, drop_last=False, num_workers=4, pin_memory=True, worker_init_fn=seed_worker)

        img_features = []
        text_features = []

        with torch.no_grad():
            for img, text in tqdm(dataloader):
                img = img.to(self.device)
                text = text.to(self.device)
                img_features.append(self.clip_model.encode_image(img).squeeze().type(torch.float64).cpu())
                text_features.append(self.clip_model.encode_text(text).squeeze().type(torch.float64).cpu())

        img_features = torch.cat(img_features)
        text_features = torch.cat(text_features)

        if self.args.preprocess_embed == "clip_normalize":
            img_features = F.normalize(img_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

        gap = img_features - text_features

        print("Gap shape: {}".format(gap.shape))
        torch.save(gap, save_path)
        print("Saved modality gap to {}".format(save_path))

        return gap

    def rect_train_one_model(self, save=True):
        self.set_mode("rect")
        self.rect_model = RectModel(
            dataset=self.args.dataset,
            classi_feature_extractor=self.feature_extractor if not self.args.model == "clip" else self.clip_model.encode_image,
            clip_model=self.clip_model,
            target_clip_embs_path = self.args.target_words_clip_embs_path,
            spurious_clip_embs_path = self.args.spurious_words_clip_embs_path,
            inv_proj=self.proj_model,
            classi_emb_dim=self.classi_emb_dim,
            num_classes=self.erm_dataset.num_classes,
            text_train_aug=self.args.text_train_aug,
            preprocess_embed=self.args.preprocess_embed,
            train_aug_ratio=self.args.text_train_aug_ratio,
        )
        
        self.rect_val_evaluator = TLDREvaluator(
            testset=self.rect_val_dataset,
            group_partition=self.rect_val_dataset.group_partition,
            group_weights=self.erm_dataset.group_weights,
            batch_size=self.args.rect_batch_size,
            model=self.rect_model,
            device=self.device,
            verbose=True,
            mode="rect",
            classi_emb_dim=self.classi_emb_dim,
            modality=self.rect_val_dataset.modality,
            clip_variants=self.args.clip_variants,
        )

        if self.args.rect_optimizer == "AdamW":
            self.rect_optimizer = optim.AdamW(
                self.rect_model.linear_layer.parameters(), lr=self.args.rect_lr, weight_decay=self.args.rect_weight_decay
            )
        else:
            self.rect_optimizer = optim.SGD(
                self.rect_model.linear_layer.parameters(), lr=self.args.rect_lr, momentum=0.9, weight_decay=self.args.rect_weight_decay
            )

        if not self.args.proj_only:
            text_generator = TextDatasetGenerator(self.args, self.erm_dataset.num_classes)
            self.rect_dataset = text_generator.prepare_dataset()
            self.rect_dataset = TextDataset(self.rect_dataset)

        self.set_trainer_rect()

        self.train(
            num_epochs=self.args.rect_num_epochs,
            save_path=None,
        )

    def set_trainer_erm(
        self,
    ):
        self.trainer = Trainer(
            trainset=self.erm_dataset,
            model=self.model,
            batch_size=self.args.batch_size,
            optimizer=self.erm_optimizer,
            criterion=nn.CrossEntropyLoss(),
            device=self.device,
            lr_scheduler=self.scheduler,
            max_grad_norm=None,
            verbose=self.verbose,
            args = self.args,
        )

        self.val_evaluator = self.erm_val_evaluator

    def set_trainer_rect(self,):
        if self.args.use_scheduler:
            self.lr_scheduler = CosineAnnealingLR(optimizer=self.rect_optimizer, 
                                                    T_max = self.args.rect_num_epochs,
                                                    verbose=self.verbose,)
        else:
            self.lr_scheduler = None
        self.trainer = Trainer(
            trainset=self.rect_dataset,
            model=self.rect_model,
            batch_size=self.args.rect_batch_size,
            optimizer=self.rect_optimizer,
            criterion=nn.CrossEntropyLoss(),
            device=self.device,
            lr_scheduler= self.lr_scheduler,
            max_grad_norm=None,
            verbose=self.verbose,
            sampler=CustomIndicesSampler([], shuffle=True),
            args = self.args,
        )

        self.val_evaluator = self.rect_val_evaluator

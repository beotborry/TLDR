import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import pickle
from datasets.text_template import openai_imagenet_template
from utils.random_seed import seed_randomness

class RectModel(nn.Module):
    def __init__(self, dataset, classi_feature_extractor, clip_model, target_clip_embs_path, spurious_clip_embs_path, inv_proj, classi_emb_dim, num_classes, text_train_aug, preprocess_embed="none", train_aug_ratio = 1.0):
        seed_randomness(torch_module=torch, numpy_module=np, random_module=random)
        super().__init__()
        self.dataset = dataset
        self.classi_feature_extractor = classi_feature_extractor
        self.clip_model = clip_model
        
        self.target_clip_embs_dict = pickle.load(open(target_clip_embs_path.replace("clip_normalize", "none"), "rb"))
        self.spurious_clip_embs_dict = pickle.load(open(spurious_clip_embs_path.replace("clip_normalize", "none"), "rb"))
        
        self.inv_proj = inv_proj
        self.linear_layer = nn.Linear(classi_emb_dim, num_classes)
        self.text_train_aug = text_train_aug
        self.preprocess_embed = preprocess_embed
        self.train_aug_ratio = train_aug_ratio
        print(f"Preprocess embed: {self.preprocess_embed}")

        for p in self.inv_proj.parameters():
            p.requires_grad = False
        
        try:
            for p in self.classi_feature_extractor.parameters():
                p.requires_grad = False

            self.classi_feature_extractor.eval()
        except:
            pass
        self.inv_proj.eval()
        self.clip_model.eval()
    
    def forward(self, data, modality="text"):
        if modality == "text":
            self.inv_proj.cuda()
            self.linear_layer.cuda()

            target_embs = []
            spurious_embs = []
            if not self.text_train_aug:
                target_embs = [self.target_clip_embs_dict[target_word][text_target] for text_target, target_word in zip(data['text_target'], data['attributes']['target_name'])]
                spurious_embs = [self.spurious_clip_embs_dict[spurious_word][text_spurious] for text_spurious, spurious_word in zip(data['text_spurious'], data['attributes']['spurious_name'])]
            else:
                if self.train_aug_ratio < 1:
                    transform = torch.rand(len(data['text_target'])) > self.train_aug_ratio
                elif self.train_aug_ratio == 1:
                    transform = [True] * len(data['text_target'])

                random_idx = np.random.randint(0, len(openai_imagenet_template)-1, len(data['text_target']))

                target_embs = [self.target_clip_embs_dict[target_word][f"a photo of a {target_word}.".lower()] if not transform[i] else list(self.target_clip_embs_dict[target_word].values())[random_idx[i]] for i, target_word in enumerate(data['attributes']['target_name'])]
                spurious_embs = [self.spurious_clip_embs_dict[spurious_word][f"a photo of a {spurious_word}.".lower()] if not transform[i] else list(self.spurious_clip_embs_dict[spurious_word].values())[random_idx[i]] for i, spurious_word in enumerate(data['attributes']['spurious_name'])]
                    
            target_embs = torch.tensor(np.stack(target_embs)).type(torch.float32).cuda()
            spurious_embs = torch.tensor(np.stack(spurious_embs)).type(torch.float32).cuda()

            if self.preprocess_embed == "clip_normalize":
                target_embs = F.normalize(target_embs, dim=-1)
                spurious_embs = F.normalize(spurious_embs, dim=-1)

            emb = (target_embs + spurious_embs) / 2
            
            if self.preprocess_embed == "clip_normalize":
                emb = F.normalize(emb, dim=-1)

            emb = self.inv_proj(emb)
            return self.linear_layer(emb.squeeze())
        
        else:
            if data.shape[-1] == self.linear_layer.in_features:
                return self.linear_layer(data.float())
            else:
                try:
                    self.classi_feature_extractor.to(data.device)
                except:
                    pass
                classi_emb = self.classi_feature_extractor(data).float()
                return self.linear_layer(classi_emb)

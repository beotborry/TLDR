import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import clip
import random
from sklearn.linear_model import LogisticRegression
from datasets.text_template import openai_imagenet_template
from datasets.generate_text import TextDatasetGenerator
from utils.random_seed import seed_randomness

class RectModelAFR(nn.Module):
    def __init__(self, erm_model, classi_emb_dim, num_classes, weights):
        seed_randomness(torch_module=torch, numpy_module=np, random_module=random)
        super().__init__()
        self.classi_feature_extractor = erm_model.backbone
        self._classifier = erm_model.classifier

        self.classifier = nn.Linear(classi_emb_dim, num_classes)
        self.init_weights = (self._classifier.weight.detach().clone(), self._classifier.bias.detach().clone())
        
        self.classifier.weight.data = self.init_weights[0].clone()
        self.classifier.bias.data = self.init_weights[1].clone()

        self.classifier.train()

        for p in self.classi_feature_extractor.parameters():
            p.requires_grad = False

        self.classi_feature_extractor.eval()
        del self._classifier
    def forward(self, embs, modality=None):
        try:
            pred = self.classifier(embs)
        except:
            embs = self.classi_feature_extractor(embs)
            pred = self.classifier(embs)
        return pred
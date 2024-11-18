import itertools
import json
import numpy as np
import torch
import os
import clip
import torch.nn.functional as F
import random
import string
import pickle
import torch.nn as nn
from tqdm import tqdm
from datasets.text_template import waterbirds_template, celeba_template, openai_imagenet_template, spuco_animal_template, drml_waterbirds_template
from models.model_factory import model_factory
from models.projector import Projector
from utils.random_seed import seed_worker
from scipy.stats import false_discovery_control, mannwhitneyu, wilcoxon, ttest_rel
from datasets.dataset_factory import DatasetFactory
from sklearn.neighbors import KNeighborsClassifier
from utils.random_seed import seed_randomness, set_seed

class TextDatasetGenerator:
    def __init__(self, args, num_classes, num_groups=2) -> None:
        seed_randomness(random_module=random, torch_module=torch, numpy_module=np)
        self.args = args
        self.dataset = args.dataset
        self.num_classes = num_classes
        self.num_groups = num_groups
        self.prompt_type = args.prompt_type
        self.template = self.get_template()
        self.template['spurious']['0'] = [x.lower() for x in self.template['spurious']['0']]
        self.template['spurious']['1'] = [x.lower() for x in self.template['spurious']['1']]
        self.clip_model = self.load_clip_model()
        self.erm_model = self.load_erm_model() 
        self.projector = self.load_projector()
        ds_factory = DatasetFactory(dataset_name=args.dataset, root_dir=args.root_dir, augment=False, args=args)
        self.train_dataset, self.val_dataset, _ = ds_factory.get_dataset()
        self.gap = None


    def load_gap(self):
        if self.args.n_gap_estimates == 0:
            return None
        gap_path = os.path.join(self.args.log_dir, self.args.log_name + f"pe_{self.args.preprocess_embed}_train_gap_n_sample_{self.args.n_gap_estimates}.pt")
        gap = torch.load(gap_path)
        gap = gap.mean(dim=0)
        return gap

    def load_erm_model(self):
        erm_model = model_factory(
            arch = self.args.model,
            input_shape=(224, 224, 3),
            num_classes=self.num_classes,
            pretrained=True,
            backbone_freeze=self.args.backbone_freeze
        )
        self.classi_dim = erm_model.classifier.in_features
        erm_model_path = os.path.join(self.args.log_dir.replace("DFR+", ""), self.args.log_name + "_erm_model.pt")

        erm_model.load_state_dict(torch.load(erm_model_path))
        erm_model.cuda()
        erm_model.eval()
        return erm_model

    def load_projector(self):
        if self.args.model == "clip":
            return nn.Identity()
        
        proj_model = Projector(self.classi_dim, self.clip_model.visual.output_dim, self.args.proj_model, self.args.proj_n_layers, use_relu=self.args.model=="resnet50")
        proj_model_path = self.args.proj_save_path
        
        proj_model.load_state_dict(torch.load(proj_model_path))
        proj_model.cuda()
        proj_model.eval()
        return proj_model.inv_proj

    def load_clip_model(self):
        clip_model, _ = clip.load(self.args.clip_variants, device="cuda")
        clip_model.eval()
        return clip_model

    def load_img_embeddings(self):
        train_emb_save_path = self.args.train_emb_save_path
        val_emb_save_path = self.args.val_emb_save_path

        self.train_img_emb = torch.load(train_emb_save_path)[:, :self.erm_model.classifier.in_features]
        self.val_img_emb = torch.load(val_emb_save_path)[:, :self.erm_model.classifier.in_features]
        self.train_labels = self.train_dataset.labels
        self.val_labels = self.val_dataset.labels

        self.train_class_centroids = dict([(str(c), self.train_img_emb[self.train_labels == c].mean(dim=0).cuda()) for c in range(self.num_classes)])
        self.val_class_centroids = dict([(str(c), self.val_img_emb[self.val_labels == c].mean(dim=0).cuda()) for c in range(self.num_classes)])

    def get_label_fn(self, filtered_target, filtered_spurious):
        if self.dataset in ['waterbirds', 'celeba']:
            self.target_to_label = {x.lower(): int(key) \
                                    for key in filtered_target.keys() \
                                        for x in filtered_target[key]}

            self.spurious_to_label = {x.lower() : int(key) \
                                    for key in filtered_spurious.keys() \
                                        for x in filtered_spurious[key]}
        elif self.dataset == "spuco_animal":
            self.birds_target_to_label = {x.lower(): int(key) for key in ['0', '1'] for x in filtered_target[key]}
            self.dogs_target_to_label = {x.lower(): int(key) for key in ['2', '3'] for x in filtered_target[key]}

            self.birds_spurious_to_label = {x.lower(): int(key) for key in ['0', '1'] for x in filtered_spurious[key]}
            self.dogs_spurious_to_label = {x.lower(): int(key) for key in ['2', '3'] for x in filtered_spurious[key]}
            
    def get_prompt(self,):
        def prompt_fn(x: dict) -> list:
            return [(f"a photo of a {x['target']}.".lower(), f"a photo of a {x['spurious']}.".lower())]
        return prompt_fn
                             
    def get_template(self):
        if self.dataset == 'waterbirds':
            return waterbirds_template
        elif self.dataset == 'celeba':
            return celeba_template
        elif self.dataset == "spuco_animal":
            return spuco_animal_template
        
    def clip_semantic_filter(self, words, mode):
        def cossim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
        if mode == 'target':
            class_anchor = dict([list(self.template['target_class_anchor'].items())[int(key)] for key in words.keys()])
        elif mode == 'spurious':
            class_anchor = dict([list(self.template['spurious_class_anchor'].items())[int(key)] for key in words.keys()])

        prompts = {}
        # here key is '0', '1'
        for key in words.keys():
            prompts[key] = []
            prompts[key] = [(self.template['spurious_neutral_prompt'].format(x).lower(), x) for x in words[key]]
            
        filtered_words = {}
        with torch.no_grad():
            for key in prompts.keys():
                filtered_words[key] = []
                for pair in tqdm(prompts[key]):
                    prompt, word = pair
                    text = clip.tokenize(prompt).cuda()

                    text_features = self.clip_model.encode_text(text).type(torch.float32).cuda()
                    if self.args.preprocess_embed == "clip_normalize":
                        text_features = F.normalize(text_features, p=2, dim=-1)

                    cossims = []
                    for class_key in class_anchor.keys():
                        anchor = clip.tokenize(class_anchor[class_key]).cuda()

                        anchor_features = self.clip_model.encode_text(anchor).type(torch.float32).cuda()
                        if self.args.preprocess_embed == "clip_normalize":
                            anchor_features = F.normalize(anchor_features, p=2, dim=-1)
                        cos_sim = cossim(text_features.cpu().numpy().squeeze(), anchor_features.cpu().numpy().squeeze())
                
                        cossims.append(cos_sim)
                    cossims = np.array(cossims)

                    if np.argmax(cossims) == int(key) % 2: # % 2 for spuco_animal
                        filtered_words[key].append(word.lower()) 

        return filtered_words

    def target_word_filter(self, words : dict) -> dict:
        filtered_words = {}
        with torch.no_grad():
            for key in words.keys():
                label = int(key)
                filtered_words[key] = []
                for word in tqdm(words[key]):
                    prompt = self.template['spurious_neutral_prompt'].format(word).lower()
                    text = clip.tokenize(prompt).cuda()
                    text_features = self.clip_model.encode_text(text).type(torch.float32).cuda()
                    if self.args.preprocess_embed == "clip_normalize":
                        text_features = F.normalize(text_features, p=2, dim=-1)

                    text_features = self.projector(text_features)

                    logit = self.erm_model.classifier(text_features)
                    if logit.argmax(dim=-1) == label:
                        filtered_words[key].append(word)
                    
        return filtered_words
    
    def get_attributes_combinations(self, filtered_target, filtered_spurious):
        if self.dataset in ["waterbirds", "celeba"]:
            target = list(set(list(itertools.chain.from_iterable(filtered_target.values()))))
            spurious = list(set(list(itertools.chain.from_iterable(filtered_spurious.values()))))

            attributes = {
                'spurious': set([x.lower() for x in spurious]),
                'target': set([x.lower() for x in target]),
            }

            attributes_combinations = [dict(zip(attributes, x)) for x in sorted(itertools.product(*attributes.values()))]
        elif self.dataset == "spuco_animal":
            print([len(x) for x in filtered_target.values()])
            print([len(x) for x in filtered_spurious.values()])
            target_birds = list(set(list(filtered_target['0']))) + list(set(list(filtered_target['1'])))
            spurious_birds = list(set(list(filtered_spurious['0']))) + list(set(list(filtered_spurious['1'])))

            target_dogs = list(set(list(filtered_target['2']))) + list(set(list(filtered_target['3'])))
            spurious_dogs = list(set(list(filtered_spurious['2']))) + list(set(list(filtered_spurious['3'])))

            attributes_birds = {
                'spurious': set([x.lower() for x in spurious_birds]),
                'target': set([x.lower() for x in target_birds]),
            }

            attributes_dogs = {
                'spurious': set([x.lower() for x in spurious_dogs]),
                'target': set([x.lower() for x in target_dogs]),
            }

            birds_attributes_combinations = [dict(zip(attributes_birds, x)) for x in sorted(itertools.product(*attributes_birds.values()))]
            dogs_attributes_combinations = [dict(zip(attributes_dogs, x)) for x in sorted(itertools.product(*attributes_dogs.values()))]

            attributes_combinations = (birds_attributes_combinations, dogs_attributes_combinations)            
        return attributes_combinations
    
    def get_filtered_words(self,):
        if self.dataset in ["waterbirds", "celeba"]:
            filtered_target = self.clip_semantic_filter(self.template['target'], 'target')
            print("Seed: ", self.args.seed, f"Number of target words by key After target semantic filter: {[len(x) for x in filtered_target.values()]}")
        elif self.dataset == "spuco_animal":
            filtered_target_birds = self.clip_semantic_filter(dict(list(self.template['target'].items())[:2]), 'target')
            filtered_target_dogs = self.clip_semantic_filter(dict(list(self.template['target'].items())[2:]), 'target')
            filtered_target = {'0': filtered_target_birds['0'], '1': filtered_target_birds['1'], '2': filtered_target_dogs['2'], '3': filtered_target_dogs['3']}
            print("Seed: ", self.args.seed, f"Number of target words by key After target semantic filter: {[len(x) for x in filtered_target.values()]}")
            
        filtered_target = self.target_word_filter(filtered_target)
        print("Seed: ", self.args.seed, f"Number of target words by key After target logit filter: {[len(x) for x in filtered_target.values()]}")
            
        if self.dataset in ["waterbirds", "celeba"]:
            spurious = self.clip_semantic_filter(self.template['spurious'], 'spurious')
            print("Seed: ", self.args.seed, f"Number of spurious words by key After spurious semantic filter: {[len(x) for x in spurious.values()]}")
        elif self.dataset == "spuco_animal":
            spurious_birds = self.clip_semantic_filter(dict(list(self.template['spurious'].items())[:2]), 'spurious')
            spurious_dogs = self.clip_semantic_filter(dict(list(self.template['spurious'].items())[2:]), 'spurious')
            spurious = {'0': spurious_birds['0'], '1': spurious_birds['1'], '2': spurious_dogs['2'], '3': spurious_dogs['3']}
            print("Seed: ", self.args.seed, f"Number of spurious words by key After spurious semantic filter: {[len(x) for x in spurious.values()]}")

        return filtered_target, spurious
            
    def prepare_dataset(self, ):
        if not os.path.exists(self.args.target_filtered_words_path) and not os.path.exists(self.args.spurious_filtered_words_path):
            filtered_target, filtered_spurious = self.get_filtered_words()
            torch.save(filtered_target, self.args.target_filtered_words_path)
            torch.save(filtered_spurious, self.args.spurious_filtered_words_path)
        else:
            filtered_target = torch.load(self.args.target_filtered_words_path)
            filtered_spurious = torch.load(self.args.spurious_filtered_words_path)

        self.get_label_fn(filtered_target, filtered_spurious)

        print("Seed: ", self.args.seed, filtered_spurious)

        attributes_combinations = self.get_attributes_combinations(filtered_target, filtered_spurious)

        if self.dataset in ['waterbirds', 'celeba']:
            generator = self.get_prompt()
            text_train_data = [
                {
                    "text_target": text_target,
                    "text_spurious": text_spu,
                    "label": self.target_to_label[x["target"]],
                    "spurious": self.spurious_to_label[x["spurious"]],
                    "attributes": {
                        "target": self.target_to_label[x["target"]],
                        "spurious": self.spurious_to_label[x["spurious"]],
                        "target_name": x["target"],
                        "spurious_name": x["spurious"],
                    },
                }
                for x in attributes_combinations
                for text_target, text_spu in generator(x)
            ]
        elif self.dataset == 'spuco_animal':
            generator = self.get_prompt()
            birds_train_data = [
                {
                    "text_target": text_target,
                    "text_spurious": text_spu,
                    "label": self.birds_target_to_label[x["target"]],
                    "spurious": self.birds_spurious_to_label[x["spurious"]],
                    "attributes": {
                        "target": self.birds_target_to_label[x["target"]],
                        "spurious": self.birds_spurious_to_label[x["spurious"]],
                        "target_name": x["target"],
                        "spurious_name": x["spurious"],
                    },
                }
                for x in attributes_combinations[0]
                for text_target, text_spu in generator(x)
            ]

            dogs_train_data = [
                {
                    "text_target": text_target,
                    "text_spurious": text_spu,
                    "label": self.dogs_target_to_label[x["target"]],
                    "spurious": self.dogs_spurious_to_label[x["spurious"]],
                    "attributes": {
                        "target": self.dogs_target_to_label[x["target"]],
                        "spurious": self.dogs_spurious_to_label[x["spurious"]],
                        "target_name": x["target"],
                        "spurious_name": x["spurious"],
                    },
                }
                for x in attributes_combinations[1]
                for text_target, text_spu in generator(x)
            ]

            text_train_data = birds_train_data + dogs_train_data

        print(len(text_train_data))
        del self.clip_model
        del self.erm_model
        del self.projector
        return text_train_data

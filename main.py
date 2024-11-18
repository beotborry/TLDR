import torch
import random
import numpy as np
import pandas as pd
import wandb
import os
import torch.nn as nn
import clip

from utils.random_seed import set_seed, seed_randomness
from models.model_factory import model_factory
from datasets.dataset_factory import DatasetFactory
from arguments import get_args
from utils.get_optim_n_scheduler import get_optim_n_scheduler
from utils.logger import make_log_name, save_model, get_tldr_log_name, last_epoch_evaluation
from datasets.proj_dataset_wrapper import ProjDatasetWrapper
from evaluate.tldr_evaluator import TLDREvaluator
from models.projector import Projector
from last_layer_retrain.tldr import TLDR
from datasets.coco_dataset import CoCoDataset

def main(args):
    wandb.init(
        project=args.wandb_proj_name,
        config=args,
        name=args.date + f'_{args.exp_name}',
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.log_dir, args.log_name = make_log_name(args)
    args.erm_save_path, args.proj_save_path, args.rect_save_path, args.train_emb_save_path, args.val_emb_save_path, args.target_filtered_words_path, args.spurious_filtered_words_path, \
        args.target_words_clip_embs_path, args.spurious_words_clip_embs_path = get_tldr_log_name(args)
    ############################## Randomness ##################################
    set_seed(args.seed)
    seed_randomness(torch_module=torch, numpy_module=np, random_module=random)

    ############################## Dataset #####################################
    ds_factory = DatasetFactory(dataset_name=args.dataset, root_dir=args.root_dir, augment=False, args=args)

    train_dataset, val_dataset, test_dataset = ds_factory.get_dataset(clip_transform=False)

    proj_dataset_train = ProjDatasetWrapper(train_dataset)
    proj_dataset_val = ProjDatasetWrapper(val_dataset)

    coco_full = CoCoDataset(annotations_file='raw/captions_val2014.json', img_folder = 'validation/data', n_sample=args.n_gap_estimates * 2)
    coco_train, coco_val = torch.utils.data.random_split(coco_full, [args.n_gap_estimates, args.n_gap_estimates])
                
    ############################## Model #######################################
    model = model_factory(
        args.model, train_dataset[0][0].shape, train_dataset.num_classes, pretrained=args.pretrained, backbone_freeze=args.backbone_freeze
    ).to(device)
        
    feature_extractor = nn.Sequential(*list(model.children())[:-1]).eval()
    classi_emb_dim = feature_extractor(torch.randn(1, 3, 224, 224).cuda()).squeeze().shape[0]
    _clip_model, _ = clip.load(args.clip_variants, device="cuda")
    
    proj_model = Projector(classi_emb_dim, _clip_model.visual.output_dim, proj_model=args.proj_model, proj_n_layers=args.proj_n_layers, proj_activ=args.proj_activ, use_relu=args.model == "resnet50").to(device)
    del _clip_model

    ############################## Training ####################################
    optimizer, scheduler = get_optim_n_scheduler(model, args)
        
    erm_evaluator = TLDREvaluator(
        testset=val_dataset,
        group_partition=val_dataset.group_partition,
        group_weights=train_dataset.group_weights,
        batch_size=args.batch_size,
        model=model,
        device=device,
        verbose=True,
        mode="erm",
        classi_emb_dim=classi_emb_dim,
        clip_variants=args.clip_variants,
    )
    
    trainer = TLDR(
        model=model,
        proj_model=proj_model,
        erm_dataset=train_dataset,
        proj_dataset=proj_dataset_train,
        proj_val_dataset=proj_dataset_val,
        rect_val_dataset = val_dataset,
        scheduler=scheduler,
        device=device,
        verbose=True,
        args=args,
        erm_val_evaluator=erm_evaluator,
        erm_optimizer=optimizer,
        classi_emb_dim=classi_emb_dim,
        proj_gap_dataset=coco_train,
        proj_gap_val_dataset=coco_val,
    )

    trainer.train_all()

    ############################## Evaluation ##################################
    evaluator = last_epoch_evaluation(model, trainer, device, train_dataset, test_dataset, 'iid', args)
    results = pd.DataFrame(index=[0])
    results["timestamp"] = pd.Timestamp.now()
    results["seed"] = args.seed
    results["pretrained"] = args.pretrained
    results["lr"] = args.lr
    results["weight_decay"] = args.weight_decay
    results["momentum"] = args.momentum
    results["num_epochs"] = args.num_epochs
    results["batch_size"] = args.batch_size

    results["worst_group_accuracy"] = evaluator.worst_group_accuracy[1]
    results["average_accuracy"] = evaluator.average_accuracy

    for key in evaluator.accuracies.keys():
        results[f"{key}_accuracy"] = evaluator.accuracies[key]
        
    evaluator = last_epoch_evaluation(model, trainer, device, train_dataset, test_dataset, 'best_wga', args)

    results["early_stopping_worst_group_accuracy"] = evaluator.worst_group_accuracy[1]
    results["early_stopping_average_accuracy"] = evaluator.average_accuracy

    for key in evaluator.accuracies.keys():
        results[f"early_stopping_{key}_accuracy"] = evaluator.accuracies[key]

        evaluator = last_epoch_evaluation(model, trainer, device, train_dataset, test_dataset, 'best_cb', args)
        
        results["cb_early_stopping_worst_group_accuracy"] = evaluator.worst_group_accuracy[1]
        results["cb_early_stopping_average_accuracy"] = evaluator.average_accuracy
        
        for key in evaluator.accuracies.keys():
            results[f"cb_early_stopping_{key}_accuracy"] = evaluator.accuracies[key]

    # pd to dict
    results = results.to_dict(orient="records")[0]
    print(results)

    wandb.log(results)

    ############################## Save Model ##################################
    save_model(model, args, save_last=True, save_best=False)
    save_model(trainer.best_model, args, save_last=False, save_best=True)


if __name__ == "__main__":
    args = get_args()
    main(args)

import os
import torch
from datasets.spurious_target_dataset_wrapper import SpuriousTargetDatasetWrapper
from evaluate.evaluator import Evaluator
from evaluate.tldr_evaluator import TLDREvaluator

def make_log_dir(args):
    log_dir = os.path.join(
        "results",
        args.method,
        args.dataset,
        args.date,
        args.model if args.pretrained is True else f"{args.model}_scratch",
    )

    if os.path.exists(log_dir) is False:
        os.makedirs(log_dir)
    return log_dir

def make_log_name(args):
    log_dir = os.path.join(
        "results",
        args.method,
        args.dataset,
        args.date,
        args.model if args.pretrained is True else f"{args.model}_scratch",
    )

    if os.path.exists(log_dir) is False:
        os.makedirs(log_dir)

    log_name = f"optim_{args.optimizer}_sche_{args.scheduler}_seed_{args.seed}_lr_{args.lr}_wd_{args.weight_decay}_bs_{args.batch_size}_epochs_{args.num_epochs}_augment_{args.augment}"

    return log_dir, log_name


def get_tldr_log_name(args):
    erm_save_path = os.path.join(
            args.log_dir,
            args.log_name + "_erm_model.pt",
    )

    proj_save_path = os.path.join(
        os.path.join(args.log_dir, args.exp_name),
        args.log_name
        + f"model_{args.proj_model}_{args.proj_n_layers}_proj_wd_{args.proj_weight_decay}_pe_{args.preprocess_embed}_n_gap_{args.n_gap_estimates}_gap_ds_{args.gap_dataset}_proj_model.pt",
    )
        
    train_emb_save_path = os.path.join(
        args.log_dir,
        args.log_name + f"pe_{args.preprocess_embed}_train_emb.pt",
    )
    val_emb_save_path = os.path.join(
        args.log_dir,
        args.log_name + \
        f"pe_{args.preprocess_embed}_val_emb.pt",
    )
    rect_save_path = os.path.join(
        os.path.join(args.log_dir, args.exp_name),
        args.log_name + \
        f"_pe_{args.preprocess_embed}_prom_{args.prompt_type}_rect_lr_{args.rect_lr}_rect_epochs_{args.rect_num_epochs}_rect_bs_{args.rect_batch_size}",
    )

    rect_save_path += "_rect_model.pt"

    target_filtered_words_path = os.path.join(
        os.path.join(args.log_dir, args.exp_name),
        args.log_name + \
        f"_pe_{args.preprocess_embed}_prom_{args.prompt_type}_n_gap_{args.n_gap_estimates}_gap_ds_{args.gap_dataset}_target_filtered_words.pt",
    )

    spurious_filtered_words_path = os.path.join(
        os.path.join(args.log_dir, args.exp_name),
        args.log_name + \
        f"_pe_{args.preprocess_embed}_prom_{args.prompt_type}_n_gap_{args.n_gap_estimates}_gap_ds_{args.gap_dataset}_spurious_filtered_words.pt",
    )



    target_words_clip_embs_path = os.path.join(
        args.log_dir,
        f"{args.dataset}_pe_{args.preprocess_embed}_target_words_clip_embs.pkl"
    )

    spurious_words_clip_embs_path = os.path.join(
        args.log_dir,
        f"{args.dataset}_pe_{args.preprocess_embed}_spurious_words_clip_embs.pkl"
    )
        
    return erm_save_path, proj_save_path, rect_save_path, train_emb_save_path, val_emb_save_path, target_filtered_words_path, spurious_filtered_words_path, target_words_clip_embs_path, spurious_words_clip_embs_path
 

def save_model(model, args, save_best=False, save_last=False):
    pt_name = f"{args.method}_{args.log_name}"

    if save_best:
        pt_name += "_best"
    if save_last:
        pt_name += "_last"

    pt_name += ".pt"

    if not os.path.exists(os.path.join(args.log_dir, args.exp_name)):
        os.makedirs(os.path.join(args.log_dir, args.exp_name))

    save_path = os.path.join(os.path.join(args.log_dir, args.exp_name), pt_name)
    torch.save(model.state_dict(), save_path)
    
def last_epoch_evaluation(model, trainer, device, train_dataset, test_dataset, option, args):
    
    if option == 'iid':
        if args.method == "TLDR":
            evaluator = TLDREvaluator(
                testset=test_dataset,
                group_partition=test_dataset.group_partition,
                group_weights=train_dataset.group_weights,
                batch_size=64,
                model=trainer.rect_model,
                device=device,
                verbose=True,
                mode="eval",
                classi_emb_dim=2048,
                modality="image",
                clip_variants=args.clip_variants,
            )
        else:
            evaluator = Evaluator(
                testset=test_dataset,
                group_partition=test_dataset.group_partition,
                group_weights=train_dataset.group_weights,
                batch_size=64,
                model=model,
                device=device,
                trainer=trainer,
                verbose=True,
            )

    elif option == 'best_wga':
        if args.method == "TLDR":
            evaluator = TLDREvaluator(
                testset=test_dataset,
                group_partition=test_dataset.group_partition,
                group_weights=train_dataset.group_weights,
                batch_size=64,
                model=trainer.best_model,
                device=device,
                verbose=True,
                mode="eval",
                classi_emb_dim=2048,
                modality="image",
                clip_variants=args.clip_variants,
            )
        else:
            evaluator = Evaluator(
                testset=test_dataset,
                group_partition=test_dataset.group_partition,
                group_weights=train_dataset.group_weights,
                batch_size=args.batch_size,
                model=trainer.best_model,
                device=device,
                verbose=True,
                trainer=trainer,
            )
        
    elif option == "best_cb":
        if args.method == "TLDR":
            evaluator = TLDREvaluator(
                testset=test_dataset,
                group_partition=test_dataset.group_partition,
                group_weights=train_dataset.group_weights,
                batch_size=64,
                model=trainer.cb_best_model,
                device=device,
                verbose=True,
                mode="eval",
                classi_emb_dim=2048,
                modality="image",
                clip_variants=args.clip_variants,
            )
        else:
            evaluator = Evaluator(
                testset=test_dataset,
                group_partition=test_dataset.group_partition,
                group_weights=train_dataset.group_weights,
                batch_size=args.batch_size,
                model=trainer.cb_best_model,
                device=device,
                verbose=True,
                trainer=trainer,
            )

    evaluator.evaluate()    
    return evaluator
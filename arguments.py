import argparse


def get_args():
    parser = argparse.ArgumentParser(description="TLDR")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--date", type=str, default="2021-06-01", help="Date")
    parser.add_argument("--exp_name", type=str, default="test", help="Experiment name")
    parser.add_argument("--root_dir", type=str, default="./data", help="Root directory for datasets")
    parser.add_argument("--wandb_proj_name", type=str, required=True, help="Wandb project name")

    # Model related hyperparameters
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "vitb16"], help="Model to use")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained model")
    parser.add_argument("--backbone_freeze", action="store_true", help="Freeze backbone")

    # Optimizer related hyperparameters
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "AdamW", "Adam"], help="Optimizer to use")

    # Scheduler related hyperparameters
    parser.add_argument("--scheduler", type=str, default=None, choices=["cosine"], help="Scheduler to use")

    # Training related hyperparameters
    parser.add_argument(
        "--method", choices=["ERM", "TLDR"]
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="waterbirds",
        choices=["waterbirds", "spuco_animal", "celeba"],
        help="Dataset to use",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs")

    # temp : compare DFR error in spuco
    parser.add_argument("--augment", action="store_true", help="Use data augmentation")

    # CLIPREC related hyperparameters
    parser.add_argument("--proj_model", choices=['linear', 'mlp'], default='linear', help="Projection model")
    parser.add_argument("--proj_n_layers", default=1, type=int, help="Number of layers for projection model")
    parser.add_argument("--proj_activ", choices=['relu', 'leaky_relu'], default='relu', help="Projection activation")
    parser.add_argument("--proj_method", choices=['optim', 'analytic'], default='analytic', help="Projection method")
    parser.add_argument("--proj_weight_decay", default=1e-4, type=float, help="Weight decay for projection")
    parser.add_argument('--use_mean_gap', action='store_true', help='Use mean gap')
    parser.add_argument('--n_gap_estimates', type=int, default=1000, help='Number of gap estimates')
    parser.add_argument('--gap_dataset', type=str, default='coco_val', choices=['coco_val'])
    parser.add_argument(
        "--preprocess_embed",
        choices=["normalize", "scaling", "none", "clip_normalize"],
        default="none",
        help="Preprocessing method for embedding",
    )
    parser.add_argument('--proj_only', action='store_true', help='Only train projection')
    parser.add_argument("--rect_lr", default=1e-3, type=float, help="Learning rate for rectification")
    parser.add_argument("--rect_weight_decay", default=1e-4, type=float, help="Weight decay for rectification")
    parser.add_argument("--rect_num_epochs", default=300, type=int, help="Number of epochs for rectification")
    parser.add_argument("--rect_batch_size", default=64, type=int, help="Batch size for rectification")
    parser.add_argument(
        "--rect_optimizer",
        type=str,
        default="AdamW",
        choices=["SGD", "AdamW"],
        help="Optimizer to use for rectification",
    )
    parser.add_argument("--use_scheduler", action="store_true", help="Use scheduler", default=False)    
    

    # Temporary
    parser.add_argument("--prompt_type", type=str, default="prompt_sep", choices=["prompt_sep"], help="Prompt type")
    parser.add_argument("--text_train_aug", action="store_true", help="Use text train augmentation", default=False)
    parser.add_argument("--text_train_aug_ratio", type=float, default=1.0, help="Text train aug ratio")
    parser.add_argument("--llr_modality", choices=['image', 'text'], default='image', help="LLR modality")
    parser.add_argument("--clip_variants", choices=["ViT-B/32", "ViT-B/16", "ViT-L/14"], default="ViT-B/32", help="CLIP variants")
    
    args = parser.parse_args()
    return args

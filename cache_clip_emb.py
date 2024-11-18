import clip
import torch
from datasets.text_template import celeba_template, openai_imagenet_template, waterbirds_template, spuco_animal_template
import itertools
from tqdm import tqdm
from argparse import ArgumentParser
import pickle
import os
from utils.logger import make_log_dir

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['celeba', 'waterbirds', 'spuco_animal'], default='waterbirds')
    parser.add_argument('--target_embs_path', type=str)
    parser.add_argument('--spurious_embs_path', type=str)
    parser.add_argument('--method', type=str, default='TLDR')
    parser.add_argument('--date', type=str, default='Waterbirds_TLDR')
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "vitb16"], help="Model to use")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained model")

    args = parser.parse_args()
    
    log_dir = make_log_dir(args)

    clip_model, _ = clip.load("ViT-B/32", device="cuda")
    clip_model.eval()

    target_text_embs = {}
    spu_text_embs = {}

    if args.dataset == 'waterbirds':
        target_ensemble = itertools.product(list(set(waterbirds_template['target']['0'] + waterbirds_template['target']['1'])), openai_imagenet_template)
        spu_ensemble = itertools.product(list(set(waterbirds_template['spurious']['0'] + waterbirds_template['spurious']['1'])), openai_imagenet_template)
    elif args.dataset == 'celeba':
        target_ensemble = itertools.product(list(set(celeba_template['target']['0'] + celeba_template['target']['1'])), openai_imagenet_template)
        spu_ensemble = itertools.product(list(set(celeba_template['spurious']['0'] + celeba_template['spurious']['1'])), openai_imagenet_template)
    elif args.dataset == 'spuco_animal':
        target_ensemble = itertools.product(list(set(spuco_animal_template['target']['0'] + spuco_animal_template['target']['1'] + spuco_animal_template['target']['2'] + spuco_animal_template['target']['3'])), openai_imagenet_template)
        spu_ensemble = itertools.product(list(set(spuco_animal_template['spurious']['0'] + spuco_animal_template['spurious']['1'] + spuco_animal_template['spurious']['2'] + spuco_animal_template['spurious']['3'])), openai_imagenet_template)

    with torch.no_grad():
        for item in tqdm(target_ensemble):
            # break
            text = item[1](item[0]).lower()
            emb = clip.tokenize(text).cuda()
            emb = clip_model.encode_text(emb).detach().cpu().numpy()
            try:
                target_text_embs[item[0].lower()].append((text, emb))
            except:
                target_text_embs[item[0].lower()] = [(text,emb)]

    for key in target_text_embs.keys():
        target_text_embs[key] = dict(target_text_embs[key])

    pickle.dump(target_text_embs, open(os.path.join(log_dir, args.target_embs_path), 'wb'))

    with torch.no_grad():
        for item in tqdm(spu_ensemble):
            text = item[1](item[0]).lower()
            emb = clip.tokenize(text).cuda()
            emb = clip_model.encode_text(emb).detach().cpu().numpy()

            try:
                spu_text_embs[item[0].lower()].append((text, emb))
            except:
                spu_text_embs[item[0].lower()] = [(text, emb)]

    for key in spu_text_embs.keys():
        spu_text_embs[key] = dict(spu_text_embs[key])

    pickle.dump(spu_text_embs, open(os.path.join(log_dir, args.spurious_embs_path), 'wb'))
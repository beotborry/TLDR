# waterbirds
CUDA_VISIBLE_DEVICES=0 python3 cache_clip_emb.py --dataset waterbirds --date Waterbirds_TLDR --model resnet50 --pretrained --target_embs_path waterbirds_pe_none_target_words_clip_embs.pkl --spurious_embs_path waterbirds_pe_none_spurious_words_clip_embs.pkl


DEFAULT="--dataset waterbirds --wandb_proj_name TLDR_WACV --exp_name 240703_waterbirds --gap_dataset coco_val --model resnet50 --date Waterbirds_TLDR --use_mean_gap --pretrained --num_epochs 1 --method TLDR --optimizer SGD --lr 0.0001 --weight_decay 0.01 --augment --batch_size 32 --preprocess_embed none --proj_method analytic --proj_model linear"
rect_wd=1e-4
for rect_lr in 1e-4
do
    for i in 1000,50;
    do
        IFS=',' read n_gap lamb <<< "${i}"
        proj_wd=$lamb

        for aug_ratio in 1
        do
            CUDA_VISIBLE_DEVICES=0 python3 main.py $DEFAULT \
                --proj_weight_decay $proj_wd --proj_n_layers 1 --n_gap_estimates $n_gap \
                --rect_lr $rect_lr --rect_weight_decay $rect_wd  --rect_num_epochs 1 --rect_batch_size 128 --use_scheduler --seed 0 --prompt_type prompt_sep --text_train_aug --text_train_aug_ratio 1
        done
    done
done
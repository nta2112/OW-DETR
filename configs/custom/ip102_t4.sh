#!/usr/bin/env bash

# OW-DETR Task 4 Config (25 classes dataset - Task 4: 6 classes)
# Classes: 19..24 (PREV: 19, CUR: 6)

EXP_DIR="/kaggle/working/exps/ip102_t4"
PY_ARGS=${@:1}

torchrun --nproc_per_node=2 main_open_world.py \
    --output_dir ${EXP_DIR} \
    --dataset owod \
    --dec_layers 6 \
    --num_queries 100 \
    --batch_size 2 \
    --lr 0.0002 \
    --PREV_INTRODUCED_CLS 19 \
    --CUR_INTRODUCED_CLS 6 \
    --data_root 'data/IP102' \
    --train_set 't4_ft' \
    --val_set 'val' \
    --test_set 'test' \
    --num_classes 26 \
    --unmatched_boxes \
    --top_unk 5 \
    --featdim 1024 \
    --NC_branch \
    --backbone 'dino_resnet50' \
    --pretrain '/kaggle/working/exps/ip102_t3/best_checkpoint.pth' \
    --num_workers 4 \
    --eval_every 1 \
    --cache_mode \
    --nc_epoch 2 \
    --filter_pct 0.5 \
    --epochs 51 \
    ${PY_ARGS}

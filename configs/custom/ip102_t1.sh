#!/usr/bin/env bash

# OW-DETR Task 1 Config (25 classes dataset - Task 1: 7 classes)
# Classes: 0..6 (PREV: 0, CUR: 7)

EXP_DIR="/kaggle/working/exps/ip102_t1"
PY_ARGS=${@:1}

torchrun --nproc_per_node=2 main_open_world.py \
    --output_dir ${EXP_DIR} \
    --dataset owod \
    --dec_layers 6 \
    --num_queries 100 \
    --batch_size 2 \
    --PREV_INTRODUCED_CLS 0 \
    --CUR_INTRODUCED_CLS 7 \
    --data_root 'data/IP102' \
    --train_set 't1_train' \
    --val_set 'val' \
    --test_set 'test' \
    --num_classes 26 \
    --unmatched_boxes \
    --top_unk 5 \
    --featdim 1024 \
    --NC_branch \
    --backbone 'dino_resnet50' \
    --pretrain 'models/deformable_detr_coco_converted.pth' \
    --num_workers 4 \
    --eval_every 1 \
    --cache_mode \
    --epochs 51 \
    ${PY_ARGS}

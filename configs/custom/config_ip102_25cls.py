"""
Configuration module for running OW-DETR on modified IP102 dataset (25 classes, 4 tasks)

Kaggle Dataset Paths:
- Images: /kaggle/input/datasets/nta212/ip102-for-object-detection/VOC2007/JPEGImages
- Annotations: /kaggle/input/datasets/nta212/ip102-for-object-detection/VOC2007/Annotations
- COCO train.json: /kaggle/input/datasets/nta212/ip102-for-object-detection/train.json
"""

DATASET_CONFIG = {
    "num_classes": 26, # 25 known classes + 1 unknown token (idx 25)
    "total_dataset_classes": 25,
    "data_root": "data/IP102",
    "kaggle_img_dir": "/kaggle/input/datasets/nta212/ip102-for-object-detection/VOC2007/JPEGImages",
    "kaggle_xml_dir": "/kaggle/input/datasets/nta212/ip102-for-object-detection/VOC2007/Annotations",
    "kaggle_coco_train": "/kaggle/input/datasets/nta212/ip102-for-object-detection/train.json",
}

TASK_CONFIGS = {
    "task1": {
        "task_name": "ip102_t1",
        "prev_intro_cls": 0,
        "cur_intro_cls": 7,
        "class_range": list(range(0, 7)),
        "train_set": "t1_train",
        "val_set": "val",
        "test_set": "test",
        "epochs": 51,
        "batch_size": 2,
        "lr": 0.0002,
        "pretrain": "models/deformable_detr_coco_converted.pth",
        "output_dir": "/kaggle/working/exps/ip102_t1",
    },
    "task2": {
        "task_name": "ip102_t2",
        "prev_intro_cls": 7,
        "cur_intro_cls": 6,
        "class_range": list(range(7, 13)),
        "train_set": "t2_ft",
        "val_set": "val",
        "test_set": "test",
        "epochs": 51,
        "batch_size": 2,
        "lr": 0.0002,
        "nc_epoch": 2,
        "filter_pct": 0.5,
        "pretrain": "/kaggle/working/exps/ip102_t1/best_checkpoint.pth",
        "output_dir": "/kaggle/working/exps/ip102_t2",
    },
    "task3": {
        "task_name": "ip102_t3",
        "prev_intro_cls": 13,
        "cur_intro_cls": 6,
        "class_range": list(range(13, 19)),
        "train_set": "t3_ft",
        "val_set": "val",
        "test_set": "test",
        "epochs": 51,
        "batch_size": 2,
        "lr": 0.0002,
        "nc_epoch": 2,
        "filter_pct": 0.5,
        "pretrain": "/kaggle/working/exps/ip102_t2/best_checkpoint.pth",
        "output_dir": "/kaggle/working/exps/ip102_t3",
    },
    "task4": {
        "task_name": "ip102_t4",
        "prev_intro_cls": 19,
        "cur_intro_cls": 6,
        "class_range": list(range(19, 25)),
        "train_set": "t4_ft",
        "val_set": "val",
        "test_set": "test",
        "epochs": 51,
        "batch_size": 2,
        "lr": 0.0002,
        "nc_epoch": 2,
        "filter_pct": 0.5,
        "pretrain": "/kaggle/working/exps/ip102_t3/best_checkpoint.pth",
        "output_dir": "/kaggle/working/exps/ip102_t4",
    }
}

COMMON_MODEL_FLAGS = {
    "dataset": "owod",
    "dec_layers": 6,
    "num_queries": 100,
    "unmatched_boxes": True,
    "top_unk": 5,
    "featdim": 1024,
    "NC_branch": True,
    "backbone": "dino_resnet50",
    "num_workers": 4,
    "eval_every": 1,
    "cache_mode": True,
}

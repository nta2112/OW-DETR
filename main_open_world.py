# ------------------------------------------------------------------------
# OW-DETR: Open-world Detection Transformer
# Akshita Gupta^, Sanath Narayan^, K J Joseph, Salman Khan, Fahad Shahbaz Khan, Mubarak Shah
# https://arxiv.org/pdf/2112.01513.pdf
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from datasets.coco import make_coco_transforms
from datasets.torchvision_datasets.open_world import OWDetection
from engine import evaluate, train_one_epoch, viz
from models import build_model

# Đăng ký argparse.Namespace làm safe global để hỗ trợ torch.load trên PyTorch 2.6+
if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([argparse.Namespace])


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=51, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--sgd', action='store_true')
    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')
    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    # dataset parameters
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--eval_every', default=1, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    ## OWOD
    parser.add_argument('--PREV_INTRODUCED_CLS', default=0, type=int)
    parser.add_argument('--CUR_INTRODUCED_CLS', default=20, type=int)
    parser.add_argument('--top_unk', default=5, type=int)
    parser.add_argument('--unmatched_boxes', default=False, action='store_true')
    parser.add_argument('--featdim', default=1024, type=int)
    parser.add_argument('--pretrain', default='', help='initialized from the pre-training model')
    parser.add_argument('--train_set', default='', help='training txt files')
    parser.add_argument('--val_set', default='val', help='validation txt files')
    parser.add_argument('--test_set', default='', help='testing txt files')
    parser.add_argument('--NC_branch', default=False, action='store_true')
    parser.add_argument('--nc_loss_coef', default=2, type=float)
    parser.add_argument('--invalid_cls_logits', default=False, action='store_true', help='owod setting')
    parser.add_argument('--nc_epoch', default=0, type=int)
    parser.add_argument('--num_classes', default=81, type=int)
    parser.add_argument('--backbone', default='resnet50', type=str, help="Name of the convolutional backbone to use")
    parser.add_argument('--dataset', default='owod')
    parser.add_argument('--data_root', default='../data/OWDETR', type=str)
    parser.add_argument('--bbox_thresh', default=0.3, type=float)
    parser.add_argument('--filter_pct', default=-1.0, type=float, help='percentage of data to keep')
    return parser

def main(args):
    utils.init_distributed_mode(args)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    print(model_without_ddp)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_train, dataset_val, dataset_test = get_datasets(args)
    
    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
            sampler_test = samplers.NodeDistributedSampler(dataset_test, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
            sampler_test = samplers.DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)
    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                  drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                  pin_memory=True)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.dataset == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    elif args.dataset == "coco":
        base_ds = get_coco_api_from_dataset(dataset_test if args.eval else dataset_val)
    else:
        base_ds = dataset_test if args.eval else dataset_val

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        frozen_state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model_without_ddp.detr.load_state_dict(frozen_state_dict)

    output_dir = Path(args.output_dir)

    if args.pretrain:
        print('Initialized from the pre-training model')
        checkpoint = torch.load(args.pretrain, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        msg = model_without_ddp.load_state_dict(state_dict, strict=False)
        print(msg)

        if args.PREV_INTRODUCED_CLS > 0:
            print(f"Re-initializing class_embed weights and biases for classes {args.PREV_INTRODUCED_CLS} to {args.num_classes - 2}")
            import math
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            with torch.no_grad():
                for embed in model_without_ddp.class_embed:
                    # Reset bias for classes from PREV_INTRODUCED_CLS to num_classes - 2
                    embed.bias[args.PREV_INTRODUCED_CLS : args.num_classes - 1].fill_(bias_value)
                    # Reset weight for classes from PREV_INTRODUCED_CLS to num_classes - 2
                    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(embed.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    torch.nn.init.uniform_(embed.weight[args.PREV_INTRODUCED_CLS : args.num_classes - 1], -bound, bound)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        resume_state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(resume_state_dict, strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
        # check the resumed model
        if (not args.eval and not args.viz and args.dataset in ['coco', 'voc']):
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, args
            )
        if args.eval:
            test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_test, base_ds, device, args.output_dir, args)
            if args.output_dir:
                utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
            return

    if args.viz:
        viz(model, criterion, postprocessors, data_loader_test if args.eval else data_loader_val, base_ds, device, args.output_dir)
        return

    # Cấu hình Early Stopping
    best_known_map = -1.0
    no_improvement_epochs = 0
    patience = 2
    min_delta_pct = 1.03

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.nc_epoch, args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if args.epochs == 1:
                checkpoint_paths.append(output_dir / 'best_checkpoint.pth')
            # extra checkpoint before LR drop and every 5 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        # Chạy validation sau mỗi 1 epoch (Bỏ qua nếu chỉ chạy 1 epoch trong quá trình train)
        if args.dataset in ['owod'] and not (args.epochs == 1 and not args.eval):
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, args
            )
            
            # Tính Known mAP
            num_known_classes = args.PREV_INTRODUCED_CLS + args.CUR_INTRODUCED_CLS
            known_map = float(coco_evaluator.AP[:num_known_classes].mean())
            print(f"Epoch {epoch} - Known mAP: {known_map:.4f}% (Best: {max(0.0, best_known_map):.4f}%)")
            
            if best_known_map < 0:
                best_known_map = known_map
                no_improvement_epochs = 0
                if args.output_dir and known_map > 0.0:
                    print("Lưu checkpoint tốt nhất khởi tạo (best_checkpoint.pth)...")
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, output_dir / 'best_checkpoint.pth')
            else:
                # Nếu mAP còn nhỏ (< 1.0%), dùng độ tăng tuyệt đối (ít nhất 0.05% mAP) để tránh nhiễu ngẫu nhiên.
                # Khi mAP đã lớn (>= 1.0%), yêu cầu độ tăng tương đối (3%).
                if best_known_map < 1.0:
                    is_improved = (known_map >= best_known_map + 0.05)
                    improvement_detail = f"Đạt: {known_map:.4f}% >= Yêu cầu tăng tuyệt đối: {best_known_map + 0.05:.4f}%"
                else:
                    required_val = best_known_map * min_delta_pct
                    is_improved = (known_map >= required_val)
                    improvement_detail = f"Đạt: {known_map:.4f}% >= Yêu cầu tương đối: {required_val:.4f}%"

                if known_map > 0.0 and is_improved:
                    print(f"-> Known mAP cải thiện vượt trội ({improvement_detail})")
                    best_known_map = known_map
                    no_improvement_epochs = 0
                    if args.output_dir:
                        print("Lưu checkpoint tốt nhất mới (best_checkpoint.pth)...")
                        utils.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'args': args,
                        }, output_dir / 'best_checkpoint.pth')
                else:
                    no_improvement_epochs += 1
                    detail_req = f">= {best_known_map + 0.05:.4f}% (tuyệt đối)" if best_known_map < 1.0 else f">= {best_known_map * min_delta_pct:.4f}% (tương đối)"
                    print(f"-> Không cải thiện vượt trội (Yêu cầu {detail_req}, Đạt: {known_map:.4f}%). Số epoch liên tiếp không cải thiện: {no_improvement_epochs}/{patience}")
                    if no_improvement_epochs >= patience:
                        print("Early stopping triggered. Huấn luyện dừng lại do Known mAP không cải thiện sau 2 epoch liên tiếp.")
                        break
        else:
            test_stats = {}

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            if args.dataset in ['owod'] and epoch % args.eval_every == 0 and epoch > 0:
                # for evaluation logs
                if coco_evaluator is not None:
                    (output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    output_dir / "eval" / name)

    if args.output_dir and utils.is_main_process():
        best_ckpt_path = output_dir / 'best_checkpoint.pth'
        if not best_ckpt_path.exists():
            ckpt_path = output_dir / 'checkpoint.pth'
            if ckpt_path.exists():
                import shutil
                shutil.copy(ckpt_path, best_ckpt_path)
                print(f"Copying final checkpoint {ckpt_path} to {best_ckpt_path} as fallback.")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def get_datasets(args):
    print(args.dataset)
    if args.dataset == 'owod':
        train_set = args.train_set
        val_set = args.val_set
        test_set = args.test_set
        dataset_train = OWDetection(args, args.owod_path, ["2007"], image_sets=[args.train_set], transforms=make_coco_transforms(args.train_set), filter_pct=args.filter_pct)
        dataset_val = OWDetection(args, args.owod_path, ["2007"], image_sets=[args.val_set], transforms=make_coco_transforms(args.val_set), filter_pct=0.5)
        dataset_test = OWDetection(args, args.owod_path, ["2007"], image_sets=[args.test_set], transforms=make_coco_transforms(args.test_set), filter_pct=0.5)
    else:
        raise ValueError("Wrong dataset name")

    print(args.dataset)
    print(args.train_set)
    print(args.val_set)
    print(args.test_set)
    print(dataset_train)
    print(dataset_val)
    print(dataset_test)

    return dataset_train, dataset_val, dataset_test


def set_dataset_path(args):
    args.owod_path = os.path.join(args.data_root, 'VOC2007')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    set_dataset_path(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
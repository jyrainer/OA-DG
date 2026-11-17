#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import os.path as osp
import json
import copy
import argparse
from pathlib import Path
from typing import Optional

import mmcv
import numpy as np
import torch

from mmcv.image import tensor2imgs
from mmdet.datasets import build_dataset, build_dataloader

# 레포에서 사용하는 oamix 방식
oamix_config=dict(
    type='OAMix', version='augmix',
    num_views=1, keep_orig=False, severity=10,
    # multi-level transformation
    random_box_ratio=(3, 1/3), random_box_scale=(0.01, 0.1),
    # object-aware mixing # scales, ratios
    oa_random_box_scale=(0.005, 0.1), oa_random_box_ratio=(3, 1 / 3),
    spatial_ratio=4, sigma_ratio=0.3, # smoothing strategy to improve speed
)

def build_minimal_cfg(ann_file: str, img_root: str, samples_per_gpu: int, workers_per_gpu: int, class_names):
    """mmdet Config 없이 dict로 최소 학습 파이프라인 구성 (CocoDataset + OAMix 포함)"""
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', img_scale=[(1280, 600), (1280, 720)], keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        oamix_config,
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    ]

    data = dict(
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        train=dict(
            type='CocoDataset',
            ann_file=ann_file,
            img_prefix=img_root,
            classes=tuple(class_names),           # COCO json의 categories 사용
            pipeline=train_pipeline,
            test_mode=False,
        ),
    )

    cfg = mmcv.Config(
        dict(
            data=data,
            gpu_ids=[0],
            seed=0,
            log_level='INFO',
            optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
            optimizer_config=dict(grad_clip=None),
            runner=dict(type='EpochBasedRunner', max_epochs=1),
            # dummy but required by some builders
            work_dir='./work_dirs/_oamix_dump',
            checkpoint_config=dict(interval=0),
            log_config=dict(interval=50, hooks=[dict(type='TextLoggerHook')]),
        )
    )
    
    cfg.custom_imports = dict(imports=['mmdet.datasets.pipelines.oa_mix'], allow_failed_imports=False)

    return cfg


@torch.no_grad()
def run_dump(ann_file: str,
             img_root: str,
             out_root: str,
             split_name: str = 'train',
             limit: Optional[int] = None,
             samples_per_gpu: int = 1,
             workers_per_gpu: int = 2,
             keep_subdirs: bool = True,
             fname_suffix: str = '_oamix'):
    """파이프라인 결과(이미지+주석)를 저장"""
    out_root = Path(out_root)
    out_img_dir = out_root / 'images'
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_ann_path = out_root / f'{split_name}.json'

    # 원본 COCO의 categories를 그대로 유지
    with open(ann_file, 'r') as f:
        src = json.load(f)
    categories = src.get('categories', [])
    
    class_names = [c['name'] for c in sorted(categories, key=lambda x: x['id'])]

    cfg = build_minimal_cfg(ann_file, img_root, samples_per_gpu, workers_per_gpu, class_names)
    ds_cfg = copy.deepcopy(cfg.data.train)

    dataset = build_dataset(ds_cfg)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False
    )



    images_out = []
    annotations_out = []
    ann_id = 1
    img_id = 1

    saved = 0
    for batch in data_loader:
        imgs = batch['img'].data[0]            # Tensor [N,C,H,W]
        metas = batch['img_metas'].data[0]     # list of dict
        bboxes_list = batch.get('gt_bboxes', None)
        labels_list = batch.get('gt_labels', None)

        N = len(metas)
        assert len(imgs) == N

        for i in range(N):
            meta = metas[i]
            img_tensor = imgs[i].unsqueeze(0)  # [1,C,H,W]

            norm_cfg = meta.get('img_norm_cfg', dict(mean=[0,0,0], std=[1,1,1], to_rgb=False))
            img_np = tensor2imgs(img_tensor, **norm_cfg)[0]  # BGR uint8

            # 파일명/경로
            ori_filename = meta.get('ori_filename') or meta.get('filename', f'{img_id:012d}.png')
            ori_filename = str(ori_filename).lstrip('/')
            stem, ext = osp.splitext(ori_filename)
            if ext.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                ext = '.png'
            if not keep_subdirs:
                stem = osp.basename(stem)

            save_rel = f'{stem}{fname_suffix}{ext}'
            save_abs = out_img_dir / save_rel
            save_abs.parent.mkdir(parents=True, exist_ok=True)

            # 저장 (mmcv.imwrite expects BGR)
            mmcv.imwrite(img_np, str(save_abs))

            # COCO images 항목
            H, W = img_np.shape[:2]
            images_out.append({
                'id': img_id,
                'file_name': save_rel,
                'height': H,
                'width': W,
            })

            # COCO annotations 항목
            if bboxes_list is not None and labels_list is not None:
                # mmdet batch 포맷: DataContainer -> Tensor
                bboxes = bboxes_list.data[0][i].cpu().numpy() if torch.is_tensor(bboxes_list.data[0][i]) else np.asarray(bboxes_list.data[0][i])
                labels = labels_list.data[0][i].cpu().numpy() if torch.is_tensor(labels_list.data[0][i]) else np.asarray(labels_list.data[0][i])
                # bboxes: (num, 4) in xyxy -> COCO xywh
                print(labels)
                for bb, cls in zip(bboxes, labels):
                    x1, y1, x2, y2 = bb.tolist()
                    w = max(0.0, float(x2 - x1))
                    h = max(0.0, float(y2 - y1))
                    x = float(x1)
                    y = float(y1)
                    area = w * h
                    annotations_out.append({
                        'id': ann_id,
                        'image_id': img_id,
                        'category_id': int(cls) + 1,
                        'bbox': [x, y, w, h],
                        'area': float(area),
                        'iscrowd': 0,
                        'segmentation': []
                    })
                    ann_id += 1

            img_id += 1
            saved += 1
            if limit is not None and saved >= limit:
                break

        if limit is not None and saved >= limit:
            break

    # COCO json 덤프
    coco_out = {
        'images': images_out,
        'annotations': annotations_out,
        'categories': categories
    }
    with open(out_ann_path, 'w') as f:
        json.dump(coco_out, f)

    print(f'[Done] Saved {len(images_out)} images -> {out_img_dir}')
    print(f'[Done] COCO annotations -> {out_ann_path}')


def parse_args():
    p = argparse.ArgumentParser(description='Dump OA-Mix processed dataset (images + COCO annotations)')
    p.add_argument('--ann-file', required=True, help='COCO annotation (train.json or val.json)')
    p.add_argument('--img-root', required=True, help='images/ root dir')
    p.add_argument('--out-root', required=True, help='output root dir (will create images/ and split.json)')
    p.add_argument('--split-name', default='train', help='name for output json (train|val|custom)')
    p.add_argument('--limit', type=int, default=None, help='max samples to dump')
    p.add_argument('--samples-per-gpu', type=int, default=1)
    p.add_argument('--workers-per-gpu', type=int, default=2)
    p.add_argument('--no-keep-subdirs', action='store_true', help='flatten subdirs in output file paths')
    p.add_argument('--suffix', default='_oamix', help='filename suffix for output images')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_dump(
        ann_file=args.ann_file,
        img_root=args.img_root,
        out_root=args.out_root,
        split_name=args.split_name,
        limit=args.limit,
        samples_per_gpu=args.samples_per_gpu,
        workers_per_gpu=args.workers_per_gpu,
        keep_subdirs=not args.no_keep_subdirs,
        fname_suffix=args.suffix
    )

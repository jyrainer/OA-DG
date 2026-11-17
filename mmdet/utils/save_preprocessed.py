# utils/save_preprocessed.py
from pathlib import Path
import os
import os.path as osp
import copy
from typing import Optional, Literal, Dict, Any, List

import mmcv
from mmcv.image import tensor2imgs
import torch
import numpy as np

from mmdet.datasets import build_dataloader, build_dataset


@torch.no_grad()
def save_preprocessed_images(
    cfg,
    out_dir: str,
    split: Literal["train", "val", "test"] = "train",
    samples_per_gpu: Optional[int] = None,
    workers_per_gpu: Optional[int] = None,
    limit: Optional[int] = None,
    keep_subdirs_from_ori: bool = True,
    filename_suffix: str = "",
) -> None:
    """
    mmdet Config의 데이터 파이프라인(예: OA-Mix 포함)을 그대로 태워서,
    '전처리된 텐서'를 디노멀라이즈하고 이미지를 파일로 저장한다.

    Args:
        cfg: mmcv.Config (이미 cfg = mmcv.Config.fromfile(...)로 로드된 객체)
        out_dir: 저장할 최상위 디렉토리
        split: 'train' | 'val' | 'test' 중 대상 split
        samples_per_gpu: 배치 크기(미지정 시 cfg.data.*.samples_per_gpu 사용)
        workers_per_gpu: 워커 수(미지정 시 cfg.data.workers_per_gpu 사용)
        limit: 저장 개수 제한(디버깅용)
        keep_subdirs_from_ori: 원본 파일 경로의 서브디렉터리 구조를 유지할지 여부
        filename_suffix: 파일명 뒤에 붙일 접미사(예: "_oamix")
    """
    assert split in ["train", "val", "test"]
    cfg = copy.deepcopy(cfg)

    # datacfg 선택
    data_cfg = getattr(cfg.data, split)

    # 배치/워커 설정 (미지정 시 cfg 값을 사용)
    spg = samples_per_gpu if samples_per_gpu is not None else data_cfg.get("samples_per_gpu", 1)
    wpg = workers_per_gpu if workers_per_gpu is not None else cfg.data.get("workers_per_gpu", 2)

    # test_mode 주의: 학습 파이프라인을 그대로 태울 때는 False가 일반적
    # (OA-Mix 등 학습 전용 변환이 test_mode=True 에선 비활성화일 수 있음)
    if isinstance(data_cfg, dict):
        data_cfg = copy.deepcopy(data_cfg)
        data_cfg["test_mode"] = False

    dataset = build_dataset(data_cfg)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=spg,
        workers_per_gpu=wpg,
        dist=False,
        shuffle=False
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for batch_idx, data in enumerate(data_loader):
        # 배치 텐서 및 메타
        # - mmdet의 DataContainer로 감싸져 있음
        imgs = data["img"].data[0]            # Tensor [N, C, H, W]
        img_metas = data["img_metas"].data[0] # List[dict], 길이 N

        # img_norm_cfg에서 mean/std/BGR 여부 가져와서 복원
        # tensor2imgs는 배치 텐서를 받아 리스트[np.ndarray(H,W,C)]로 변환
        # (디폴트는 BGR uint8)
        assert len(imgs) == len(img_metas)
        for i in range(len(imgs)):
            meta = img_metas[i]
            img_tensor = imgs[i].unsqueeze(0)  # [1,C,H,W]
            # mmdet/ mmcv가 기대하는 키 포함 여부 확인
            norm_cfg = meta.get("img_norm_cfg", dict(mean=[0,0,0], std=[1,1,1], to_rgb=False))

            # 복원
            img_list = tensor2imgs(img_tensor, **norm_cfg)  # -> List[np.uint8 HxWxC] (BGR)
            restored = img_list[0]

            # 파일명 구성
            # ori_filename 이 있으면 그걸 활용(하위 폴더 구조 유지 옵션)
            # 없으면 filename 사용
            ori_filename = meta.get("ori_filename", None) or meta.get("filename", f"{batch_idx:06d}_{i:02d}.png")
            ori_filename = str(ori_filename)

            if keep_subdirs_from_ori:
                rel_path = ori_filename
            else:
                rel_path = osp.basename(ori_filename)

            rootless = rel_path.lstrip("/")

            # 확장자/디렉터리 정리
            stem, ext = osp.splitext(rootless)
            if ext.lower() not in [".png", ".jpg", ".jpeg", ".bmp"]:
                ext = ".png"  # 안전하게 png로 저장

            # 최종 경로
            save_path = out_dir / f"{stem}{filename_suffix}{ext}"
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # 저장 (mmcv.imwrite는 BGR ndarray를 그대로 저장)
            mmcv.imwrite(restored, str(save_path))
            saved += 1

            # limit에 도달하면 조기 종료
            if limit is not None and saved >= limit:
                return

    return


if __name__ == "__main__":
    import mmcv

    cfg = mmcv.Config.fromfile("configs/your_coco_cfg.py")

    # 예) OA-Mix가 train 파이프라인에 들어 있다면,
    #     해당 파이프라인을 통과한 결과물(=mix된 결과)을 그대로 저장
    save_preprocessed_images(
        cfg,
        out_dir="/tmp/OAMixDump/train",
        split="train",
        samples_per_gpu=1,
        workers_per_gpu=2,
        limit=None,                 # 필요시 100 같은 숫자로 제한
        keep_subdirs_from_ori=True, # 원본의 하위 폴더 구조 유지
        filename_suffix="_oamix"    # 파일명 뒤에 _oamix 붙이기
    )

# src/datasets/frame_selector_dataset.py
import os
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# -------- Video backend (prefer decord) --------
_HAS_DECORD = False
try:
    from decord import VideoReader, cpu  # type: ignore
    import decord  # type: ignore
    decord.bridge.set_bridge("torch")
    _HAS_DECORD = True
except Exception:
    _HAS_DECORD = False

try:
    from torchvision.io import read_video
    _HAS_TV_READ = True
except Exception:
    _HAS_TV_READ = False

from transformers import AutoTokenizer


# =========================================================
# Video helpers (same spirit as your existing dataset)
# =========================================================

def _safe_clamp_indices(idxs: torch.Tensor, n_frames: int) -> torch.Tensor:
    return idxs.clamp(min=0, max=max(n_frames - 1, 0))


def _read_video_frames(path: str, indices: torch.Tensor) -> torch.Tensor:
    """
    Read frames at `indices` from video.
    Returns uint8 tensor [T,H,W,C].
    """
    if _HAS_DECORD:
        vr = VideoReader(path, ctx=cpu(0))
        frames = vr.get_batch(indices.tolist())
        return frames
    if not _HAS_TV_READ:
        raise RuntimeError("No video backend. Install decord or torchvision video support.")
    v, _, _ = read_video(path, pts_unit="sec")  # [Tv,H,W,C], uint8
    idxs = _safe_clamp_indices(indices, v.shape[0])
    return v[idxs]


def _get_video_meta(path: str) -> Tuple[int, float]:
    """
    Return (n_total_frames, fps).
    """
    if _HAS_DECORD:
        vr = VideoReader(path, ctx=cpu(0))
        fps = float(vr.get_avg_fps())
        if fps <= 0:
            fps = 30.0
        return len(vr), fps

    if not _HAS_TV_READ:
        raise RuntimeError("No video backend. Install decord or torchvision video support.")
    v, _, info = read_video(path, pts_unit="sec")
    n_total = v.shape[0]
    fps = float(info.get("video_fps", 30.0)) if isinstance(info, dict) else 30.0
    if fps <= 0:
        fps = 30.0
    return n_total, fps


def _preprocess_frames(
    frames_uint8_thwc: torch.Tensor,
    img_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """
    Input : uint8 [T,H,W,C]
    Output: float [T,C,H,W] normalized + resized
    """
    x = frames_uint8_thwc
    if x.dtype != torch.uint8:
        x = x.to(torch.uint8)

    x = x.float() / 255.0               # [T,H,W,C]
    x = x.permute(0, 3, 1, 2).contiguous()  # [T,C,H,W]

    x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)

    mean_t = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std_t  = torch.tensor(std,  dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    x = (x - mean_t) / std_t
    return x


# =========================================================
# Batch dataclass
# =========================================================

@dataclass
class FrameSelectorBatch:
    video: torch.Tensor              # [B, N, C, H, W]
    input_ids: torch.Tensor          # [B, L]
    attention_mask: torch.Tensor     # [B, L]
    labels: torch.Tensor             # [B, N]  (float: 0/1 or soft)
    ids: List[str]
    video_paths: List[str]
    sample_indices: torch.Tensor     # [B, N]  (frame indices in original video, for debug/infer)
    fps: torch.Tensor                # [B]


# =========================================================
# Dataset (Stage-1 localizer)
# =========================================================

class FrameSelectorDataset(Dataset):
    """
    Stage-1: Learn to localize support time from (video, question).

    Strategy:
    - Sample N frames uniformly over whole video (downsample).
    - Build time-based label from support_frames (seconds).
      Default label is "gaussian + ensure nearest positive".
    """

    def __init__(
        self,
        json_path: str,
        root_dir: str,
        tokenizer_name: str = "vinai/phobert-base",
        split: str = "train",
        num_samples: int = 96,          # N (recommend 64~96)
        img_size: int = 224,
        max_len: int = 64,

        # label controls
        label_mode: str = "gaussian",   # "gaussian" or "binary"
        window_sec: float = 1.0,        # used if label_mode="binary"
        sigma_sec: float = 0.6,         # used if label_mode="gaussian"
        ensure_nearest_positive: bool = True,

        seed: int = 42,
    ):
        assert split in ("train", "test")
        assert label_mode in ("gaussian", "binary")

        self.root_dir = root_dir
        self.split = split
        self.num_samples = num_samples
        self.img_size = img_size
        self.max_len = max_len

        self.label_mode = label_mode
        self.window_sec = window_sec
        self.sigma_sec = sigma_sec
        self.ensure_nearest_positive = ensure_nearest_positive

        self.rng = random.Random(seed)

        with open(json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        self.data = obj["data"] if isinstance(obj, dict) and "data" in obj else obj

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)

    def __len__(self):
        return len(self.data)

    def _uniform_sample_indices(self, n_total: int, N: int) -> torch.Tensor:
        """
        Uniformly sample N indices in [0, n_total-1]. Returns [N] long.
        Safe when n_total < N (will duplicate due to linspace->long).
        """
        if n_total <= 1:
            return torch.zeros(N, dtype=torch.long)
        return torch.linspace(0, n_total - 1, steps=N).long()

    def _build_labels(
        self,
        sample_indices: torch.Tensor,   # [N], frame idx in original video
        fps: float,
        support_frames_sec: Optional[List[float]],
    ) -> torch.Tensor:
        """
        Build labels over sampled timesteps [N].
        Returns float [N] (0/1 or soft in [0,1]).
        """
        N = sample_indices.numel()
        y = torch.zeros(N, dtype=torch.float32)

        if not support_frames_sec:
            return y

        # convert sample indices to time (seconds)
        sample_times = sample_indices.float() / max(fps, 1e-6)  # [N]

        for t_sec in support_frames_sec:
            t_sec = float(t_sec)

            if self.label_mode == "binary":
                mask = (sample_times - t_sec).abs() <= self.window_sec
                y[mask] = 1.0
            else:
                # gaussian soft target
                sigma = max(self.sigma_sec, 1e-6)
                soft = torch.exp(-0.5 * ((sample_times - t_sec) / sigma) ** 2)  # [N]
                y = torch.maximum(y, soft)

            if self.ensure_nearest_positive:
                nearest = torch.argmin((sample_times - t_sec).abs())
                y[nearest] = 1.0

        return y

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        qid = item.get("id", str(idx))
        rel_video_path = item["video_path"]
        video_path = os.path.join(self.root_dir, rel_video_path)

        n_total, fps = _get_video_meta(video_path)

        # 1) sample N frames over whole video
        sample_indices = self._uniform_sample_indices(n_total, self.num_samples)  # [N]
        frames = _read_video_frames(video_path, sample_indices)                    # [N,H,W,C]
        video = _preprocess_frames(frames, img_size=self.img_size)                 # [N,C,H,W]

        # 2) labels from support_frames (seconds)
        support_list = None
        if self.split == "train":
            support_list = item.get("support_frames", None)
            if support_list is not None and not isinstance(support_list, list):
                support_list = None

        labels = self._build_labels(sample_indices, fps, support_list)             # [N]

        # 3) text encode: QUESTION ONLY (stage-1 localizer)
        enc = self.tokenizer(
            item["question"],
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_attention_mask=True,
        )

        return {
            "id": qid,
            "video": video,                       # [N,C,H,W]
            "enc": enc,                           # dict
            "labels": labels,                     # [N]
            "video_path": rel_video_path,
            "sample_indices": sample_indices,     # [N]
            "fps": float(fps),
        }


# =========================================================
# Collator
# =========================================================

class FrameSelectorCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, Any]]) -> FrameSelectorBatch:
        videos = torch.stack([b["video"] for b in batch], dim=0)           # [B,N,C,H,W]
        labels = torch.stack([b["labels"] for b in batch], dim=0)          # [B,N]
        sample_indices = torch.stack([b["sample_indices"] for b in batch], dim=0)  # [B,N]
        fps = torch.tensor([b["fps"] for b in batch], dtype=torch.float32) # [B]

        padded = self.tokenizer.pad(
            [b["enc"] for b in batch],
            padding=True,
            return_tensors="pt",
        )  # input_ids [B,L], attention_mask [B,L]

        return FrameSelectorBatch(
            video=videos,
            input_ids=padded["input_ids"],
            attention_mask=padded["attention_mask"],
            labels=labels,
            ids=[b["id"] for b in batch],
            video_paths=[b["video_path"] for b in batch],
            sample_indices=sample_indices,
            fps=fps,
        )


# =========================================================
# Loader builder
# =========================================================

def build_frame_selector_loader(
    root_dir: str,
    train_json: str,
    tokenizer_name: str = "vinai/phobert-base",
    batch_size: int = 4,
    num_workers: int = 0,  # Windows debug: 0
    num_samples: int = 96,
    img_size: int = 224,
    max_len: int = 64,
    label_mode: str = "gaussian",
    window_sec: float = 1.0,
    sigma_sec: float = 0.6,
    ensure_nearest_positive: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, AutoTokenizer]:
    ds = FrameSelectorDataset(
        json_path=train_json,
        root_dir=root_dir,
        tokenizer_name=tokenizer_name,
        split="train",
        num_samples=num_samples,
        img_size=img_size,
        max_len=max_len,
        label_mode=label_mode,
        window_sec=window_sec,
        sigma_sec=sigma_sec,
        ensure_nearest_positive=ensure_nearest_positive,
        seed=seed,
    )
    collate = FrameSelectorCollator(ds.tokenizer)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    return loader, ds.tokenizer


# =========================================================
# Quick sanity check
# =========================================================
if __name__ == "__main__":
    root_dir = r"F:\zaloaiproject\dataset\traffic_buddy_train+public_test"
    train_json = os.path.join(root_dir, "train", "train.json")

    loader, tokenizer = build_frame_selector_loader(
        root_dir=root_dir,
        train_json=train_json,
        batch_size=2,
        num_workers=0,
        num_samples=96,
        img_size=224,
        max_len=64,
        label_mode="gaussian",  # recommended
        sigma_sec=0.6,
        ensure_nearest_positive=True,
    )

    batch = next(iter(loader))
    print("video:", batch.video.shape)  # [B,N,C,H,W]
    print("input_ids:", batch.input_ids.shape)
    print("labels:", batch.labels.shape, "pos_count:", (batch.labels > 0.5).sum().item())
    print("sample_indices:", batch.sample_indices.shape)
    print("fps:", batch.fps)

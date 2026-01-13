# src/datasets/trafficbuddy_dataset.py
import os
import json
import random
import re
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

# -------- Text tokenizer --------
# pip install transformers sentencepiece
from transformers import AutoTokenizer


# =========================================================
# Utilities
# =========================================================
def strip_choice_prefix(s: str) -> str:
    """
    Remove prefixes like 'A. ', 'B. ', 'C. ', 'D. ' if present.
    Keep content.
    """
    s = s.strip()
    return re.sub(r"^[A-Da-d]\s*[\.\)\:\-]\s*", "", s).strip()


def answer_to_index(answer: str, choices: List[str]) -> int:
    """
    Robust mapping answer -> index.
    - If answer exactly matches a choice string => use that index
    - Else if answer begins with 'A'/'B'/'C'/'D' => map letter
    - Else match after stripping prefixes
    """
    answer = answer.strip()
    if answer in choices:
        return choices.index(answer)

    m = re.match(r"^([A-Da-d])\b", answer)
    if m:
        letter = m.group(1).upper()
        idx = ord(letter) - ord("A")
        if 0 <= idx < len(choices):
            return idx

    ans_stripped = strip_choice_prefix(answer)
    choices_stripped = [strip_choice_prefix(c) for c in choices]
    if ans_stripped in choices_stripped:
        return choices_stripped.index(ans_stripped)

    raise ValueError(f"Cannot map answer='{answer}' to choices={choices}")


def _safe_clamp_indices(idxs: torch.Tensor, n_frames: int) -> torch.Tensor:
    return idxs.clamp(min=0, max=max(n_frames - 1, 0))


def _make_clip_indices_centered(
    center: int,
    num_frames: int,
    frame_stride: int,
    n_total: int,
) -> torch.Tensor:
    """
    Create ordered indices for a clip centered at `center`.
    Returns shape [num_frames]
    """
    span = (num_frames - 1) * frame_stride + 1
    start = center - span // 2
    idxs = torch.arange(num_frames, dtype=torch.long) * frame_stride + start
    return _safe_clamp_indices(idxs, n_total)


def _read_video_frames(path: str, indices: torch.Tensor) -> torch.Tensor:
    """
    Read frames at `indices` from video at `path`.
    Returns uint8 tensor [T, H, W, C] (C=3).
    """
    if _HAS_DECORD:
        vr = VideoReader(path, ctx=cpu(0))
        frames = vr.get_batch(indices.tolist())  # torch uint8 [T,H,W,C]
        return frames
    else:
        if not _HAS_TV_READ:
            raise RuntimeError("No video backend available. Install decord or torchvision with video support.")
        v, _, _ = read_video(path, pts_unit="sec")  # [Tv, H, W, C], uint8
        n_total = v.shape[0]
        idxs = _safe_clamp_indices(indices, n_total)
        return v[idxs]


def _get_video_meta(path: str) -> Tuple[int, float]:
    """
    Return (n_total_frames, fps).
    """
    if _HAS_DECORD:
        vr = VideoReader(path, ctx=cpu(0))
        n_total = len(vr)
        fps = float(vr.get_avg_fps())
        if fps <= 0:
            fps = 30.0
        return n_total, fps
    else:
        if not _HAS_TV_READ:
            raise RuntimeError("No video backend available. Install decord or torchvision with video support.")
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
    Input: uint8 [T,H,W,C]
    Output: float [T,C,H,W] normalized + resized
    """
    x = frames_uint8_thwc
    if x.dtype != torch.uint8:
        x = x.to(torch.uint8)

    x = x.float() / 255.0
    x = x.permute(0, 3, 1, 2).contiguous()  # [T,C,H,W]
    x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)

    mean_t = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x - mean_t) / std_t


# =========================================================
# Batch dataclass
# =========================================================
@dataclass
class VideoTextBatch:
    video: torch.Tensor              # [B, T, C, H, W]
    input_ids: torch.Tensor          # [B, Cmax, L]
    attention_mask: torch.Tensor     # [B, Cmax, L]
    choice_mask: torch.Tensor        # [B, Cmax] True=real choice, False=pad choice
    labels: Optional[torch.Tensor]   # [B] or None
    ids: List[str]
    video_paths: List[str]


# =========================================================
# Dataset
# =========================================================
class TrafficBuddyDataset(Dataset):
    """
    Stage-2 / Video Swin ready (single clip):
      - returns ONE clip per sample: video [T,C,H,W]
      - center is from support_frames (train) or middle (test)
      - text returns enc_list: variable number of choices per sample
    """

    def __init__(
        self,
        json_path: str,
        root_dir: str,
        tokenizer_name: str = "vinai/phobert-base",
        split: str = "train",  # "train" or "test"
        num_frames: int = 16,  # T
        frame_stride: int = 2,
        img_size: int = 224,
        max_len: int = 128,
        seed: int = 42,
    ):
        assert split in ("train", "test")
        self.json_path = json_path
        self.root_dir = root_dir
        self.split = split

        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.img_size = img_size
        self.max_len = max_len

        self.rng = random.Random(seed)

        with open(json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        self.data = obj["data"] if isinstance(obj, dict) and "data" in obj else obj

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)

    def __len__(self) -> int:
        return len(self.data)

    def _sample_clip_indices(self, video_path: str, item: Dict[str, Any]) -> torch.Tensor:
        """
        Return indices for ONE clip: shape [T]
        """
        n_total, fps = _get_video_meta(video_path)
        T = self.num_frames
        S = self.frame_stride

        if self.split == "train":
            has_support = (
                "support_frames" in item
                and isinstance(item["support_frames"], list)
                and len(item["support_frames"]) > 0
            )
            if has_support:
                t_sec = float(self.rng.choice(item["support_frames"]))
                center = int(round(t_sec * fps))
            else:
                center = self.rng.randint(0, max(n_total - 1, 0))
        else:
            center = n_total // 2 if n_total > 0 else 0

        return _make_clip_indices_centered(center, T, S, n_total)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]

        qid = item["id"]
        question = item["question"]
        choices = item["choices"]
        rel_video_path = item["video_path"]
        video_path = os.path.join(self.root_dir, rel_video_path)

        # ---- text tokenize: (question, choice_i) ----
        enc_list = []
        for c in choices:
            c_clean = strip_choice_prefix(c)
            enc = self.tokenizer(
                question,
                c_clean,
                truncation=True,
                max_length=self.max_len,
                padding=False,
                return_attention_mask=True,
            )
            enc_list.append(enc)

        # ---- label ----
        label = None
        if self.split == "train":
            label = answer_to_index(item["answer"], choices)

        # ---- video: ONE clip ----
        clip_idxs = self._sample_clip_indices(video_path, item)         # [T]
        frames = _read_video_frames(video_path, clip_idxs)              # uint8 [T,H,W,C]
        video = _preprocess_frames(frames, img_size=self.img_size)      # float [T,C,H,W]

        return {
            "id": qid,
            "video": video,                 # [T,C,H,W]
            "enc_list": enc_list,           # list[dict], variable length
            "label": label,
            "video_path": rel_video_path,
        }


# =========================================================
# Collator (AUTO choices per batch)
# =========================================================
class TrafficBuddyCollator:
    """
    Auto pad number of choices per batch:
      - Cmax = max(len(enc_list)) inside the batch
      - output tensors: [B, Cmax, L]
      - choice_mask tells which choices are real
    """

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, Any]]) -> VideoTextBatch:
        # video: [B,T,C,H,W]
        videos = torch.stack([b["video"] for b in batch], dim=0)

        # auto Cmax for THIS batch
        choice_lens = [len(b["enc_list"]) for b in batch]
        Cmax = max(choice_lens)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0

        flat: List[Dict[str, Any]] = []
        choice_mask_list: List[List[bool]] = []

        for b in batch:
            encs = list(b["enc_list"])
            c = len(encs)

            # [Cmax] mask
            mask = [True] * c + [False] * (Cmax - c)
            choice_mask_list.append(mask)

            # pad with dummy choices
            for _ in range(Cmax - c):
                encs.append({"input_ids": [pad_id], "attention_mask": [0]})

            flat.extend(encs)  # length += Cmax

        # pad token length L
        padded = self.tokenizer.pad(flat, padding=True, return_tensors="pt")  # [B*Cmax, L]
        B = len(batch)
        L = padded["input_ids"].shape[-1]

        input_ids = padded["input_ids"].view(B, Cmax, L)
        attention_mask = padded["attention_mask"].view(B, Cmax, L)
        choice_mask = torch.tensor(choice_mask_list, dtype=torch.bool)

        labels = None
        if batch[0]["label"] is not None:
            labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

        ids = [b["id"] for b in batch]
        vpaths = [b["video_path"] for b in batch]

        return VideoTextBatch(
            video=videos,
            input_ids=input_ids,
            attention_mask=attention_mask,
            choice_mask=choice_mask,
            labels=labels,
            ids=ids,
            video_paths=vpaths,
        )


# =========================================================
# Loader builder
# =========================================================
def build_loaders(
    root_dir: str,
    train_json: str,
    test_json: str,
    tokenizer_name: str = "vinai/phobert-base",
    batch_size: int = 2,
    num_workers: int = 2,
    num_frames: int = 16,
    frame_stride: int = 2,
    img_size: int = 224,
    max_len: int = 128,
):
    train_ds = TrafficBuddyDataset(
        json_path=train_json,
        root_dir=root_dir,
        tokenizer_name=tokenizer_name,
        split="train",
        num_frames=num_frames,
        frame_stride=frame_stride,
        img_size=img_size,
        max_len=max_len,
    )
    test_ds = TrafficBuddyDataset(
        json_path=test_json,
        root_dir=root_dir,
        tokenizer_name=tokenizer_name,
        split="test",
        num_frames=num_frames,
        frame_stride=frame_stride,
        img_size=img_size,
        max_len=max_len,
    )

    collate = TrafficBuddyCollator(train_ds.tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    return train_loader, test_loader, train_ds.tokenizer


# =========================================================
# Quick sanity
# =========================================================
if __name__ == "__main__":
    root_dir = r"F:\zaloaiproject\dataset\traffic_buddy_train+public_test"
    train_json = os.path.join(root_dir, "train", "train.json")
    test_json = os.path.join(root_dir, "public_test", "public_test.json")

    train_loader, test_loader, tokenizer = build_loaders(
        root_dir=root_dir,
        train_json=train_json,
        test_json=test_json,
        batch_size=2,
        num_workers=0,     # Windows debug
        num_frames=16,
        frame_stride=2,
        img_size=224,
        max_len=128,
    )

    batch = next(iter(train_loader))
    print("===== SANITY CHECK =====")
    print("video:", batch.video.shape)                       # [B,T,C,H,W]
    print("input_ids:", batch.input_ids.shape)               # [B,Cmax,L]
    print("attention_mask:", batch.attention_mask.shape)     # [B,Cmax,L]
    print("choice_mask:", batch.choice_mask.shape)           # [B,Cmax]
    print("labels:", batch.labels)
    print("ids:", batch.ids[:2])
    print("video_paths:", batch.video_paths[:2])

    # For Video Swin (torchvision) expected input: [B, C, T, H, W]
    video_for_swin = batch.video.permute(0, 2, 1, 3, 4)
    print("video_for_swin:", video_for_swin.shape, "(expected [B,C,T,H,W])")

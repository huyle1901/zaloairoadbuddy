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


def strip_choice_prefix(s: str) -> str:
    """
    Remove prefixes like 'A. ', 'B. ', 'C. ', 'D. ' if present.
    Keep content.
    """
    s = s.strip()
    # matches: "A.", "A)", "A:", "A -", ...
    return re.sub(r"^[A-Da-d]\s*[\.\)\:\-]\s*", "", s).strip()


def answer_to_index(answer: str, choices: List[str]) -> int:
    """
    Robust mapping answer -> index.
    - If answer exactly matches a choice string => use that index
    - Else if answer begins with 'A'/'B'/'C'/'D' => map letter
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

    # last resort: try matching by stripped text
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
    Create indices for a clip centered at `center`.
    Returns shape [num_frames]
    """
    span = (num_frames - 1) * frame_stride + 1
    start = center - span // 2
    idxs = torch.arange(num_frames, dtype=torch.long) * frame_stride + start
    idxs = _safe_clamp_indices(idxs, n_total)
    return idxs


def _uniform_centers(n_total: int, k: int) -> List[int]:
    """
    Choose k centers uniformly over [0, n_total-1].
    """
    if n_total <= 1:
        return [0] * k
    if k == 1:
        return [n_total // 2]
    # midpoints of k bins
    centers = []
    for i in range(k):
        # in [0,1]
        t = (i + 0.5) / k
        centers.append(int(round(t * (n_total - 1))))
    return centers


def _read_video_frames(path: str, indices: torch.Tensor) -> torch.Tensor:
    """
    Read frames at `indices` from video at `path`.
    Returns uint8 tensor [T, H, W, C] (C=3).
    """
    if _HAS_DECORD:
        vr = VideoReader(path, ctx=cpu(0))
        # decord torch bridge returns torch tensor [T,H,W,C] uint8
        frames = vr.get_batch(indices.tolist())
        return frames
    else:
        if not _HAS_TV_READ:
            raise RuntimeError(
                "No video backend available. Install decord or torchvision with video support."
            )
        # torchvision read_video loads full video; slower but works
        v, _, _ = read_video(path, pts_unit="sec")  # [Tv, H, W, C], uint8
        n_total = v.shape[0]
        idxs = _safe_clamp_indices(indices, n_total)
        frames = v[idxs]
        return frames


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
        # torchvision doesn't expose fps easily without decoding; assume 30
        # You can improve this if you parse with PyAV.
        if not _HAS_TV_READ:
            raise RuntimeError(
                "No video backend available. Install decord or torchvision with video support."
            )
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

    x = x.float() / 255.0  # [T,H,W,C]
    x = x.permute(0, 3, 1, 2).contiguous()  # [T,C,H,W]

    # resize each frame (treat T as batch)
    x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)

    mean_t = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    x = (x - mean_t) / std_t
    return x


@dataclass
class VideoTextBatch:
    video: torch.Tensor              # [B, K, T, C, H, W]
    input_ids: torch.Tensor          # [B, num_choices, L]
    attention_mask: torch.Tensor     # [B, num_choices, L]
    labels: Optional[torch.Tensor]   # [B] or None
    ids: List[str]
    video_paths: List[str]


class TrafficBuddyDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        root_dir: str,
        tokenizer_name: str = "vinai/phobert-base",
        split: str = "train",  # "train" or "test"
        num_clips: int = 8,    # K
        num_frames: int = 8,   # T
        frame_stride: int = 2,
        img_size: int = 224,
        max_len: int = 128,
        support_prob: float = 1.0,  # in train: probability to include the support-centered clip
        seed: int = 42,
    ):
        assert split in ("train", "test")
        self.json_path = json_path
        self.root_dir = root_dir
        self.split = split

        self.num_clips = num_clips
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.img_size = img_size
        self.max_len = max_len
        self.support_prob = support_prob

        self.rng = random.Random(seed)

        with open(json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        # supports either {"data":[...]} or a list directly
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

        # -------- choose center --------
        if self.split == "train":
            has_support = (
                "support_frames" in item
                and isinstance(item["support_frames"], list)
                and len(item["support_frames"]) > 0
            )

            if has_support:
                # always use support frame
                t_sec = float(self.rng.choice(item["support_frames"]))
                center = int(round(t_sec * fps))
            else:
                center = self.rng.randint(0, max(n_total - 1, 0))
        else:
            # test: take middle of video
            center = n_total // 2 if n_total > 0 else 0

        # -------- get ordered frame indices --------
        idxs = _make_clip_indices_centered(center, T, S, n_total)  # [T]
        return idxs


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]

        qid = item["id"]
        question = item["question"]
        choices = item["choices"]
        rel_video_path = item["video_path"]
        video_path = os.path.join(self.root_dir, rel_video_path)

        # ---- text tokenize: (question, choice_i) ----
        # IMPORTANT: don't use answer for retrieval/query in train (avoid leakage)
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

        # ---- video clips ----
        clip_idxs = self._sample_clip_indices(video_path, item)   # [T]

        frames = _read_video_frames(video_path, clip_idxs)        # [T,H,W,C]
        video = _preprocess_frames(frames, img_size=self.img_size)  # [T,C,H,W]

        return {
            "id": qid,
            "video": video,        # [T,C,H,W]
            "enc_list": enc_list,
            "label": label,
            "video_path": rel_video_path,
        }

class TrafficBuddyCollator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, Any]]) -> VideoTextBatch:
        # video: [B,K,T,C,H,W]
        videos = torch.stack([b["video"] for b in batch], dim=0)

        # text: pad to [B, num_choices, L]
        num_choices = len(batch[0]["enc_list"])
        assert all(len(b["enc_list"]) == num_choices for b in batch), "num_choices must be constant in a batch"

        flat = []
        for b in batch:
            flat.extend(b["enc_list"])  # list length B*num_choices

        padded = self.tokenizer.pad(
            flat,
            padding=True,
            return_tensors="pt",
        )  # dict: input_ids [B*C, L], attention_mask [B*C, L]

        B = len(batch)
        L = padded["input_ids"].shape[-1]
        input_ids = padded["input_ids"].view(B, num_choices, L)
        attention_mask = padded["attention_mask"].view(B, num_choices, L)

        labels = None
        if batch[0]["label"] is not None:
            labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

        ids = [b["id"] for b in batch]
        vpaths = [b["video_path"] for b in batch]

        return VideoTextBatch(
            video=videos,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            ids=ids,
            video_paths=vpaths,
        )


def build_loaders(
    root_dir: str,
    train_json: str,
    test_json: str,
    tokenizer_name: str = "vinai/phobert-base",
    batch_size: int = 2,
    num_workers: int = 2,
    num_clips: int = 8,
    num_frames: int = 8,
    frame_stride: int = 2,
    img_size: int = 224,
    max_len: int = 128,
):
    train_ds = TrafficBuddyDataset(
        json_path=train_json,
        root_dir=root_dir,
        tokenizer_name=tokenizer_name,
        split="train",
        num_clips=num_clips,
        num_frames=num_frames,
        frame_stride=frame_stride,
        img_size=img_size,
        max_len=max_len,
        support_prob=1.0,
    )
    test_ds = TrafficBuddyDataset(
        json_path=test_json,
        root_dir=root_dir,
        tokenizer_name=tokenizer_name,
        split="test",
        num_clips=num_clips,
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


# -------------- Quick sanity (no video decode if you comment out __getitem__ video part) --------------
if __name__ == "__main__":
    # Example (Windows):
    # root_dir = r"E:\ZALOAI\dataset\traffic_buddy_train+public_test"
    root_dir = r"F:\zaloaiproject\dataset\traffic_buddy_train+public_test"
    train_json = os.path.join(root_dir, "train", "train.json")
    test_json = os.path.join(root_dir, "public_test", "public_test.json")

    
    train_loader, test_loader, tokenizer = build_loaders(
        root_dir=root_dir,
        train_json=train_json,
        test_json=test_json,
        batch_size=2,
        num_workers=0,  # Windows: start with 0 for debugging
        num_clips=8,
        num_frames=16,
        frame_stride=2,
        img_size=224,
        max_len=128,
    )

    batch = next(iter(train_loader))
    print("video:", batch.video.shape)              # [B,K,T,C,H,W]
    print("input_ids:", batch.input_ids.shape)      # [B,choices,L]
    print("labels:", batch.labels)
    print("ids:", batch.ids[:2])

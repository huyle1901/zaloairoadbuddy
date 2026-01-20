import os
import json
import time
import math
import argparse
from dataclasses import asdict
from typing import Dict, List, Tuple, Optional

import torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

from src.datasets.frame_selector_dataset import (
    FrameSelectorDataset,
    FrameSelectorCollator,
)
from src.models.frame_selector.frame_selector_model import FrameSelectorModel

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE =", device)

if device.type == "cuda":
    print("GPU =", torch.cuda.get_device_name(0))
    print("CUDA version (torch build) =", torch.version.cuda)

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_support_map(json_path: str) -> Dict[str, List[float]]:
    """
    Build id -> support_frames(seconds) map from a json.
    Works with:
      - {"data":[...]} or list directly
    """
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    data = obj["data"] if isinstance(obj, dict) and "data" in obj else obj

    m: Dict[str, List[float]] = {}
    for it in data:
        qid = str(it.get("id", ""))
        supp = it.get("support_frames", None)
        if isinstance(supp, list) and len(supp) > 0:
            # ensure float
            m[qid] = [float(x) for x in supp]
        else:
            m[qid] = []
    return m

def soft_ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits : [B, N]
    targets: [B, N] in [0,1] (gaussian or binary)

    Normalize targets into a distribution p, then do:
      L = - sum_i p_i * log_softmax(logits)_i

    If a row sums to 0 (no support), we ignore that row.
    """
    logp = torch.log_softmax(logits, dim=-1)             # [B,N]
    sums = targets.sum(dim=-1, keepdim=True)             # [B,1]
    mask = (sums.squeeze(-1) > 0)

    if mask.any():
        p = targets[mask] / sums[mask].clamp_min(1e-12)  # [Bm,N]
        loss = -(p * logp[mask]).sum(dim=-1).mean()
        return loss

    return torch.zeros([], device=logits.device, dtype=logits.dtype)

@torch.no_grad()
def eval_localizer_exact(
    model: torch.nn.Module,
    loader,
    device: str,
    support_map: Dict[str, List[float]],
    topk: Tuple[int, ...] = (1, 3, 5),
    tol_sec: Tuple[float, ...] = (1.0, 2.0),
) -> Dict[str, float]:
    """
    Exact metric using val_json support_frames:
      Recall@K within ±tol_sec of ANY support time.
    """
    model.eval()


    # counters
    hit = {(k, t): 0 for k in topk for t in tol_sec}
    total = 0
    skipped = 0

    for batch in loader:
        video = batch.video.to(device)                 # [B,N,C,H,W]
        input_ids = batch.input_ids.to(device)         # [B,L]
        attention_mask = batch.attention_mask.to(device)
        logits = model(video, input_ids, attention_mask)  # [B,N]

        B, N = logits.shape
        max_k = max(topk)
        top_idx = torch.topk(logits, k=max_k, dim=-1).indices  # [B,max_k]

        # convert top-k indices -> time (seconds)
        # sample_indices: [B,N] (original frame index)

        sample_indices = batch.sample_indices.to(device)
        top_frames = sample_indices.gather(1, top_idx)         # [B, max_k]
        fps = batch.fps.to(device).unsqueeze(1).clamp_min(1e-6)
        top_times = top_frames.float() / fps

        for i in range(B):
            qid = str(batch.ids[i])
            supp = support_map.get(qid, [])
            if not supp:
                skipped += 1
                continue

            total += 1
            supp_t = torch.tensor(supp, device=device).view(1, -1)  # [1,S]
            # abs diff of each top time to nearest support time
            # dists: [max_k, S] -> min over S => [max_k]
            dists = (top_times[i].view(-1, 1) - supp_t).abs().min(dim=1).values  # [max_k]

            for k in topk:
                d_k = dists[:k]
                for t in tol_sec:
                    if (d_k <= t).any():
                        hit[(k, t)] += 1

    out: Dict[str, float] = {}
    denom = max(total, 1)
    for k in topk:
        for t in tol_sec:
            out[f"Recall@{k}_tol{t:g}s"] = hit[(k, t)] / denom

    out["eval_total_used"] = float(total)
    out["eval_skipped_no_support"] = float(skipped)
    return out


def make_loader(
    root_dir: str,
    json_path: str,
    batch_size: int,
    num_workers: int,
    num_samples: int,
    img_size: int,
    max_len: int,
    seed: int,
    shuffle: bool,
    label_mode: str = "gaussian",
    sigma_sec: float = 0.6,
    ensure_nearest_positive: bool = True,
):
    ds = FrameSelectorDataset(
        json_path=json_path,
        root_dir=root_dir,
        split="train",  # IMPORTANT: need support_frames -> labels
        num_samples=num_samples,
        img_size=img_size,
        max_len=max_len,
        label_mode=label_mode,
        sigma_sec=sigma_sec,
        ensure_nearest_positive=ensure_nearest_positive,
        seed=seed,
    )
    collate = FrameSelectorCollator(ds.tokenizer)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    return loader, ds.tokenizer


def train(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    train_loader, tokenizer = make_loader(
        root_dir=args.root_dir,
        json_path=args.train_json,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_samples=args.num_samples,
        img_size=args.img_size,
        max_len=args.max_len,
        seed=args.seed,
        shuffle=True,
        label_mode=args.label_mode,
        sigma_sec=args.sigma_sec,
        ensure_nearest_positive=True,
    )

    val_loader, _ = make_loader(
        root_dir=args.root_dir,
        json_path=args.val_json,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_samples=args.num_samples,
        img_size=args.img_size,
        max_len=args.max_len,
        seed=args.seed,
        shuffle=False,  # IMPORTANT
        label_mode=args.label_mode,
        sigma_sec=args.sigma_sec,
        ensure_nearest_positive=True,
    )

    # for exact eval
    support_map = load_support_map(args.val_json)

    # Model
    model = FrameSelectorModel(
        bert_name=args.bert_name,
        vision_name=args.vision_name,
        proj_dim=args.proj_dim,
        freeze_text=args.freeze_text,
        freeze_vision=args.freeze_vision,
    ).to(device)

    # Optimizer: start with head only
    head_params = list(model.v_proj.parameters()) + list(model.t_proj.parameters()) + [model.log_temp]
    opt = AdamW(
        [{"params": head_params, "lr": args.lr_head, "weight_decay": args.weight_decay}],
        lr=args.lr_head,
        weight_decay=args.weight_decay,
    )

    scaler = GradScaler(enabled=(device == "cuda"))

    best_score = -1.0
    did_add_vision = False
    did_add_text = False

    steps_per_epoch = len(train_loader)
    

    for epoch in range(1, args.epochs + 1):
        model.train()

        # --- unfreeze schedule ---
        if (epoch == args.unfreeze_vision_epoch) and (not did_add_vision):
            for p in model.vision.parameters():
                p.requires_grad = True
            opt.add_param_group(
                {"params": model.vision.parameters(), "lr": args.lr_backbone, "weight_decay": args.weight_decay}
            )
            did_add_vision = True
            print(f"[epoch {epoch}] Unfroze VISION backbone (lr={args.lr_backbone})")

        if (epoch == args.unfreeze_text_epoch) and (not did_add_text):
            for p in model.text.parameters():
                p.requires_grad = True
            opt.add_param_group(
                {"params": model.text.parameters(), "lr": args.lr_backbone, "weight_decay": args.weight_decay}
            )
            did_add_text = True
            print(f"[epoch {epoch}] Unfroze TEXT backbone (lr={args.lr_backbone})")

        total_loss = 0.0
        n_steps = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader, start=1):
            video = batch.video.to(device)
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            targets = batch.labels.to(device)

            opt.zero_grad(set_to_none=True)

            with autocast(enabled=(device == "cuda")):
                logits = model(video, input_ids, attention_mask)
                loss = soft_ce_loss(logits, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            curr_loss = float(loss.item())
            total_loss += curr_loss
            n_steps += 1

            if args.log_every > 0 and (batch_idx % args.log_every == 0):
                avg_loss = total_loss / n_steps
                print(
                    f"[Epoch {epoch}/{args.epochs}] "
                    f"[Batch {batch_idx}/{steps_per_epoch}] "
                    f"batch_loss={curr_loss:.4f} avg_loss={avg_loss:.4f}"
                )

        avg_loss = total_loss / max(n_steps, 1)
        dt = time.time() - t0
        print(f"[Epoch {epoch}/{args.epochs}] done | avg_loss={avg_loss:.4f} time={dt:.1f}s")

        # --- eval exact ---
        metrics = eval_localizer_exact(
            model=model,
            loader=val_loader,
            device=device,
            support_map=support_map,
            topk=(1, 3, 5),
            tol_sec=(1.0, 2.0),
        )

        # pick a score for model selection
        score = metrics.get("Recall@3_tol2s", 0.0)

        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"loss={avg_loss:.4f} time={dt:.1f}s | "
            f"R@1(1s)={metrics['Recall@1_tol1s']:.3f} "
            f"R@3(2s)={metrics['Recall@3_tol2s']:.3f} "
            f"(used={int(metrics['eval_total_used'])}, skipped={int(metrics['eval_skipped_no_support'])})"
        )

        # --- save best ---
        if score > best_score:
            best_score = score
            ckpt_path = os.path.join(args.save_dir, "best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "args": vars(args),
                    "best_score": best_score,
                    "tokenizer_name": args.bert_name,
                    "num_samples": args.num_samples,
                },
                ckpt_path,
            )
            print(f"  -> saved best: {ckpt_path} (score={best_score:.4f})")

    print("Done. Best score:", best_score)


def build_argparser():
    p = argparse.ArgumentParser("Train Frame Selector (stage-1)")

    p.add_argument("--root_dir", type=str, required=True)
    p.add_argument("--train_json", type=str, required=True)
    p.add_argument("--val_json", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="checkpoints/frame_selector")

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)  # Windows: 0
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--num_samples", type=int, default=96)  # N frames per video
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--max_len", type=int, default=64)

    p.add_argument("--label_mode", type=str, default="gaussian", choices=["gaussian", "binary"])
    p.add_argument("--sigma_sec", type=float, default=0.6)

    p.add_argument("--bert_name", type=str, default="vinai/phobert-base")
    p.add_argument("--vision_name", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "resnet18"])
    p.add_argument("--proj_dim", type=int, default=256)

    p.add_argument("--freeze_text", action="store_true", default=True)
    p.add_argument("--freeze_vision", action="store_true", default=True)

    p.add_argument("--unfreeze_vision_epoch", type=int, default=3)
    p.add_argument("--unfreeze_text_epoch", type=int, default=5)

    p.add_argument("--lr_head", type=float, default=1e-4)
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--log_every", type=int, default=50)

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)

# split_train_val.py
import argparse
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union


def load_json(path: str) -> Tuple[Union[Dict[str, Any], List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    Returns:
      - raw_obj: the original json object (dict with "data" or list)
      - data: list of samples
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], list):
        data = raw["data"]
    elif isinstance(raw, list):
        data = raw
    else:
        raise ValueError("Unsupported JSON format. Expected a list or a dict with key 'data'.")

    return raw, data


def save_json(path: str, raw_template: Union[Dict[str, Any], List[Dict[str, Any]]], data: List[Dict[str, Any]]) -> None:
    """
    Preserve original format:
      - If template was dict with "data", save as dict and replace data
      - If template was list, save list
    """
    if isinstance(raw_template, dict) and "data" in raw_template:
        out = dict(raw_template)
        out["data"] = data
    else:
        out = data

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def group_by_key(data: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    missing = 0
    for item in data:
        if key not in item or item[key] is None:
            missing += 1
            # put missing keys into their own pseudo group to keep deterministic
            g = "__MISSING__"
        else:
            g = str(item[key])
        groups[g].append(item)

    if missing > 0:
        print(f"[WARN] {missing} samples missing '{key}', grouped into '__MISSING__'.")

    return groups


def split_groups(
    groups: Dict[str, List[Dict[str, Any]]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str], List[str]]:
    """
    Split by group keys (video_path) to prevent leakage.
    Returns: train_items, val_items, train_group_keys, val_group_keys
    """
    assert 0.0 < val_ratio < 1.0

    keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    n_val_groups = max(1, int(round(len(keys) * val_ratio)))
    val_keys = set(keys[:n_val_groups])
    train_keys = set(keys[n_val_groups:])

    train_items: List[Dict[str, Any]] = []
    val_items: List[Dict[str, Any]] = []

    for k in keys:
        if k in val_keys:
            val_items.extend(groups[k])
        else:
            train_items.extend(groups[k])

    return train_items, val_items, sorted(train_keys), sorted(val_keys)


def stats(name: str, items: List[Dict[str, Any]], group_key: str) -> None:
    vids = set()
    n_support = 0
    n_answer = 0
    n_choices_var = 0
    choices_lens = defaultdict(int)

    for it in items:
        vids.add(str(it.get(group_key, "__MISSING__")))
        if isinstance(it.get("support_frames", None), list) and len(it["support_frames"]) > 0:
            n_support += 1
        if "answer" in it and it["answer"] not in (None, ""):
            n_answer += 1
        if "choices" in it and isinstance(it["choices"], list):
            choices_lens[len(it["choices"])] += 1

    if len(choices_lens) > 1:
        n_choices_var = 1

    print(f"\n=== {name} ===")
    print(f"samples: {len(items)}")
    print(f"unique {group_key}: {len(vids)}")
    print(f"samples with support_frames: {n_support}")
    print(f"samples with answer: {n_answer}")
    if n_choices_var:
        dist = ", ".join([f"{k}:{v}" for k, v in sorted(choices_lens.items())])
        print(f"choices length distribution: {dist}")
    else:
        if len(choices_lens) == 1:
            k = next(iter(choices_lens.keys()))
            print(f"choices length: {k}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to train.json")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio by video groups (default 0.1)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    ap.add_argument("--group_key", type=str, default="video_path", help="Group key to avoid leakage (default video_path)")
    ap.add_argument("--train_name", type=str, default="train_split.json", help="Output train file name")
    ap.add_argument("--val_name", type=str, default="val_split.json", help="Output val file name")
    args = ap.parse_args()

    raw, data = load_json(args.input)
    groups = group_by_key(data, args.group_key)

    train_items, val_items, train_keys, val_keys = split_groups(
        groups=groups,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # sanity: no overlap group keys
    overlap = set(train_keys).intersection(set(val_keys))
    if overlap:
        raise RuntimeError(f"Leak detected: overlapping group keys: {list(overlap)[:5]}")

    out_train = os.path.join(args.out_dir, args.train_name)
    out_val = os.path.join(args.out_dir, args.val_name)

    save_json(out_train, raw, train_items)
    save_json(out_val, raw, val_items)

    print("\n[SPLIT DONE]")
    print("input:", args.input)
    print("out_train:", out_train)
    print("out_val:", out_val)
    print(f"val_ratio(by groups): {args.val_ratio}, seed: {args.seed}, group_key: {args.group_key}")
    print(f"groups: total={len(groups)}, train_groups={len(train_keys)}, val_groups={len(val_keys)}")

    stats("TRAIN_SPLIT", train_items, args.group_key)
    stats("VAL_SPLIT", val_items, args.group_key)


if __name__ == "__main__":
    main()

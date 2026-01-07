import os, json
from collections import defaultdict

def load_items(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj["data"] if isinstance(obj, dict) and "data" in obj else obj

def get_nframes_decord(video_path: str) -> int:
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0))
    return len(vr)

def main():
    root_dir = r"F:\zaloaiproject\dataset\traffic_buddy_train+public_test"
    train_json = os.path.join(root_dir, "train", "train.json")
    test_json  = os.path.join(root_dir, "public_test", "public_test.json")

    items = load_items(train_json) + load_items(test_json)

    # lấy unique video paths để đỡ đọc lại nhiều lần
    rel_paths = sorted({it["video_path"] for it in items})
    abs_paths = [os.path.normpath(os.path.join(root_dir, p)) for p in rel_paths]

    max_frames = -1
    max_video = None
    errors = []

    for i, vp in enumerate(abs_paths, 1):
        try:
            n = get_nframes_decord(vp)
            if n > max_frames:
                max_frames = n
                max_video = vp
        except Exception as e:
            errors.append((vp, str(e)))

        if i % 50 == 0:
            print(f"[{i}/{len(abs_paths)}] current max = {max_frames}")

    print("\n=== RESULT ===")
    print("Max frames:", max_frames)
    print("Video:", max_video)

    if errors:
        print("\n=== ERRORS (first 10) ===")
        for vp, msg in errors[:10]:
            print("-", vp, "->", msg)

if __name__ == "__main__":
    main()

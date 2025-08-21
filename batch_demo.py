import os
import h5py
import subprocess
import argparse
from pathlib import Path

"""
Standalone batch runner for demo.py across FAIR-Play splits.

- Picks split-specific checkpoints automatically:
  <ckpt_root>/split{k}/mono2binaural/{visual,audio}_best.pth
- Reads IDs from each split's test.h5 (dataset key: 'audio'), extracting the stem (e.g., '001353')
- Builds demo.py CLI per ID and executes it
- Creates per-split output folders:
  <output_root>/<output_split_prefix><k>/BaseSplit<k>_<ID>/
- No shell wrapper needed
"""

# Ensure CUDA allocator setting is always set (no .sh wrapper required)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def list_ids_from_h5(h5_path: str):
    ids = []
    with h5py.File(h5_path, 'r') as f:
        audio_paths = f['audio']  # byte strings of wav paths
        for item in audio_paths:
            p = item.decode()
            id_ = Path(p).stem  # "001353.wav" -> "001353"
            ids.append(id_)
    return ids


def main():
    ap = argparse.ArgumentParser(description="Batch runner for demo.py over FAIR-Play splits")
    ap.add_argument("--dry_run", action="store_true", help="Only print commands, do not execute them")

    # --- Sensible defaults so you can run with few/no flags ---
    ap.add_argument("--fairplay_root", default="/scratch/yc01847/FAIR-Play",
                    help="Root of FAIR-Play (contains binaural_audios/, frames/, splits/, etc.)")
    ap.add_argument("--splits_root", default=None,
                    help="Optional. If omitted, uses <fairplay_root>/splits")
    ap.add_argument("--output_root", default="/scratch/yc01847/Base_demo_output",
                    help="Where to write demo outputs")

    # NEW: prefix that becomes Base_demo_split1, Base_demo_split2, ...
    ap.add_argument("--output_split_prefix", default="Base_demo_split",
                    help="Prefix for per-split output folders (e.g., Base_demo_split -> Base_demo_split1, ...)")

    ap.add_argument("--num_splits", type=int, default=10)

    # Split-specific checkpoints are derived automatically under this root
    ap.add_argument("--ckpt_root", default="/scratch/yc01847/2.5D-Visual-Sound/checkpoints",
                    help="Root containing split{k}/mono2binaural/{visual,audio}_best.pth")

    # Demo/runtime knobs
    ap.add_argument("--audio_sampling_rate", type=int, default=16000)
    ap.add_argument("--input_audio_length", type=float, default=10.0)
    ap.add_argument("--hop_size", type=float, default=0.05)

    # Path to demo.py (in case you moved it)
    ap.add_argument("--demo_py", default="demo.py", help="Path to demo.py")

    args = ap.parse_args()

    # Resolve derived paths
    splits_root = args.splits_root or os.path.join(args.fairplay_root, "splits")
    binaural_dir = os.path.join(args.fairplay_root, "binaural_audios")
    frames_dir   = os.path.join(args.fairplay_root, "frames")

    Path(args.output_root).mkdir(parents=True, exist_ok=True)

    print("[info] PYTORCH_CUDA_ALLOC_CONF=", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
    print("[info] fairplay_root=", args.fairplay_root)
    print("[info] splits_root=", splits_root)
    print("[info] ckpt_root=", args.ckpt_root)
    print("[info] output_root=", args.output_root)
    print("[info] output_split_prefix=", args.output_split_prefix)
    print("[info] num_splits=", args.num_splits)
    print("[info] demo_py=", args.demo_py)

    for k in range(1, args.num_splits + 1):
        split_dir = os.path.join(splits_root, f"split{k}")
        h5_path = os.path.join(split_dir, "test.h5")  # change to train.h5 if needed

        if not os.path.isfile(h5_path):
            print(f"[skip] {h5_path} not found")
            continue

        ids = list_ids_from_h5(h5_path)
        print(f"[split{k}] {len(ids)} items")

        # derive split-specific checkpoints
        vis_ckpt = os.path.join(args.ckpt_root, f"split{k}", "mono2binaural", "visual_best.pth")
        aud_ckpt = os.path.join(args.ckpt_root, f"split{k}", "mono2binaural", "audio_best.pth")
        if not (os.path.isfile(vis_ckpt) and os.path.isfile(aud_ckpt)):
            print(f"[warn] Missing checkpoints for split{k}:")
            print(f"       {vis_ckpt}")
            print(f"       {aud_ckpt}")
            # skip this split if weights are missing
            continue

        # per-split output base
        split_folder = f"{args.output_split_prefix}{k}"
        base_output_dir = os.path.join(args.output_root, split_folder)
        Path(base_output_dir).mkdir(parents=True, exist_ok=True)

        for id_ in ids:
            input_audio_path = os.path.join(binaural_dir, f"{id_}.wav")

            # Prefer a frames folder named <ID> containing 000001.png, ... else fall back to an .mp4
            candidate_frame_dir = os.path.join(frames_dir, id_)
            if os.path.isdir(candidate_frame_dir):
                video_frame_path = candidate_frame_dir
            else:
                video_frame_path = os.path.join(frames_dir, f"{id_}.mp4")

            # one output dir per (split, id)
            output_dir_root = os.path.join(base_output_dir, f"BaseSplit{k}_{id_}")
            Path(output_dir_root).mkdir(parents=True, exist_ok=True)

            cmd = [
                "python", args.demo_py,
                "--input_audio_path", input_audio_path,
                "--video_frame_path", video_frame_path,
                "--hdf5FolderPath", split_dir,
                "--weights_visual", vis_ckpt,
                "--weights_audio",  aud_ckpt,
                "--output_dir_root", output_dir_root,
                "--input_audio_length", str(args.input_audio_length),
                "--hop_size", str(args.hop_size),
                "--audio_sampling_rate", str(args.audio_sampling_rate),
            ]

            print(">>", " ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True)
            else:
                print("[dry run] Command not executed.")


if __name__ == "__main__":
    main()
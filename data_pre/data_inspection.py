import os
import h5py
import shutil

def extract_video_ids(h5_path):
    ids = []
    with h5py.File(h5_path, "r") as f:
        # assuming file paths are stored as strings under some dataset
        for key in f.keys():
            data = f[key][()]
            if isinstance(data, bytes):  # decode if needed
                data = data.decode()
            if data.endswith(".wav"):
                file_id = os.path.splitext(os.path.basename(data))[0]
                ids.append(file_id)
    return ids

def grab_split_videos(split, fairplay_root, dest_root):
    split_dir = os.path.join(fairplay_root, "splits", split)
    video_dir = os.path.join(fairplay_root, "videos")

    for subset in ["train.h5", "val.h5", "test.h5"]:
        h5_path = os.path.join(split_dir, subset)
        ids = extract_video_ids(h5_path)

        # create destination folder
        dest_folder = os.path.join(dest_root, split, subset.replace(".h5", ""))
        os.makedirs(dest_folder, exist_ok=True)

        # copy/symlink the videos
        for file_id in ids:
            src = os.path.join(video_dir, f"{file_id}.mp4")
            dst = os.path.join(dest_folder, f"{file_id}.mp4")
            if os.path.exists(src):
                shutil.copy(src, dst)  # or use os.symlink(src, dst) for space efficiency

        print(f"Saved {len(ids)} videos for {subset} in {dest_folder}")

# Example usage
fairplay_root = "/scratch/yc01847/FAIR-Play"
dest_root = "/scratch/yc01847/FAIR_Play_inspection/video_splits"
grab_split_videos("split8", fairplay_root, dest_root)
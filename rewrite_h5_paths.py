import h5py
import sys

def rewrite_h5_paths(h5_path, old_prefix, new_prefix):
    with h5py.File(h5_path, 'r+') as f:
        key = f['audio']  # get the 'audio' key
        for i in range(len(key)):
            path = key[i].decode()
            new_path = path.replace(old_prefix, new_prefix)
            key[i] = new_path.encode()
            print(f"[{i}] ✅ Updated: {path} → {new_path}")

if __name__ == "__main__": 
    # if len(sys.argv) != 4:
    #     print("Usage: python rewrite_h5_paths.py <h5_path> <old_prefix> <new_prefix>")
    #     sys.exit(1)

    # h5_path = sys.argv[1]
    # old_prefix = sys.argv[2]
    # new_prefix = sys.argv[3]
    h5_path = '/scratch/yc01847/FAIR-Play/splits/split9/test.h5'
    old_prefix = '/private/home/rhgao/datasets/BINAURAL_MUSIC_ROOM/binaural16k/'
    new_prefix = '/scratch/yc01847/FAIR-Play/binaural_audios/'

    rewrite_h5_paths(h5_path, old_prefix, new_prefix)
    print(f"✅ Finished rewriting paths in {h5_path}")
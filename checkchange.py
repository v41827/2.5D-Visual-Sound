# import h5py

# with h5py.File("/scratch/yc01847/FAIR-Play/splits/split1/val.h5", 'r') as f:
#     dset = f['audio']
#     for i in range(len(dset)):
#         path = dset[i].decode()
#         if "private/home" in path:
#             print("❌ Still old path:", path)

import h5py

with h5py.File("/scratch/yc01847/FAIR-Play/splits/split1/test.h5", "r") as f:
    dset = f["audio"]
    for i in range(20):
        print(dset[i].decode())
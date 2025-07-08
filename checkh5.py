import h5py
# original .h5 file path
f = h5py.File('/scratch/yc01847/FAIR-Play/splits/split1/train.h5', 'r')
dset = f['audio']

# old path prefix
old_prefix = '/private/home/rhgao/datasets/'
# new path prefix (my own folder)
new_prefix = '/scratch/yc01847/FAIR-Play/'

# print the first 3 paths with the new prefix
for i in range(3):
    path = dset[i].decode()  # convert byte to str
    new_path = path.replace(old_prefix, new_prefix)
    print(new_path)
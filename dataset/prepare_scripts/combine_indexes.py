import config
import h5py
import numpy as np
import os

def get_sub_filepaths(folder):
    paths = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            path = os.path.join(root, name)
            paths.append(path)
    return paths

def combine_full_indexes(indexes_hdf5s_dir, full_indexes_hdf5_path):
    """Combine all balanced and unbalanced indexes hdf5s to a single hdf5. This 
    combined indexes hdf5 is used for training with full data (~20k balanced 
    audio clips + ~1.9m unbalanced audio clips).
    """

    classes_num = config.classes_num

    # Paths
    paths = get_sub_filepaths(indexes_hdf5s_dir)
    paths = [path for path in paths if (
        'train' in path and 'full_train' not in path and 'mini' not in path)]

    print('Total {} hdf5 to combine.'.format(len(paths)))

    with h5py.File(full_indexes_hdf5_path, 'w') as full_hf:
        full_hf.create_dataset(
            name='audio_name', 
            shape=(0,), 
            maxshape=(None,), 
            dtype='S20')
        
        full_hf.create_dataset(
            name='target', 
            shape=(0, classes_num), 
            maxshape=(None, classes_num), 
            dtype='u1')

        full_hf.create_dataset(
            name='waveform', 
            shape=(0,), 
            maxshape=(None,), 
            dtype=h5py.vlen_dtype(np.dtype('uint8')))

        full_hf.create_dataset(
            name='hdf5_path', 
            shape=(0,), 
            maxshape=(None,), 
            dtype='S200')

        full_hf.create_dataset(
            name='index_in_hdf5', 
            shape=(0,), 
            maxshape=(None,), 
            dtype=np.int32)

        for path in paths:
            with h5py.File(path, 'r') as part_hf:
                print(path)
                n = len(full_hf['audio_name'][:])
                new_n = n + len(part_hf['audio_name'][:])

                full_hf['audio_name'].resize((new_n,))
                full_hf['audio_name'][n : new_n] = part_hf['audio_name'][:]

                full_hf['target'].resize((new_n, classes_num))
                full_hf['target'][n : new_n] = part_hf['target'][:]

                full_hf['waveform'].resize((new_n,))
                full_hf['waveform'][n : new_n] = part_hf['waveform'][:]

                full_hf['hdf5_path'].resize((new_n,))
                full_hf['hdf5_path'][n : new_n] = part_hf['hdf5_path'][:]

                full_hf['index_in_hdf5'].resize((new_n,))
                full_hf['index_in_hdf5'][n : new_n] = part_hf['index_in_hdf5'][:]
                
    print('Write combined full hdf5 to {}'.format(full_indexes_hdf5_path))

if __name__ == "__main__":
    combine_full_indexes('audioset/hdf5s/indexes', 'audioset/hdf5s/indexes/full_train.h5')


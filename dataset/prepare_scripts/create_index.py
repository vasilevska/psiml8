import os
import h5py
import numpy as np

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

def create_indexes(waveforms_hdf5_path, indexes_hdf5_path):
    """Create indexes a for dataloader to read for training. When users have 
    a new task and their own data, they need to create similar indexes. The 
    indexes contain meta information of "where to find the data for training".
    """

    # Paths
    create_folder(os.path.dirname(indexes_hdf5_path))

    with h5py.File(waveforms_hdf5_path, 'r') as hr:
        with h5py.File(indexes_hdf5_path, 'w') as hw:
            audios_num = len(hr['audio_name'])
            hw.create_dataset('audio_name', data=hr['audio_name'][:], dtype='S20')
            hw.create_dataset('target', data=hr['target'][:], dtype='u1')
            hw.create_dataset('waveform', data=hr['waveform'][:], dtype=h5py.vlen_dtype(np.dtype('uint8')))
            hw.create_dataset('hdf5_path', data=[waveforms_hdf5_path.encode()] * audios_num, dtype='S200')
            hw.create_dataset('index_in_hdf5', data=np.arange(audios_num), dtype=np.int32)

    print('Write to {}'.format(indexes_hdf5_path))

if __name__ == '__main__':
    create_indexes('audioset/hdf5s/waveforms/balanced_train_segments.hdf', 'audioset/hdf5s/indexes/balanced_train/balanced_train.h5')
    create_indexes('audioset/hdf5s/waveforms/eval_segments.hdf', 'audioset/hdf5s/indexes/eval.h5')
    create_indexes('audioset/hdf5s/waveforms/unbalanced_train_segments.hdf', 'audioset/hdf5s/indexes/unbalanced_train/unbalanced_train.h5')
    """
    for i in range(1,41):
        if i<10:
            i = "0"+ str(i)
        else:
            i = str(i)
        create_indexes('audioset/hdf5s/waveforms/unbalanced_train/unbalanced_train_part' + i + '.h5', 'audioset/hdf5s/indexes/unbalanced_train/unbalanced_train_part' + i + '.h5')
    """
    
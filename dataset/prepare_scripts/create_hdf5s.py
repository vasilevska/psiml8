import logging
import h5py
import time
import config
import os
import numpy as np
import librosa
from multiprocessing import Process

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('\\')[-1]
    na = os.path.splitext(na_ext)[0]
    return na

def pad_or_truncate(x, audio_length):
    """Pad all audio to specific length."""
    if len(x) <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - len(x))), axis=0)
    else:
        return x[0 : audio_length]

def create_logging(log_dir, filemode):
    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
        i1 += 1
        
    log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging


def read_metadata(csv_path, classes_num, id_to_ix):
    """Read metadata of AudioSet from a csv file.
    Args:
      csv_path: str
    Returns:
      meta_dict: {'audio_name': (audios_num,), 'target': (audios_num, classes_num)}
    """

    with open(csv_path, 'r') as fr:
        lines = fr.readlines()
        lines = lines[3:]   # Remove heads

    audios_num = len(lines)
    targets = np.zeros((audios_num, classes_num), dtype=np.bool)
    audio_names = []
 
    for n, line in enumerate(lines):
        items = line.split(', ')
        """items: ['--4gqARaEJE', '0.000', '10.000', '"/m/068hy,/m/07q6cd_,/m/0bt9lr,/m/0jbk"\n']"""

        audio_name = 'Y{}.wav'.format(items[0])   # Audios are started with an extra 'Y' when downloading
        label_ids = items[3].split('"')[1].split(',')

        audio_names.append(audio_name)

        # Target
        for id in label_ids:
            ix = id_to_ix[id]
            targets[n, ix] = 1
    
    meta_dict = {'audio_name': np.array(audio_names), 'target': targets}
    return meta_dict


def float32_to_int16(x):
    assert np.max(np.abs(x)) <= 1.2
    x = np.clip(x, -1, 1)
    return (x * 32767.).astype(np.int16)


def pack_waveforms_to_hdf5(audios_dir, csv_path, waveforms_hdf5_path, mini_data = None):
    """Pack waveform and target of several audio clips to a single hdf5 file. 
    This can speed up loading and training.
    """

    clip_samples = config.clip_samples
    classes_num = config.classes_num
    sample_rate = config.sample_rate
    id_to_ix = config.id_to_ix

    # Paths
    if mini_data:
        prefix = 'mini_'
        waveforms_hdf5_path += '.mini'
    else:
        prefix = ''

    create_folder(os.path.dirname(waveforms_hdf5_path))

    logs_dir = 'audioset/_logs/pack_waveforms_to_hdf5/{}{}'.format(prefix, get_filename(csv_path))
    create_folder(logs_dir)
    create_logging(logs_dir, filemode='w')
    logging.info('Write logs to {}'.format(logs_dir))
    
    # Read csv file
    meta_dict = read_metadata(csv_path, classes_num, id_to_ix)

    if mini_data:
        mini_num = 10
        for key in meta_dict.keys():
            meta_dict[key] = meta_dict[key][0 : mini_num]

    audios_num = len(meta_dict['audio_name'])

    # Pack waveform to hdf5
    total_time = time.time()

    with h5py.File(waveforms_hdf5_path, 'w') as hf:
        hf.create_dataset('audio_name', shape=((audios_num,)), dtype='S20')
        hf.create_dataset('waveform', shape=((audios_num, clip_samples)), dtype=np.int16)
        hf.create_dataset('target', shape=((audios_num, classes_num)), dtype=np.bool)
        hf.attrs.create('sample_rate', data=sample_rate, dtype=np.int32)

        # Pack waveform & target of several audio clips to a single hdf5 file
        for n in range(audios_num):
            audio_path = os.path.join(audios_dir, meta_dict['audio_name'][n])

            if os.path.isfile(audio_path):
                logging.info('{} {}'.format(n, audio_path))
                (audio, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
                audio = pad_or_truncate(audio, clip_samples)

                hf['audio_name'][n] = meta_dict['audio_name'][n].encode()
                hf['waveform'][n] = float32_to_int16(audio)
                hf['target'][n] = meta_dict['target'][n]
            #else:
            #   logging.info('{} File does not exist! {}'.format(n, audio_path))

    logging.info('Write to {}'.format(waveforms_hdf5_path))
    logging.info('Pack hdf5 time: {:.3f}'.format(time.time() - total_time))

if __name__ == '__main__':
    pack_waveforms_to_hdf5('audioset/mp3_audios/balanced_train_segments', 'audioset/metadata/balanced_train_segments.csv', 'audioset/hdf5s/waveforms/balanced_train.h5')
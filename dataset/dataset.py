import numpy as np
import logging
import av
import io
from torch.utils.data import Dataset
import h5py
import random
import torch
from .prepare_scripts import config 

class Audioset(Dataset):
    def __init__(self, index_path, idc, config, eval_mode = False):
        """
        Args:
           index_path: the link to each audio
           idc: npy file, the number of samples in each class, computed in main
           config: the config.py module 
           eval_model (bool): to indicate if the dataset is a testing dataset
        """
        self.config = config
        #self.dataset_file = 
        self.idc = idc
        self.index_path = index_path
        dataset_file = h5py.File(index_path, "r")
        self.total_size = len(dataset_file["audio_name"])
        self.classes_num = config.classes_num
        self.eval_mode = eval_mode
        self.clip_length = 10 * config.sample_rate
        

        if not eval_mode:
            self.generate_queue()
        else:
            if self.config.debug:
                self.total_size = 1000
            self.queue = []
            for i in range(self.total_size):
                target = dataset_file["target"][i]
                if np.sum(target) > 0:
                    self.queue.append(i)
            self.total_size = len(self.queue)
        logging.info("total dataset size: %d" %(self.total_size))
        logging.info("class num: %d" %(self.classes_num))
    
    def __len__(self):
        return self.total_size
    
    def generate_queue(self):
        self.queue = []      
        if self.config.debug:
            self.total_size = 1000
        if self.config.balanced_data:
            while len(self.queue) < self.total_size:
                if self.config.class_filter is not None:
                    class_set = self.config.class_filter[:]
                else:
                    class_set = [*range(self.classes_num)]
                random.shuffle(class_set)
                for d in class_set:
                    l = len(self.idc[d]) - 1
                    if l>0:
                        self.queue += [self.idc[d][random.randint(0,l)]]
                #self.queue += [self.idc[d][random.randint(0, len(self.idc[d]) - 1)] for d in class_set]
            self.queue = self.queue[:self.total_size]
        else:
            self.queue = [*range(self.total_size)]
            random.shuffle(self.queue)
        
        logging.info("queue regenerated:%s" %(self.queue[-5:]))
    def decode_mp3(self, mp3_arr):
        """
        decodes an array if uint8 representing an mp3 file
        :rtype: np.array
        """
        container = av.open(io.BytesIO(mp3_arr.tobytes()))
        stream = next(s for s in container.streams if s.type == 'audio')
        # print(stream)
        a = []
        for i, packet in enumerate(container.demux(stream)):
            for frame in packet.decode():
                a.append(frame.to_ndarray().reshape(-1))
        waveform = np.concatenate(a)
        if waveform.dtype != 'float32':
            raise RuntimeError("Unexpected wave type")
        return waveform
    def pad_or_truncate(self, x, audio_length):
        """Pad all audio to specific length."""
        if len(x) <= audio_length:
            return np.concatenate((x, np.zeros(audio_length - len(x), dtype=np.float32)), axis=0)
        else:
            return x[0: audio_length]
    def pydub_augment(self, waveform, gain_augment=7, ir_augment=0):
        #if ir_augment and torch.rand(1) < ir_augment:
        #    ir = get_ir_sample()
        #    waveform = convolve(waveform, ir, 'full')
        if gain_augment:
            gain = torch.randint(gain_augment * 2, (1,)).item() - gain_augment
            amp = 10 ** (gain / 20)
            waveform = waveform * amp
        return waveform

    def __getitem__(self, index):
        """Load waveform and target of an audio clip.
        Args:
            index: the index number
        Return: {
            "hdf5_path": str,
            "index_in_hdf5": int,
            "audio_name": str,
            "waveform": (clip_samples,),
            "target": (classes_num,)
        }
        """
        dataset_file = h5py.File(self.index_path, "r")
        s_index = self.queue[index]
        audio_name = dataset_file["audio_name"][s_index].decode()
        hdf5_path = dataset_file["hdf5_path"][s_index].decode()
        r_idx = dataset_file["index_in_hdf5"][s_index]
        target = dataset_file["target"][s_index].astype(np.float32)
        arr = dataset_file['waveform'][s_index]
        waveform = self.decode_mp3(arr)
    
        if (not self.eval_mode):
            waveform = self.pydub_augment(waveform)
            waveform = self.pad_or_truncate(waveform, self.clip_length)
            waveform = self.resample(waveform)
            #mora da se popravi jer nema 2 dela waveforma
            k = random.randint(0, self.total_size)
            arr2 = dataset_file['waveform'][k]
            target2 = dataset_file["target"][k]
            waveform2 = self.decode_mp3(arr2)
            mix_sample = int(len(waveform2) * random.uniform(self.config.token_label_range[0],self.config.token_label_range[1]))
            mix_position = random.randint(0, len(waveform2) - mix_sample - 1)
            mix_waveform = np.concatenate(
                [waveform[:mix_position], 
                waveform2[mix_position:mix_position+mix_sample],
                waveform[mix_position+mix_sample:]],
                axis=0
            )
            mix_target = np.concatenate([
                np.tile(target,(mix_position,1)),
                np.tile(target2, (mix_sample, 1)),
                np.tile(target, (len(waveform) - mix_position - mix_sample, 1))],
                axis=0
            ) 
            data_dict = {
                "hdf5_path": hdf5_path,
                "index_in_hdf5": r_idx,
                "audio_name": audio_name,
                "waveform": mix_waveform,
                "target": mix_target
            }
        else:
            waveform = self.pad_or_truncate(waveform, self.clip_length)
            waveform = self.resample(waveform)
            data_dict = {
                "hdf5_path": hdf5_path,
                "index_in_hdf5": r_idx,
                "audio_name": audio_name,
                "waveform": waveform,
                "target": target
            }
        return data_dict

    def resample(self, waveform):
        """Resample.
        Args:
          waveform: (clip_samples,)
        Returns:
          (resampled_clip_samples,)
        """
        if self.config.sample_rate == 32000:
            return waveform
        elif self.config.sample_rate == 16000:
            return waveform[0:: 2]
        elif self.config.sample_rate == 8000:
            return waveform[0:: 4]
        else:
            raise Exception('Incorrect sample rate!') 

if __name__ == '__main__':
    full_train_idc = np.load('full_train_idc.npy', allow_pickle=True)
    a = Audioset('audioset/hdf5s/indexes/full_train.h5', full_train_idc, config, False)
    print(a.__getitem__(0))

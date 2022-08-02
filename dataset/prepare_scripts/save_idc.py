import config
import os
import numpy as np
import logging
import h5py
from tqdm import tqdm

def process_idc(index_path, classes_num, filename):
    # load data
    logging.info("Load Data...............")
    idc = [[] for _ in range(classes_num)]
    with h5py.File(index_path, "r") as f:
        for i in tqdm(range(len(f["target"]))):
            t_class = np.where(f["target"][i])[0]
            for t in t_class:
                idc[t].append(i)
    print(idc)
    np.save(filename, idc)
    logging.info("Load Data Succeed...............")

def save_idc():
    train_index_path = 'audioset/hdf5s/indexes/full_train.h5'
    eval_index_path = 'audioset/hdf5s/indexes/eval.h5'
    process_idc(train_index_path, config.classes_num,  "full_train_idc.npy")
    process_idc(eval_index_path, config.classes_num, "eval_idc.npy")

if __name__ == "__main__":
    save_idc()
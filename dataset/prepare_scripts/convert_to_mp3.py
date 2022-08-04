import argparse
import multiprocessing
import glob
import os

# Replace this with the dataset downloaded using PANN scripts
source_path = "audioset/audios/"

# Replace with the output directory
out_path = "audioset/mp3_audios/"

all_num = 0


def process_folder(fol="balanced_train_segments"):
    print("now working on ", fol)
    os.makedirs(out_path + fol, exist_ok=True)
    all_files = list(glob.glob(source_path + fol + "/*.wav"))
    print(f"it has {len(all_files)}")
    global all_num
    all_num = len(all_files)
    cmds = [(i, file, out_path + fol + "/" + os.path.basename(file)[:-3]) for i, file in enumerate(all_files)]
    print(cmds[0])
    with multiprocessing.Pool(processes=10) as pool:
        pool.starmap(process_one, cmds)


def process_one(i, f1, f2):
    if i % 100 == 0:
        print(f"{i}/{all_num} \t", f1)
    os.system(f"ffmpeg  -hide_banner -nostats -loglevel error -n -i {f1} -codec:a mp3 -ar 32000 {f2}mp3")
    os.remove(f1)


if __name__ == '__main__':

    folders = ['balanced_train_segments', 'eval_segments', 'unbalanced_train_segments']
    #folders = ['unbalanced_train_segments']

    print("I will work on these folders:")
    print(folders)

    for fol in folders:
        process_folder(fol)
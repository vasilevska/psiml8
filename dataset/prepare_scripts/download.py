import os
import time
import glob
import logging
import datetime
import numpy as np
from multiprocessing import Process


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

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


def download_wavs(csv_path, audios_dir, file):
    """Download videos and extract audio in wav format.
    """

    # Paths
    #csv_path = args.csv_path
    #audios_dir = args.audios_dir
    #mini_data = args.mini_data
    
    
    logs_dir = 'audioset/audios/_logs/download_dataset/{}.csv'.format(file)
    

    create_folder(audios_dir)
    create_folder(logs_dir)
    create_logging(logs_dir, filemode='w')
    logging.info('Download log is saved to {}'.format(logs_dir))


    # Read csv
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    
    lines = lines[3:]   # Remove csv head info

    #f mini_data:
    #   lines = lines[0 : 10]   # Download partial data for debug
    
    download_time = time.time()

    # Download
    for (n, line) in enumerate(lines):
        
        items = line.split(', ')
        audio_id = items[0]
        start_time = float(items[1])
        end_time = float(items[2])
        duration = end_time - start_time
        
        logging.info('{} {} start_time: {:.1f}, end_time: {:.1f}'.format(
            n, audio_id, start_time, end_time))
        
        #"youtube-dl --quiet -o audios/eval_segments/Y_4gqARaEJE.wav -x https://www.youtube.com/watch?v=--4gqARaEJE"
        # Download full video of whatever format
        video_name = 'audioset/audios/{}/_Y{}.%(ext)s'.format(file, audio_id)
        download_string = "youtube-dl --quiet -o {} -x https://www.youtube.com/watch?v={}".format(video_name, audio_id)
        os.system(download_string)

        video_paths = glob.glob('audioset/audios/'+ file + '/_Y' + audio_id + '.*')
        print(video_paths)
        # If download successful
        if len(video_paths) > 0:
            video_path = video_paths[0]     # Choose one video

            # Add 'Y' to the head because some video ids are started with '-'
            # which will cause problem
            audio_path = 'audioset/audios/'+ file +'/Y' + audio_id + '.wav'

            # Extract audio in wav format
            os.system("ffmpeg -loglevel panic -i {} -ac 1 -ar 32000 -ss {} -t 00:00:{} {} "\
                .format(video_path, 
                str(datetime.timedelta(seconds=start_time)), duration, 
                audio_path))
            
            # Remove downloaded video
            #os.system("rm {}".format(video_path))
            
            logging.info("Download and convert to {}".format(audio_path))
                
    logging.info('Download finished! Time spent: {:.3f} s'.format(
        time.time() - download_time))

    logging.info('Logs can be viewed in {}'.format(logs_dir))

def split_unbalanced_csv_to_partial_csvs(unbalanced_csv_path, unbalanced_partial_csvs_dir):
    """Split unbalanced csv to part csvs. Each part csv contains up to 50000 ids. 
    """
    
    create_folder(unbalanced_partial_csvs_dir)
    
    with open(unbalanced_csv_path, 'r') as f:
        lines = f.readlines()

    lines = lines[3:]   # Remove head info
    audios_num_per_file = 50000
    
    files_num = int(np.ceil(len(lines) / float(audios_num_per_file)))
    
    for r in range(files_num):
        lines_per_file = lines[r * audios_num_per_file : 
            (r + 1) * audios_num_per_file]
        
        out_csv_path = os.path.join(unbalanced_partial_csvs_dir, 
            'unbalanced_train_segments_part{:02d}.csv'.format(r))

        with open(out_csv_path, 'w') as f:
            f.write('empty\n')
            f.write('empty\n')
            f.write('empty\n')
            for line in lines_per_file:
                f.write(line)
        
        print('Write out csv to {}'.format(out_csv_path))

if __name__ == '__main__':
    """
    os.system('wget -O "audioset/metadata/eval_segments.csv" "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv"')
    os.system('wget -O "audioset/metadata/balanced_train_segments.csv" "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv"')
    os.system('wget -O "audioset/metadata/unbalanced_train_segments.csv" "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv"')
    os.system('wget -O "audioset/metadata/class_labels_indices.csv" "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"')
    os.system('wget -O "audioset/metadata/qa_true_counts.csv" "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/qa/qa_true_counts.csv"')
    """
    #split_unbalanced_csv_to_partial_csvs('audioset/metadata/unbalanced_train_segments.csv', 'audioset/metadata/unbalanced_train_segments')
    #p1 = Process(target=download_wavs, args = ["audioset/metadata/eval_segments.csv", "audioset/audios/eval_segments", 'eval_segments'])
    #p2 = Process(target=download_wavs, args = ["audioset/metadata/balanced_train_segments.csv", "audioset/audios/balanced_train_segments", 'balanced_train_segments'])
    procs = []
    for i in range(1,41):
        if i<10:
            i = "0"+ str(i)
        else:
            i = str(i)
        procs.append(Process(target=download_wavs, args = ["audioset/metadata/unbalanced_train_segments/unbalanced_train_segments_part"+ i +".csv", "audioset/audios/unbalanced_train_segments", 'unbalanced_train_segments']))
    #p3 = Process(target=download_wavs, args = ["audioset/metadata/unbalanced_train_segments.csv", "audioset/audios/unbalanced_train_segments", 'unbalanced_train_segments'])
    #p1.start()
    #p2.start()
    
    for p in procs:
        p.start()
    #p1.join()
    #p2.join()
    for p in procs:
        p.join()
    #p3.join()
    
    
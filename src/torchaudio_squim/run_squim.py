import os
import wave
import glob
import tqdm
import math
import random
import numpy as np
import pandas as pd
import librosa

import torch
import torchaudio
from torchaudio.prototype.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE

from url_loader import YoutubeLoader

import pdb

DAPS_N_CLEAN_WAV_NUM = 100

def set_seeds(seed=777, multi_gpu=False):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if multi_gpu:
        torch.cuda.manual_seed_all(seed) # if use multi-GPU

def get_wav_duration(audio_path):
    with wave.open(audio_path) as handle:
        t = handle.getnframes() / handle.getframerate()
    return t

def load_audio(audio_file_path, sr=16000, chunk_time=0, mono=True):
    t = get_wav_duration(audio_file_path)

    # if chunk_time > t, we just use all frame ("padding mode not provided")
    if chunk_time != 0 and t > chunk_time:
        # randomly select start time given in Uniform(0, t-chunk_time)
        n_frames = t*sr
        n_chunk_frames = int(chunk_time*sr)
        start = np.random.randint(0, max(0, n_frames-n_chunk_frames))
        start_t = start/sr

        waveform, _ = librosa.load(audio_file_path, offset=start_t, duration=chunk_time, sr=sr, mono=mono)
    else:
        waveform, _ = librosa.load(audio_file_path, sr=sr, mono=mono)

    return waveform


### you should run on recently version of torchaudio
### which is not released yet, only exists in github repository

def run_squim_objective(root_wav_dir, result_csv_dir, device, wav_sr=16000, use_round=True):


    model = SQUIM_OBJECTIVE.get_model()
    model.to(device)
    model.eval()

    csv_list = sorted(glob.glob(result_csv_dir+"/*.csv"))
    for csv_path in tqdm.tqdm(csv_list):
        yt_name = os.path.splitext(os.path.basename(csv_path))[0]
        wav_path = os.path.join(root_wav_dir, yt_name, 'wav', yt_name+'.wav')
        assert(os.path.exists(wav_path)), "No Exists Wav File: {}".format(wav_path)

        df = pd.read_csv(csv_path)

        waveform, sr = torchaudio.load(wav_path)
        waveform = waveform.to(device)

        result_dict = {}
        score_names = ["STOI", "PESQ", "SI-SDR"]
        keys = []
        for name in score_names:
            key = "SQUIM_{}".format(name)
            result_dict[key] = []
            keys.append(key)

        for start_t, end_t in zip(df['start'], df['end']):

            seg_start = int(start_t * wav_sr)
            seg_end = int(end_t * wav_sr)

            scores_dict = {}
            for key in keys:
                scores_dict[key] = []
            
            seg_waveform = waveform[:,seg_start:seg_end]

            window_time = 3.0
            window_size = int(wav_sr * window_time)
            stride_size = int(0.5 * window_size)

            n_frames = int(seg_waveform.shape[1])
            n_chunks = max(1, math.ceil((n_frames-window_size+stride_size)/stride_size))
            for chunk_id in range(n_chunks):
                start = int(stride_size * chunk_id)
                end = min(start + window_size, n_frames)
                chunk_waveform = seg_waveform[:, start:end]
                with torch.no_grad():
                    scores = model(chunk_waveform)
                    for (key, score) in zip(keys, scores):
                        score = score.detach().cpu().item()
                        scores_dict[key].append(score)
                
            scores = []
            for key in keys:
                scores.append(np.mean(scores_dict[key]))

            for (key, score) in zip(keys, scores):
                result_dict[key].append(score)
                print("{}: {:.3f}, ".format(key, score), end='')
            print("")
        
        for key in keys:
            df[key] = np.array(result_dict[key])

        if use_round:
            df = df.round(3)
        
        df.to_csv(csv_path)
    
    print(">>Done SQUIM OBJECTIVE ESTIMATION.")

def run_squim_subjective(root_wav_dir, result_csv_dir, nmr_wav_arr, device, wav_sr=16000, use_round=True):

    device = torch.device("cuda")

    model = SQUIM_SUBJECTIVE.get_model()
    model.to(device)
    model.eval()

    csv_list = sorted(glob.glob(result_csv_dir+"/*.csv"))
    for csv_path in tqdm.tqdm(csv_list):
        yt_name = os.path.splitext(os.path.basename(csv_path))[0]
        wav_path = os.path.join(root_wav_dir, yt_name, 'wav', yt_name+'.wav')
        assert(os.path.exists(wav_path)), "No Exists Wav File: {}".format(wav_path)

        df = pd.read_csv(csv_path)

        waveform, sr = torchaudio.load(wav_path)
        nmr_waveform = torch.FloatTensor(nmr_wav_arr).to(device)

        mos_score_list = []
        for seg_start_t, seg_end_t in zip(df['start'], df['end']):

            seg_start = int(seg_start_t * wav_sr)
            seg_end = int(seg_end_t * wav_sr)

            seg_waveform = waveform[:,seg_start:seg_end]
            
            n_test_frames = seg_waveform.shape[1]
            chunk_size = nmr_waveform.shape[1]
            step_size = (chunk_size * 0.5)

            current_id = 0
            mos_scores = []
            n_chunk = max(1, int(math.ceil((n_test_frames-chunk_size+step_size)/step_size)))
            for chunk_id in range(n_chunk):
                start = int(step_size * chunk_id)
                end = min(start + chunk_size, n_test_frames)
                duration = int(end-start)

                chunk_test_waveform = seg_waveform[:, start:end]
                chunk_test_waveform = chunk_test_waveform.repeat(nmr_wav_arr.shape[0], 1)

                chunk_test_waveform = chunk_test_waveform.to(device)

                chunk_nmr_waveform = nmr_waveform[:,:duration]
                batch_size = 32
                n_samples = chunk_test_waveform.shape[0]
                n_chunk = int((n_samples-1)/batch_size) + 1
                scores = []
                with torch.no_grad():
                    for chunk_id in range(n_chunk):
                        b_start = chunk_id * batch_size
                        b_end = min((chunk_id+1) * batch_size, n_samples)
                        score = model(chunk_test_waveform[b_start:b_end], chunk_nmr_waveform[b_start:b_end])
                        scores.append(score)
                scores = torch.concat(scores)
                score = scores.mean().detach().cpu().item()
                mos_scores.append(score)
            
            final_score = np.mean(mos_scores)
            print("SQUIM_MOS: {}".format(final_score))

            mos_score_list.append(final_score)
        
        df['SQUIM_MOS'] = np.array(mos_score_list)

        if use_round:
            df = df.round(3)
        
        df.to_csv(csv_path)


if __name__ == '__main__':
    """
    Get an argument parser.
    """
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sr', type=int, default=16000, required = False, help='sampling rate')
    parser.add_argument('--yt_url', type=str, default='https://www.youtube.com/watch?v=jane6C4rIwc', required=False, help='path of test wav file')
    parser.add_argument('--yt_dir', type=str, default='data/youtube', required=False, help='mp4 download directory')
    parser.add_argument('--num_threads', type=int, default=0, required = False, help='number of threads')

    args = parser.parse_args().__dict__
    # url = 'https://www.youtube.com/watch?v=jane6C4rIwc'
    # url = 'https://www.youtube.com/playlist?list=PLrT4uvwaf6uw5ChxpBQnx0dA5fcmXvuB_'
    # 냥이아빠
    url = 'https://www.youtube.com/playlist?list=PL-28pfEORGTTyRFb-HLE-xlugbi8nDBb3'

    args['yt_url'] = url

    yt_url: str = args['yt_url']

    yl = YoutubeLoader(args)
    yt_dir_list = yl(yt_url)

    wav_dir = '/mnt/labelmaker/labelmaker/data/youtube'
    csv_dir = '/mnt/labelmaker/labelmaker/exps/csd/csv'

    wav_sr = 16000
    use_round = True
    max_nmr_wav_time = 3.0
    device = torch.device("cuda")
    
    set_seeds()

    nmr_dir = '/mnt/dataset/daps'
    nmr_wav_npy = os.path.join(nmr_dir, 'clean_nmr_n100_{}ms.npy'.format(max_nmr_wav_time*1000))
    if not os.path.exists(nmr_wav_npy):

        print(">>Prepare nmr waveforms")
        nmr_wav_list = sorted(glob.glob(nmr_dir+"/daps/clean/*.wav"))
        nmr_wav_list = [wav_path for wav_path in nmr_wav_list
            if not wav_path.startswith('.')]
        
        assert(len(nmr_wav_list) == DAPS_N_CLEAN_WAV_NUM), "Error not match NMR wav file number: {} : 100".foramt(len(nmr_wav_list))
        
        nmr_wav_arr = []
        for nmr_wav_path in nmr_wav_list:

            nmr_waveform = load_audio(nmr_wav_path, sr=wav_sr, chunk_time=max_nmr_wav_time)
            # nmr_waveform shape: (wav_sr*max_nmr_wav_time,)
            nmr_wav_arr.append(nmr_waveform)

        nmr_wav_arr = np.stack(nmr_wav_arr)

        np.save(nmr_wav_npy, nmr_wav_arr)
    else:
        print(">>Load prepared clean nmr waveforms")
        nmr_wav_arr = np.load(nmr_wav_npy)

    # run_squim_objective(wav_dir, csv_dir, device, wav_sr=wav_sr, use_round=use_round)

    run_squim_subjective(wav_dir, csv_dir, nmr_wav_arr, device, wav_sr=wav_sr, use_round=use_round)

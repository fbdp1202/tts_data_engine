import os
import wave
import glob
import random
import numpy as np
import pandas as pd
import librosa
import math

import torch
import torchaudio
from torchaudio.prototype.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE

import pdb

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
    for csv_path in csv_list:
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

            start = int(start_t * wav_sr)
            end = int(end_t * wav_sr)
            with torch.no_grad():
                seg_waveform = waveform[:,start:end]
                scores = model(seg_waveform)

            for (key, score) in zip(keys, scores):
                score = score.detach().cpu().item()
                result_dict[key].append(score)
                print("{}: {}".format(key, score), end='')
            print("")
        
        for key in keys:
            df[key] = np.array(result_dict[key])

        if use_round:
            df = df.round(3)
        
        df.to_csv(csv_path)
    
    print(">>Done SQUIM OBJECTIVE ESTIMATION.")

def run_squim_subjective(root_wav_dir, result_csv_dir, nrm_wav_arr, device, wav_sr=16000, use_round=True):

    device = torch.device("cuda")

    model = SQUIM_SUBJECTIVE.get_model()
    model.to(device)
    model.eval()

    csv_list = sorted(glob.glob(result_csv_dir+"/*.csv"))
    for csv_path in csv_list:
        yt_name = os.path.splitext(os.path.basename(csv_path))[0]
        wav_path = os.path.join(root_wav_dir, yt_name, 'wav', yt_name+'.wav')
        assert(os.path.exists(wav_path)), "No Exists Wav File: {}".format(wav_path)

        df = pd.read_csv(csv_path)

        waveform, sr = torchaudio.load(wav_path)
        waveform = waveform.to(device)
        nrm_waveform = torch.FloatTensor(nrm_wav_arr).to(device)

        mos_score_list = []
        for seg_start_t, seg_end_t in zip(df['start'], df['end']):

            seg_start = int(seg_start_t * wav_sr)
            seg_end = int(seg_end_t * wav_sr)

            seg_waveform = waveform[:,seg_start:seg_end]
            
            n_test_frames = seg_waveform.shape[1]
            chunk_size = nrm_waveform.shape[1]
            step_size = (chunk_size * 0.5)

            current_id = 0
            mos_scores = []
            n_chunk = max(1, int(math.ceil((n_test_frames-chunk_size+step_size)/step_size)))
            for chunk_id in range(n_chunk):
                start = int(step_size * chunk_id)
                end = min(start + chunk_size, n_test_frames)
                duration = int(end-start)

                print(start, end, duration)
                chunk_test_waveform = seg_waveform[:, start:end]
                chunk_test_waveform = chunk_test_waveform.repeat(nrm_wav_arr.shape[0], 1)

                chunk_nmr_waveform = nrm_waveform[:,:duration]
                with torch.no_grad():
                    print(chunk_test_waveform.shape, chunk_nmr_waveform.shape)
                    score = model(chunk_test_waveform, chunk_nmr_waveform)
                score = score.mean().detach().cpu().item()
                mos_scores.append(score)
            
            final_score = np.mean(mos_scores)
            print("SQUIM_MOS: {}".format(final_score))

            mos_score_list.append(final_score)
        
        df['SQUIM_MOS'] = np.array(mos_score_list)

        if use_round:
            df = df.round(3)
        
        df.to_csv(csv_path)


if __name__ == '__main__':
    root_wav_dir = '/mnt/labelmaker/labelmaker/data/youtube'
    result_csv_dir = '/mnt/labelmaker/labelmaker/exps/csd/csv'
    wav_sr = 16000
    use_round = True
    max_nrm_wav_time = 3
    device = torch.device("cuda")
    
    set_seeds()

    nmr_dir = '/mnt/dataset/daps'
    nmr_wav_npy = os.path.join(nmr_dir, 'clean_nrm_n100_{}s.npy'.format(max_nrm_wav_time))
    if not os.path.exists(nmr_wav_npy):
        print(">>Prepare nmr waveforms")
        wav_nmr_list = [wav_path for wav_path in sorted(glob.glob(nmr_dir+"/daps/clean/*.wav"))
            if not wav_path.startswith('.')]
        
        assert(len(wav_nmr_list) == 100), "Error not match NMR wav file number: {} : 100".foramt(len(wav_nmr_list))
        
        nrm_wav_arr = []
        for wav_nmr_path in wav_nmr_list:
            nrm_waveform = load_audio(wav_nmr_path, sr=wav_sr, chunk_time=max_nrm_wav_time)
            # nrm_waveform shape: (wav_sr*max_nrm_wav_time,)
            nrm_wav_arr.append(nrm_waveform)
        nrm_wav_arr = np.stack(nrm_wav_arr)
        np.save(nmr_wav_npy, nrm_wav_arr)
    else:
        print(">>Load prepared clean nmr waveforms")
        nrm_wav_arr = np.load(nmr_wav_npy)

    run_squim_objective(root_wav_dir, result_csv_dir, device, wav_sr=wav_sr, use_round=use_round)

    run_squim_subjective(root_wav_dir, result_csv_dir, nrm_wav_arr, device, wav_sr=wav_sr, use_round=use_round)
    

        
    



import os
import wave
import random
import librosa
import numpy as np

import torch


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

def myfloor(x, p):
    v = 10**p
    return int(x * v)/v

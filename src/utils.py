import os
import time
import wave
import random
import librosa
import numpy as np

import torch

from pyannote.core import Annotation, Segment


def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print("[{}]: {} sec".format(original_fn.__name__, end_time-start_time))
        return result
    return wrapper_fn

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

def load_annot_from_lab(save_lab_path, spk_id='Speech'):
    seg_arr = np.atleast_2d(np.loadtxt(save_lab_path, usecols=(0, 1)))
    annot = Annotation()
    for (start, end) in zip(seg_arr[:,0], seg_arr[:,1]):
        annot[Segment(start, end)] = spk_id
    return annot

# def load_annot_from_rttm(save_rttm_path):
#     seg_arr = np.atleast_2d(np.loadtxt(save_rttm_path, usecols=(3, 4, 7)))
#     annot = Annotation()
#     for (start, end, spk_id) in zip(seg_arr[:,0], seg_arr[:,1], seg_arr[:,2]):
#         annot[Segment(start, end)] = spk_id
#     return annot

def myfloor(x, p):
    v = 10**p
    return int(x * v)/v

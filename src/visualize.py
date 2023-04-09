import os
import json
import pickle
import numpy as np
import librosa
import librosa.display

from pyannote.core import Annotation, Segment
from pyannote.core import notebook

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import pdb

def get_labels(json_path):
    gt_diar = Annotation()

    with open(json_path) as rf:
        gt = json.load(rf)

    annotations = gt[0]['annotations'][0]['result']
    for annot in annotations:
        start   = annot['value']['start']
        end     = annot['value']['end']
        spk_id  = annot['value']['labels'][0]
        gt_diar[Segment(start, end)] = spk_id
    return gt_diar


def get_wspx_result(result):
    wspx_diar = Annotation()

    segments = result['word_segments']
    for seg in segments:
        start, end  = seg['start'], seg['end']
        spk_id      = seg['text'].split(':')[0].replace('[','').replace(']','')
        wspx_diar[Segment(start, end)] = spk_id
    return wspx_diar


def plot_waveform(y, ax, xlabel=None, ylabel=None, tight=False):
    ax.plot(y)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if tight:
        ax.autoscale(enable=True, axis='x', tight=True)
    ax.xaxis.set_visible(False)


def plot_spectogram(S_dB, sample_rate, ax):
    librosa.display.specshow(S_dB, sr=sample_rate, x_axis='time', y_axis='hz', ax=ax)
    ax.autoscale(enable=True, axis='x', tight=True)


def custom_plot(plot_func, data, ax, view_size=None, tight=False, time=True, legend=False):
    if legend:
        plot_func(data, ax=ax, time=time, legend=legend)
    else:
        plot_func(data, ax=ax, time=time)
    
    if view_size is not None:
        ax.set_xlim([0, view_size])
    elif tight:
        ax.autoscale(enable=True, axis='x', tight=True)
    ax.xaxis.set_visible(False)


def plot_annotations(annotations, save_fig_path, view_size=None):
    num_annotation = len(annotations)
    if num_annotation == 0:
        print("Empty annotation. There are no plot annotation")
        return

    fig, axes = plt.subplots(nrows=num_annotation, ncols=1, figsize=(30, num_annotation*4))
    if num_annotation == 1:
        axes = [axes]

    for _, (ax, annotation) in enumerate(zip(axes, annotations)):
        custom_plot(notebook.plot_annotation, annotation, ax, view_size=view_size)
    
    fig.tight_layout()
    fig.savefig(save_fig_path)
    plt.close('all')


def viewer(audio_file_path, result):
    label_path  = '/mnt/whisper/whisper/The_Dark_Knight_gt.json'
    gt_diar     = get_labels(label_path)

    wspx_diar   = get_wspx_result(result)

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(30, 18))

    y, sample_rate = librosa.load(audio_file_path)
    t = view_size = ((y.shape[0]*100)//sample_rate)/100.0
    y = y[:int(sample_rate*t)]
    S_dB = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=4096, win_length=4096, hop_length=512)), ref=np.max)

    if view_size > 0:
        notebook.crop = Segment(0, view_size)

    plot_waveform(y, axes[0], xlabel='sample rate * time', ylabel='energy', tight=True)
    plot_spectogram(S_dB, sample_rate, axes[1])
    custom_plot(notebook.plot_annotation,   wspx_diar,  axes[2], view_size=view_size)
    custom_plot(notebook.plot_annotation,   gt_diar,    axes[3], view_size=view_size)
    
    fig.tight_layout()
    fig.savefig('annotation.png')


if __name__ == "__main__":
    audio_path = '/mnt/whisper/whisper/The_Dark_Knight.wav'

    os.makedirs('feat', exist_ok=True)
    pkl_path = os.path.join('feat',os.path.basename(audio_path).replace('.wav','.pkl'))
    assert os.path.exists(pkl_path), f'No Exists reusult file: {pkl_path}'

    with open(pkl_path, 'rb') as handle:
        result = pickle.load(handle)

    viewer(audio_path, result)

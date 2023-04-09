import os
import time
import pickle
import numpy as np
import soundfile as sf
import librosa

import torch

from pyannote.audio import Pipeline
from pyannote.audio import Model, Pipeline

from whisperx.vad import load_vad_model, Binarize

from .utils import get_wav_duration, myfloor

import pdb


class SpeechDetector:
    def __init__(self, args):
        # VAD config
        device: str = args['device']
        hf_token: str = args['hf_token']

        self.tmp_dir: str = args['vad_tmp_dir']
        self.save_lab_dir: str = args['vad_save_lab_dir']
        os.makedirs(self.tmp_dir, exist_ok=True)
        os.makedirs(self.save_lab_dir, exist_ok=True)

        self.sr: int = args['sr']
        self.vad_onset: float = args['vad_onset']
        self.vad_offset: float = args['vad_offset']
        self.pad_onset: float = args['vad_pad_onset']
        self.pad_offset: float = args['vad_pad_offset']
        
        # VAD setup
        self.vad_model = load_vad_model(torch.device(device), self.vad_onset, 
                                        self.vad_offset, use_auth_token=hf_token)
        self.binarize = Binarize(pad_onset=self.pad_onset, pad_offset=self.pad_offset)

    def run_segmentation(self, input_audio_path):
        print(">>Performing VAD...")
        segments = self.vad_model(input_audio_path)
        return segments

    def check_audio_file(self, audio_path):
        input_audio_path = audio_path
        if not audio_path.endswith(".wav"):
            print(">>VAD requires .wav format, converting to wav as a tempfile...")
            # tfile = tempfile.NamedTemporaryFile(delete=True, suffix=".wav")
            audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
            if self.tmp_dir is not None:
                input_audio_path = os.path.join(self.tmp_dir, audio_basename + ".wav")
            else:
                input_audio_path = os.path.join(os.path.dirname(audio_path), audio_basename + ".wav")
            ffmpeg.input(audio_path, threads=0).output(input_audio_path, ac=1, ar=self.sr).run(cmd=["ffmpeg"])
        
        return input_audio_path

    def padding_vad(self, segments, wav_duration):
        segments = self.binarize(segments, wav_duration)
        return segments

    def apply_vad_segments(self, binary_segments, input_file_path, out_file_path):
        audio_arr, _  = librosa.load(input_file_path, sr=None)

        audio_seg_list = []
        for seg in binary_segments.get_timeline():
            start_f = int(seg.start * self.sr)
            end_f = int(seg.end * self.sr)
            audio_seg_list.append(audio_arr[start_f:end_f])
        
        vad_audio_arr = np.concatenate(audio_seg_list)
        sf.write(out_file_path, vad_audio_arr, self.sr)

    @staticmethod
    def plot_vad_result(annotation, fig_save_path='test_vad.png', verbose=False):
        from visualize import plot_annotations
        
        annotations = [annotation]
        plot_annotations(annotations, fig_save_path)

        if verbose:
            print('>>Plot VAD anntations: {}'.format(fig_save_path))

    @staticmethod
    def save_vad_result(annotation, save_lab_path, verbose=False):
        with open(save_lab_path, "w") as wf:
            annotation.write_lab(wf)
        
        if verbose:
            print('>>Save VAD results: {}'.format(save_lab_path))

    def __call__(self, audio_file_path, visualize=False, save_vad=True, verbose=False):

        input_audio_path = self.check_audio_file(audio_file_path)

        wav_duration = myfloor(get_wav_duration(input_audio_path), 3)

        vad_segments = self.run_segmentation(input_audio_path)

        binarized_segments = self.padding_vad(vad_segments, wav_duration)

        if visualize:
            # plot vad results
            self.plot_vad_result(binarized_segments)

        if save_vad:
            # write vad segment in .lab file
            save_lab_name = os.path.splitext(os.path.basename(audio_file_path))[0]+'.lab'
            save_lab_path = os.path.join(self.save_lab_dir, save_lab_name)

            print(">>Save VAD result in {}".format(save_lab_path))
            self.save_vad_result(binarized_segments, save_lab_path, verbose=verbose)

        return binarized_segments, input_audio_path

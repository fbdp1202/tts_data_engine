#Copyright (c) Meta Platforms, Inc. and affiliates.
#All rights reserved.

#This source code is licensed under the license found in the
#LICENSE file in the root directory of this source tree.

import os
import sys
import math
import glob
import tqdm
import pickle
import tarfile
import hashlib
import subprocess
import numpy as np

from scipy import signal
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F

from .noresqa.model import NORESQA
from .utils import load_audio

DAPS_DATASET_URL = 'https://zenodo.org/record/4660670/files/daps.tar.gz?download=1'

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def check_daps_dataset(tar_file_path):
    md5ck = md5(tar_file_path)
    md5gt = '303c130b7ce2e02b59c7ca5cd595a89c'
    if md5ck == md5gt:
        print('Checksum successful {}.'.format(tar_file_path))
    else:
        raise Warning('Checksum failed {}.'.format(tar_file_path))

def download_daps_dataset(out_dir):

    out_file_path = os.path.join(out_dir, 'daps.tar.gz')
    out = subprocess.call('wget {} -O {}'.format(DAPS_DATASET_URL, out_file_path), shell=True)
    if out != 0:
        raise ValueError('Download failed {}.'.format(url))
    
    check_daps_dataset(out_file_path)

def extract_targz_file(out_dir, tar_file_path):
    with tarfile.open(tar_file_path, "r:gz") as tar:
        subdir_and_files = [
            tarinfo for tarinfo in tar.getmembers()
            if tarinfo.name.startswith("daps/clean/")
        ]
        tar.extractall(out_dir, members=subdir_and_files)

def prepare_daps_dataset(nmr_wav_dir):
    os.makedirs(nmr_wav_dir, exist_ok=True)
    if not os.path.exists(os.path.join(nmr_wav_dir, '.done')):
        tar_file_path = os.path.join(nmr_wav_dir, 'daps.tar.gz')
        if not os.path.exists(tar_file_path):
            download_daps_dataset(nmr_wav_dir)
        
        extract_targz_file(nmr_wav_dir, tar_file_path)

        f = open(os.path.join(nmr_wav_dir, '.done'), 'wb')
        f.close()


class SpeechQualityAssigner:

    def __init__(self, args):

        output_dim: int = 40
        ssl_model_path: str = args['sqa_ssl_model_path']
        device: str = args['device']
        self.device = torch.device(device)
        self.sr: int = args['sr']
        self.model_ckpt_path: str = args['sqa_model_ckpt_path']
        self.nmr_chunk_time: float = args['sqa_nmr_chunk_time']
        self.emb_step_size: int = args['sqa_emb_step_size']

        nmr_wav_dir: str = args['sqa_nmr_wav_dir']
        nmr_feat_path: str = args['sqa_nmr_feat_path']
        os.makedirs(os.path.dirname(nmr_feat_path), exist_ok=True)

        # prepare daps dataset
        prepare_daps_dataset(nmr_wav_dir)

        self.sqa_model = NORESQA(output=output_dim, output2=output_dim, config_path=ssl_model_path)
        self.load_parameter(self.model_ckpt_path)
        self.sqa_model.to(self.device)
        self.sqa_model.eval()

        nmr_embs = self.load_nmr_emb(nmr_feat_path, nmr_wav_dir)
        self.nmr_embs = nmr_embs.to(self.device)

    def load_parameter(self, model_ckpt_path):

        model_dict = self.sqa_model.state_dict()
        params = torch.load(model_ckpt_path, map_location="cpu")['state_dict']

        pretrained_dict = {}
        for name, param in params.items():
            name = name.replace('module.', '') if 'module' in name else name
            pretrained_dict[name] = param
        model_dict.update(pretrained_dict)

        self.sqa_model.load_state_dict(pretrained_dict)

    def pred_noresqa_mos(self, test_feat, nmr_feat=None):

        with torch.no_grad():
            score = self.sqa_model(nmr_feat, test_feat).detach().cpu().numpy()[0]

        return score

    def extract_nmr_embbeddings(self, nmr_wav_dir):

        nmr_wav_list = sorted(glob.glob(nmr_wav_dir+"/*/clean/*.wav"))

        nmr_wav_list = [
            nmr_wav_path for nmr_wav_path in nmr_wav_list
            if not os.path.basename(nmr_wav_path).startswith('.')
        ]

        nmr_embs = []
        for nmr_wav in tqdm.tqdm(nmr_wav_list):
            nrm_feat = load_audio(nmr_wav, chunk_time=self.nmr_chunk_time, sr=self.sr)
            nrm_feat = torch.FloatTensor(nrm_feat).to(self.device).unsqueeze(0)

            with torch.no_grad():
                nmr_emb = self.sqa_model.extract_embeddings(nrm_feat)
            nmr_embs.append(nmr_emb.detach().cpu())

        nmr_embs = torch.vstack(nmr_embs)
        return nmr_embs

    def load_nmr_emb(self, nmr_feat_path, nmr_wav_dir, overwrite=False):

        if overwrite or not os.path.exists(nmr_feat_path):
            nmr_embs = self.extract_nmr_embbeddings(nmr_wav_dir)
            with open(nmr_feat_path, 'wb') as wf:
                pickle.dump(nmr_embs, wf, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(nmr_feat_path, 'rb') as rf:
                nmr_embs = pickle.load(rf)

        return nmr_embs

    def estimate_score(self, waveform):
        """
        Parameters
        ----------
        waveform: torch.FloatTensor (n_samples,)
            Input Raw Waveform.
        Returns
        ----------
        mos_score : float
            Detection score.
        """
        
        waveform = waveform.to(self.device).unsqueeze(0)
        with torch.no_grad():
            nmr_embs = self.nmr_embs
            test_embs = self.sqa_model.extract_embeddings(waveform)
            test_embs = test_embs.repeat(nmr_embs.shape[0], 1, 1)

            n_test_frames = test_embs.shape[2]
            n_nmr_frames = self.nmr_embs.shape[2]
            emb_step_size = self.emb_step_size

            mos_scores = []
            n_chunk = max(1, int(math.ceil(n_test_frames-n_nmr_frames+emb_step_size)/emb_step_size))
            for n in range(n_chunk):
                start = emb_step_size*n
                end = min(start + n_nmr_frames, n_test_frames)
                input_test_embs = test_embs[:,:,start:end]
                results = self.sqa_model.estimate_score_bw_embs(nmr_embs[:,:,:end-start], input_test_embs)
                mos_score = results['mos_score'].mean().detach().cpu().item()
                mos_scores.append(mos_score)
            final_mos_score = np.mean(mos_scores)
        return final_mos_score

    def __call__(self, input_audio_path, seg_arr=None, verbose=False):

        waveform = load_audio(input_audio_path, sr=self.sr)
        waveform = torch.FloatTensor(waveform)
        if seg_arr is not None:
            mos_score_list = []
            for start, end in zip(seg_arr[:,0], seg_arr[:,1]):
                seg_waveform = waveform[start:end]
                mos_score = self.estimate_score(seg_waveform)

                start_t, end_t = start/self.sr, end/self.sr
                mos_score_list.append([start_t, end_t, mos_score])
                if verbose:
                    print("{:5.3f} - {:5.3f} ({:3.3f}) : {:.3f}".format(start_t, end_t, end_t-start_t, mos_score))
            mos_score = np.array(mos_score_list)
        else:
            mos_score = self.estimate_score(waveform)
        return mos_score


if __name__ == '__main__':
    """
    Get an argument parser.
    """
    import argparse
    from utils import set_seeds
    from whisper.audio import SAMPLE_RATE

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # basic config
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="device to use for PyTorch inference")
    parser.add_argument("--seed", type=int, default=777, help="seed number")
    parser.add_argument('--test_wav_path', type=str, default='/mnt/FRCRN/The_Dark_Knight.wav', required=False, help='path of test wav file')
    # parser.add_argument('--test_wav_path', type=str, default='/mnt/FRCRN/The_Dark_Knight_SE_FRCRN.wav', required=False, help='path of test wav file')
    parser.add_argument('--test_lab_path', type=str, default='/mnt/FRCRN/The_Dark_Knight_SE_FRCRN.lab', required=False, help='path of test wav file')
    parser.add_argument('--sr', type=int, default=SAMPLE_RATE, required = False, help='sampling rate')
    parser.add_argument("--exp_dir", type=str, default='exps', help="path to experiments directory")

    # speech quality assessment config
    parser.add_argument("--sqa_ssl_model_path", type=str, default='models/sqa_models/wav2vec_small.pt', help="pretrained wav2vec base model path")
    parser.add_argument("--sqa_model_ckpt_path", type=str, default='models/sqa_models/model_noresqa_mos.pth', help="pretrained NORESQA-MOS model path")
    parser.add_argument('--sqa_nmr_wav_dir', type=str, default='/mnt/dataset/daps', required = False, help='path of clean wav file')
    parser.add_argument('--sqa_nmr_feat_path', type=str, default='sqa/noresqa/feat/daps_nmr_embs.pkl', required = False, help='path of nmr embedding pickle file')
    parser.add_argument("--sqa_nmr_chunk_time", type=float, default=3.0, help="nmr wav chunk time")
    parser.add_argument("--sqa_emb_step_size", type=int, default=50, help="embedding step size")

    args = parser.parse_args().__dict__
    args['sqa_nmr_feat_path'] = os.path.join(args['exp_dir'], args['sqa_nmr_feat_path'])

    set_seeds(args['seed'])

    test_wav_path: str = args.pop('test_wav_path')
    assert(os.path.exists(test_wav_path)), "No Exists File Name: {}".format(test_wav_path)

    test_lab_path: str = args.pop('test_lab_path')
    seg_arr = np.atleast_2d((np.loadtxt(test_lab_path, usecols=(0, 1))*args['sr']).astype(int))

    sqa_manager = SpeechQualityAssigner(args)
    score = sqa_manager(test_wav_path, seg_arr)

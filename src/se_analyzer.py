import os
import subprocess
from zipfile import ZipFile
import glob

import torch
from torchmetrics import (
    SignalNoiseRatio,
    ScaleInvariantSignalNoiseRatio,
)
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

from enhance import SpeechEnhancer
from utils import load_audio

import pdb


# valentini dataset homepage: https://datashare.ed.ac.uk/handle/10283/2791
VALENTINI_DATASET_URLS = {}
VALENTINI_DATASET_URLS['clean_testset_wav.zip'] = 'https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip?sequence=1&isAllowed=y'
VALENTINI_DATASET_URLS['noisy_testset_wav.zip'] = 'https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip?sequence=5&isAllowed=y'
VALENTINI_DATASET_URLS['testset_txt.zip'] = 'https://datashare.ed.ac.uk/bitstream/handle/10283/2791/testset_txt.zip?sequence=8&isAllowed=y'
VALENTINI_DATASET_URLS['LICENSE'] = 'https://datashare.ed.ac.uk/bitstream/handle/10283/2791/license_text?sequence=11&isAllowed=y'


def download_valentini_dataset(out_dir):
    for (key, url) in VALENTINI_DATASET_URLS.items():
        out_file_path = os.path.join(out_dir, key)
        os.makedirs(os.path.dirname(out_file_path), exist_ok=True)

        if os.path.exists(out_file_path):
            print(">>Already Exists File: {}".format(out_file_path))
            continue

        print(">>Download File Name: {}".format(key))
        
        cmd = 'wget -O {} {}'.format(out_file_path, url)
        out = subprocess.call(cmd, shell=True)
        if out != 0:
            raise ValueError("Download Failed {}.".format(url))

def full_zip_extract(out_dir):
    
    for (key, url) in VALENTINI_DATASET_URLS.items():

        if not key.endswith(".zip"):
            continue

        out_dir_path = os.path.join(out_dir, os.path.splitext(key)[0])
        if os.path.exists(out_dir_path):
            print(">>Already Exists Directory: {}".format(out_dir_path))
            continue

        zip_file_path = os.path.join(out_dir, key)

        print(">>Extract Zip Name: {}".format(zip_file_path))
        with ZipFile(zip_file_path, 'r') as zf:
            zf.extractall(out_dir)

def prepare_valentini_dataset(out_dir, overwrite=False):
    
    if os.path.exists(out_dir+"/.done") and not overwrite:
        print(">>Already Exists Valentini Dataset in {}".format(out_dir))
        return

    download_valentini_dataset(out_dir)

    full_zip_extract(out_dir)

    f = open(out_dir+"/.done", 'w')
    f.close()


class SE_Analyzer:

    def __init__(self, args):
    
        self.sr: str = args['sr']
        self.data_dir: str = args["se_dataset_dir"]
        self.se_out_postfix: str = args['se_out_postfix']

        prepare_valentini_dataset(self.data_dir)

        self.clean_dir = os.path.join(self.data_dir, "clean_testset_wav")
        self.noisy_dir = os.path.join(self.data_dir, "noisy_testset_wav")

        self.se_dir = os.path.join(self.data_dir, "se_testset_wav")
        os.makedirs(self.se_dir, exist_ok=True)

        # self.prepare_se_dataset(args)

        self.snr = SignalNoiseRatio()
        self.sisnr = ScaleInvariantSignalNoiseRatio()
        self.stoi = ShortTimeObjectiveIntelligibility(self.sr, extended=False)
        self.pesq = PerceptualEvaluationSpeechQuality(\
            self.sr, 'nb' if self.sr < 16000 else 'wb')

        self.matrics = {}
        self.matrics['snr'] = self.get_SNR
        self.matrics['sisnr'] = self.get_SI_SNR
        self.matrics['stoi'] = self.get_STOI
        self.matrics['pesq'] = self.get_PESQ

    def prepare_se_dataset(self, args):

        if os.path.exists(self.data_dir+"/.se_done"):
            print(">>Already Prepared SE dataset: {}".format(self.se_dir))
            return

        noisy_wav_list = sorted(glob.glob(self.noisy_dir+'/*.wav'))

        se_manager = SpeechEnhancer(args)
        se_wav_list = se_manager(noisy_wav_list, out_wav_dir=self.se_dir)

        assert(len(se_wav_list) == len(noisy_wav_list)),\
            print("Not Match File Number {} != {}".format(len(se_wav_list), len(noisy_wav_list)))

        f = open(self.data_dir+"/.se_done", 'w')
        f.close()
        print("Finished Prepare SE dataset in {}".format(self.se_dir))

    def get_SNR(self, pred, target):
        return self.snr(pred, target)

    def get_SI_SNR(self, pred, target):
        return self.sisnr(pred, target)

    def get_STOI(self, pred, target):
        return self.stoi(pred, target)

    def get_PESQ(self, pred, target):
        return self.pesq(pred, target)

    def get_all_metrics(self, pred, target):

        result = {}
        for (key, func) in self.matrics.items():
            result[key] = func(pred, target)
        return result

    def __call__(self):
        
        wav_list = [os.path.basename(wav_path).replace(self.se_out_postfix,'') \
            for wav_path in sorted(glob.glob(self.se_dir+"/*.wav"))]
        
        for wav_name in wav_list:
            
            clean_wav_path = os.path.join(self.clean_dir, wav_name)
            noisy_wav_path = os.path.join(self.noisy_dir, wav_name)
            se_wav_path = os.path.join(self.se_dir, wav_name.replace('.wav', self.se_out_postfix+'.wav'))

            assert(os.path.exists(clean_wav_path)), "No Exists Wav Name: {}".format(clean_wav_path)
            assert(os.path.exists(noisy_wav_path)), "No Exists Wav Name: {}".format(noisy_wav_path)
            assert(os.path.exists(se_wav_path)), "No Exists Wav Name: {}".format(se_wav_path)

            clean_wav = load_audio(clean_wav_path)
            noisy_wav = load_audio(noisy_wav_path)
            se_wav = load_audio(se_wav_path)
            
            pos = min(len(clean_wav), len(noisy_wav), len(se_wav))
            clean_wav = torch.FloatTensor(clean_wav[:pos])
            noisy_wav = torch.FloatTensor(noisy_wav[:pos])
            se_wav = torch.FloatTensor(se_wav[:pos])

            assert(clean_wav.shape == noisy_wav.shape), "({})!=({})".format(clean_wav.shape, noisy_wav.shape)
            assert(clean_wav.shape == se_wav.shape), "({})!=({})".format(clean_wav.shape, se_wav.shape)

            cn = self.get_all_metrics(clean_wav, noisy_wav)
            cs = self.get_all_metrics(clean_wav, se_wav)
            ns = self.get_all_metrics(noisy_wav, se_wav)
            print(cn)
            print(cs)
            print(ns)
            pdb.set_trace()
            print('hi')


if __name__ == "__main__":
    """
    Get an argument parser.
    """
    import torch
    import argparse
    from utils import set_seeds
    from whisper.audio import SAMPLE_RATE

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="device to use for PyTorch inference")
    parser.add_argument("--seed", type=int, default=777, help="seed number")
    parser.add_argument("--sr", type=int, default=SAMPLE_RATE, required = False, help="sampling rate")

    parser.add_argument("--se_dataset_dir", type=str, default="/mnt/dataset/velentini", required = False, help="path of valentini")

    # parser.add_argument('--se_out_postfix', type=str, default='', required=False, help='output postfix string')
    parser.add_argument('--se_out_postfix', type=str, default='_SE_FRCRN', required=False, help='output postfix string')

    args = parser.parse_args().__dict__

    set_seeds(args["seed"])

    analyzer = SE_Analyzer(args)
    analyzer()

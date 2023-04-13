import os
import json
import math
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
import torchaudio

from .utils import load_audio
from .beats.BEATs import BEATs, BEATsConfig


class SoundClassifier:
    def __init__(self, args):

        device: str = args['device']
        self.device = torch.device(device)
        self.sr = args['sr']

        # load audioset label infomation
        ontology_file_path: str = args['sc_ontology_file_path']
        labels_indices_csv_path: str = args['sc_labels_indices_csv']
        child_dict, code2name = self.load_info_audioset(ontology_file_path, labels_indices_csv_path)
        self.child_dict = child_dict
        self.code2name = code2name

        # load BEATs
        model_ckpt_path: str = args['beats_model_ckpt_path']
        assert(os.path.exists(model_ckpt_path)), print('No Exists BEATs model file: {}'.format(model_ckpt_path))

        checkpoint = torch.load(model_ckpt_path)

        cfg = BEATsConfig(checkpoint['cfg'])
        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.model.eval()

        self.label_dict = checkpoint['label_dict']

    def load_info_audioset(self, ontology_file_path, labels_indices_csv_path):
        child_dict = self.get_child_dict(ontology_file_path)
        labels = pd.read_csv(labels_indices_csv_path)

        code2name = {
            mid:name
            for mid, name in zip(labels['mid'].to_list(), labels['display_name'].to_list())
        }
        return child_dict, code2name

    @staticmethod
    def get_child_dict(ontology_file_path):
        """
            File: data/ontology.json
            Desciption: AudioSet provide Each Class Information, such as child_ids, restrictions etc.,
            Var:
                'id': encoded class code (index)
                'name': class name
                'restrictions': Class type (None, abstract, blacklist)
        """

        with open(ontology_file_path, 'r', encoding='utf8')as fp:
            ontology = json.load(fp)

        # make dictionary which contain each class information
        child_dict = {}
        for audio_class in ontology:
            cur_id = audio_class['id']
            cur_name = audio_class['name']
            cur_child_ids = audio_class['child_ids']
            # cur_restriction = audio_class['restrictions']
            child_dict[cur_id] = (cur_child_ids, cur_name)
        return child_dict

    def predict(self, waveform, mask=None, chunk_time=1.0, step_ratio=0.1, return_all=False):
        """
        Parameters
        ----------
        waveform: torch.FloatTensor (n_samples,)
            Input Raw Waveform.
        mask: torch.BoolTensor (n_samples,)
            Input Mask
        chunk_time: float
            Chunk time
        step_ratio: float
            Step ratio
        Returns
        ----------
        preds : torch.FloatTensor (n_classes,)
            posterior of sound classification.
        """
        
        chunk_size = int(chunk_time * self.sr)
        step_size = chunk_size * step_ratio

        waveform = waveform.to(self.device).unsqueeze(0)
        n_test_frames = waveform.shape[1]


        pred_list = []
        n_chunk = max(1, int(math.ceil((n_test_frames-chunk_size+step_size)/step_size)))
        for chunk_id in range(n_chunk):

            start = int(step_size * chunk_id)
            end = min(start + chunk_size, n_test_frames)
            duration = int(end-start)

            chunk_waveform = torch.zeros(1,chunk_size).to(self.device)
            chunk_waveform[:,:duration] = waveform[:,start:end]

            chunk_mask = None
            if mask is not None:
                chunk_mask = torch.zeros(1, chunk_size, dtype=torch.bool).to(self.device)
                chunk_mask[:,:duration] = mask[start:end]

            with torch.no_grad():
                pred = self.model.extract_features(chunk_waveform, padding_mask=chunk_mask)[0]
                pred = pred.squeeze(0).detach().cpu()
            pred_list.append(pred)

        preds = torch.stack(pred_list)
        pred = preds.mean(0)

        if return_all:
            return pred, preds
        else:
            return pred

    def pred_topk_with_label(self, waveform, mask=None, topk=5):
        pred = self.predict(waveform, mask=mask)
        probs, indices = pred.topk(k=topk)
        
        codes = [self.label_dict[idx.item()] for idx in indices]
        names = [self.code2name[code] for code in codes]
        results = []
        for (name, code, prob) in zip(names, codes, probs):
            results.append((code, name, prob.item()))
        return results

    def __call__(self, input_audio_path, seg_arr=None):

        waveform = load_audio(input_audio_path, sr=self.sr)
        waveform = torch.FloatTensor(waveform)
        if seg_arr is not None:
            pred_list = []
            for start, end in zip(seg_arr[:,0], seg_arr[:,1]):
                seg_waveform = waveform[start:end]
                pred = self.predict(seg_waveform)

                pred_list.append(pred.numpy())
            pred = np.stack(pred_list)
        else:
            pred = self.predict(waveform)[None,:]
        return pred


if __name__ == "__main__":
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

    # sound classification config
    parser.add_argument('--sc_ontology_file_path', type=str, default='data/BEATs/ontology.json', required=False, help='path of audioset ontology')
    parser.add_argument('--sc_labels_indices_csv', type=str, default='data/BEATs/class_labels_indices.csv', required=False, help='csv file of containing audioset label indices')
    parser.add_argument("--beats_model_ckpt_path", type=str, default='models/sc_models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt', help="pretrained BEATs model path")

    args = parser.parse_args().__dict__

    set_seeds(args['seed'])    
 
    test_wav_path: str = args.pop('test_wav_path')
    assert(os.path.exists(test_wav_path)), "No Exists File Name: {}".format(test_wav_path)

    test_lab_path: str = args.pop('test_lab_path')
    seg_arr = np.atleast_2d((np.loadtxt(test_lab_path, usecols=(0, 1))*args['sr']).astype(int))

    sc_manager = SoundClassifier(args)
    results = sc_manager(test_wav_path, seg_arr)

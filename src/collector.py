import os
import tqdm
import pandas as pd

import torch

from .utils import load_audio

from .vad import SpeechDetector
from .sqa import SpeechQualityAssigner
from .classify import SoundClassifier

import pdb


class CleanSpeechDetector:

    def __init__(self, args):

        self.sr: int = args['sr']
        self.csd_csv_dir: str = args['csd_csv_dir']
        os.makedirs(self.csd_csv_dir, exist_ok=True)

        self.vad_manager = SpeechDetector(args)
        self.sqa_manager = SpeechQualityAssigner(args)
        self.sc_manager = SoundClassifier(args)

    def run_segments(self, input_audio_path, out_vad, topk=5, save_csv=True, use_round=True):

        waveform = load_audio(input_audio_path, sr=self.sr)
        waveform = torch.FloatTensor(waveform)

        columns = ["index", "start", "end", "estimated_MOS"]
        for k in range(topk):
            for name in ['code', 'name', 'pred']:
                columns.append("top{}_{}".format(k+1, name))

        results = {}
        for col in columns:
            results[col] = []

        for idx, seg in tqdm.tqdm(enumerate(out_vad.get_timeline())):
            start_t, end_t = seg.start, seg.end
            start, end = int(start_t*self.sr), int(end_t*self.sr)

            seg_waveform = waveform[start:end]

            mos_score = self.sqa_manager.estimate_score(seg_waveform)
            sc_results = self.sc_manager.pred_topk_with_label(seg_waveform, topk=topk)

            results["index"].append(idx)
            results["start"].append(start_t)
            results["end"].append(end_t)
            results["estimated_MOS"].append(mos_score)
            for k, (code, name, prob) in enumerate(sc_results):
                for key, value in zip(['code', 'name', 'pred'], [code, name, prob]):
                    results["top{}_{}".format(k+1, key)].append(value)

        df = pd.DataFrame.from_dict(results)

        # optional
        if use_round:
            df = df.round(3)

        if save_csv:
            save_csv_name = os.path.splitext(os.path.basename(input_audio_path))[0]+'.csv'
            save_csv_path = os.path.join(self.csd_csv_dir, save_csv_name)

            df.to_csv(save_csv_path)

        return df

    def __call__(self, audio_file_path):

        binarized_segments, input_audio_path = self.vad_manager(audio_file_path)

        df = self.run_segments(input_audio_path, binarized_segments)

        return df


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
    # parser.add_argument('--test_wav_path', type=str, default='/mnt/FRCRN/The_Dark_Knight.wav', required=False, help='path of test wav file')
    parser.add_argument('--test_wav_path', type=str, default='/mnt/FRCRN/The_Dark_Knight_SE_FRCRN.wav', required=False, help='path of test wav file')
    parser.add_argument('--sr', type=int, default=SAMPLE_RATE, required = False, help='sampling rate')
    parser.add_argument("--exp_dir", type=str, default='exps', help="path to experiments directory")

    parser.add_argument("--csd_csv_dir", type=str, default='csd/csv', help="path to experiments directory")

    # vad config
    parser.add_argument("--vad_tmp_dir", default="vad/tmp_wav", help="Temporary directory to write audio file if input if not .wav format (only for VAD).")
    parser.add_argument("--vad_save_lab_dir", default="vad/lab", help="Temporary directory to write audio file if input if not .wav format (only for VAD).")
    parser.add_argument("--hf_token", type=str, default='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', help="Hugging Face Access Token to access PyAnnote gated models")
    parser.add_argument("--vad_onset", type=float, default=0.500, help="Onset threshold for VAD (see pyannote.audio), reduce this if speech is not being detected")
    parser.add_argument("--vad_offset", type=float, default=0.363, help="Offset threshold for VAD (see pyannote.audio), reduce this if speech is not being detected.")
    parser.add_argument("--vad_pad_onset", type=float, default=0.500, help="Padding Onset for VAD (see pyannote.audio)")
    parser.add_argument("--vad_pad_offset", type=float, default=0.500, help="Padding time for VAD (see pyannote.audio)")

    # speech quality assessment config
    parser.add_argument("--sqa_ssl_model_path", type=str, default='models/sqa_models/wav2vec_small.pt', help="pretrained wav2vec base model path")
    parser.add_argument("--sqa_model_ckpt_path", type=str, default='models/sqa_models/model_noresqa_mos.pth', help="pretrained NORESQA-MOS model path")
    parser.add_argument('--sqa_nmr_wav_dir', type=str, default='/mnt/dataset/daps', required = False, help='path of clean wav file')
    parser.add_argument('--sqa_nmr_feat_path', type=str, default='sqa/noresqa/feat/daps_nmr_embs.pkl', required = False, help='path of nmr embedding pickle file')
    parser.add_argument("--sqa_nmr_chunk_time", type=float, default=3.0, help="nmr wav chunk time")
    parser.add_argument("--sqa_emb_step_size", type=int, default=50, help="embedding step size") # '50 frames' means 1.0 sec

    # sound classification config
    parser.add_argument('--sc_ontology_file_path', type=str, default='data/BEATs/ontology.json', required=False, help='path of audioset ontology')
    parser.add_argument('--sc_labels_indices_csv', type=str, default='data/BEATs/class_labels_indices.csv', required=False, help='csv file of containing audioset label indices')
    parser.add_argument("--beats_model_ckpt_path", type=str, default='models/sc_models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt', help="pretrained BEATs model path")

    args = parser.parse_args().__dict__
    args['vad_tmp_dir'] = os.path.join(args['exp_dir'], args['vad_tmp_dir'])
    args['vad_save_lab_dir'] = os.path.join(args['exp_dir'], args['vad_save_lab_dir'])

    args['sqa_nmr_feat_path'] = os.path.join(args['exp_dir'], args['sqa_nmr_feat_path'])

    args['csd_csv_dir'] = os.path.join(args['exp_dir'], args['csd_csv_dir'])


    set_seeds(args['seed'])

    test_wav_path: str = args.pop('test_wav_path')
    assert(os.path.exists(test_wav_path)), "No Exists File Name: {}".format(test_wav_path)
   
    detector = CleanSpeechDetector(args)
    df = detector(test_wav_path)

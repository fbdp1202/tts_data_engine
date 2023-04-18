import os

from pyannote.core import Annotation, Segment
import malaya_speech

from .visualize import plot_annotations
from .custom_pyannote.pipeline import Pipeline


class SpeakerDiarizer:

    def __init__(self, args):
        
        model_name: str = args['diar_model_name']
        hf_token: str = args['hf_token']
        self.exp_dir: str = args['diar_exp_dir']
        os.makedirs(self.exp_dir, exist_ok=True)

        self.se_out_postfix: str = args['se_out_postfix']

        model = Pipeline.from_pretrained(model_name, use_auth_token=hf_token, embedding=args['diar_embedding'])

        self.model = model.to('cuda:0')
        
    def __call__(self, wav_path, save_rttm=True, overwrite=False):

        basename = os.path.splitext(os.path.basename(wav_path))[0]
        exp_dir = os.path.join(self.exp_dir, basename)
        rttm_dir = os.path.join(exp_dir, 'rttm')
        fig_dir = os.path.join(exp_dir, 'fig')
        os.makedirs(rttm_dir, exist_ok=True)
        os.makedirs(fig_dir, exist_ok=True)

        annotations = []

        rttm_path = os.path.join(rttm_dir, basename + '.rttm')
        if not overwrite and os.path.exists(rttm_path):
            print("\n>>SKIP Diarization: {}".format(wav_path))

            malaya_result = malaya_speech.extra.rttm.load(rttm_path)[basename]

            result = Annotation()
            for segment, _, label in malaya_result.itertracks():
                result[Segment(segment.start, segment.end)] = label

        else:
            print("\n>>Run Diarization: {}".format(wav_path))
            result = self.model(wav_path)
            if save_rttm:
                with open(rttm_path, 'wt') as wf:
                    result.write_rttm(wf)
        annotations.append(result)

        # se_wav_path = os.path.splitext(wav_path)[0] + self.se_out_postfix + '.wav'
        # if os.path.exists(se_wav_path):
        #     se_rttm_path = os.path.join(rttm_dir, basename + self.se_out_postfix + '.rttm')
        #     if os.path.exists(se_rttm_path):
        #         se_result = malaya_speech.extra.rttm.load(se_rttm_path)
        #     else:
        #         se_result = self.model(se_wav_path)
        #         if save_rttm:
        #             with open(se_rttm_path, 'wt') as wf:
        #                 se_result.write_rttm(wf)
        #     annotations.append(se_result)

        save_fig_path = os.path.join(fig_dir, basename + '.png')
        if not os.path.exists(save_fig_path) and False:
            plot_annotations(annotations, save_fig_path, wav_path=wav_path)
        return result

if __name__ == '__main__':
    """
    Get an argument parser.
    """
    import argparse
    from utils import set_seeds

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--seed", type=int, default=777, help="seed number")
    # parser.add_argument('--test_wav_path', type=str, default='data/youtube/jane6C4rIwc/wav/jane6C4rIwc.wav', required=False, help='path of test wav file')
    parser.add_argument('--test_wav_path', type=str, default='data/youtube/XNKF1kOnUL4/wav/XNKF1kOnUL4.wav', required=False, help='path of test wav file')

    parser.add_argument("--exp_dir", type=str, default='exps', help="path to experiments directory")

    parser.add_argument("--diar_exp_dir", type=str, default='sd', help="path to diarization experiments directory")
    parser.add_argument("--hf_token", type=str, default='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', help="Hugging Face Access Token to access PyAnnote gated models")
    parser.add_argument('--diar_model_name', type=str, default='pyannote/speaker-diarization@2.1', required=False, help='pretrained speaker diarization model name')

    # speech enhancement config
    parser.add_argument('--se_out_postfix', type=str, default='_SE_FRCRN', required=False, help='output postfix string')

    args = parser.parse_args().__dict__

    set_seeds(args['seed'])

    args['diar_exp_dir'] = os.path.join(args['exp_dir'], args['diar_exp_dir'])

    test_wav_path: str = args.pop('test_wav_path')
    assert(os.path.exists(test_wav_path)), "No Exists File Name: {}".format(test_wav_path)

    diarizer = SpeakerDiarizer(args)
    result = diarizer(test_wav_path)

    print("Diarization Done.")

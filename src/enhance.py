import os
import tqdm
import torch

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

class SpeechEnhancer:
    
    def __init__(self, args):

        self.out_postfix: str = args['se_out_postfix']
        
        self.model = pipeline(Tasks.acoustic_noise_suppression, model='damo/speech_frcrn_ans_cirm_16k', verbose=False)
    
    def run_enhancement(self, in_wav_path, out_wav_dir=None, overwrite=False):
        if out_wav_dir is None:
            out_wav_dir = os.path.dirname(in_wav_path)

        out_wav_name = os.path.splitext(os.path.basename(in_wav_path))[0]+self.out_postfix+'.wav'
        out_wav_path = os.path.join(out_wav_dir, out_wav_name)
        if not overwrite and os.path.exists(out_wav_path):
            print(">>Already Exists SE WAV: {}".format(out_wav_path))
            return out_wav_path

        print(">>Run Speech Enhancement: {}".format(out_wav_path))
        _ = self.model(in_wav_path, output_path=out_wav_path)
        # raw_pcm = self.model(in_wav_path, output_path=out_wav_path)['output_pcm']

        return out_wav_path

    def __call__(self, test_wav_path, out_wav_dir=None, overwrite=False):

        if isinstance(test_wav_path, str):

            out_wav_path = self.run_enhancement(test_wav_path, out_wav_dir=out_wav_dir, overwrite=overwrite)
            out_wav_list = [out_wav_path]

        elif isinstance(test_wav_path, list):

            out_wav_list = []
            for in_wav_path in tqdm.tqdm(test_wav_path):
                out_wav_path = self.run_enhancement(in_wav_path, out_wav_dir=out_wav_dir, overwrite=overwrite)
                out_wav_list.append(out_wav_path)
        
        return out_wav_list

if __name__ == "__main__":
    """
    Get an argument parser.
    """
    import argparse
    from utils import set_seeds

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--seed", type=int, default=777, help="seed number")
    parser.add_argument('--test_wav_path', type=str, default='/mnt/FRCRN/speech_with_noise1.wav', required=False, help='path of test wav file')
    # parser.add_argument('--test_wav_path', type=str, default='data/youtube/_ezibzn6K9Y/wav/_ezibzn6K9Y.wav', required=False, help='path of test wav file')
    parser.add_argument('--se_out_postfix', type=str, default='_SE_FRCRN', required=False, help='output postfix string')

    args = parser.parse_args().__dict__

    set_seeds(args['seed'])

    test_wav_path: str = args.pop('test_wav_path')
    assert(os.path.exists(test_wav_path)), "No Exists File Name: {}".format(test_wav_path)

    se_manager = SpeechEnhancer(args)
    result, out_wav_list = se_manager(test_wav_path)
    
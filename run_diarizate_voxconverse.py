import argparse
import os
import tqdm
import glob
import json
import pickle
import pandas as pd
import torch
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
from whisper.utils import (
    optional_float,
    optional_int,
    str2bool,
)
from whisper.audio import SAMPLE_RATE

from src.utils import set_seeds

from src.enhance import SpeechEnhancer

from src.diarize import SpeakerDiarizer

from src.visualize import viewer

def get_args():
    from whisper import available_models

    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--exp_dir", type=str, default='exps', help="path to experiments directory")

    parser.add_argument('--num_threads', type=int, default=0, required = False, help='number of threads')
    parser.add_argument("--seed", type=int, default=777, help="seed number")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="device to use for PyTorch inference")
    parser.add_argument('--sr', type=int, default=SAMPLE_RATE, required = False, help='sampling rate')


    parser.add_argument("--verbose", type=str2bool, default=True, help="whether to print out the progress and debug messages")

    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default=None, choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]), help="language spoken in the audio, specify None to perform language detection")

    # youtube loader config
    parser.add_argument('--yt_dir', type=str, default='data/youtube', required=False, help='mp4 download directory')

    # ASR config
    parser.add_argument("--asr_model", default="small", choices=available_models(), help="name of the Whisper model to use")
    parser.add_argument("--asr_model_dir", type=str, default=None, help="path to save model files; uses ~/.cache/whisper by default")
    parser.add_argument("--asr_output_dir", "-o", type=str, default=".", help="directory to save the outputs")
    parser.add_argument("--asr_output_format", "-f", type=str, default="all", choices=["all", "srt", "srt-word", "vtt", "txt", "tsv", "ass", "ass-char", "pickle", "vad"], help="format of the output file; if not specified, all available formats will be produced")

    # speech enhancement config
    parser.add_argument('--se_out_postfix', type=str, default='_SE_FRCRN', required=False, help='output postfix string')
    parser.add_argument('--use_se', type=bool, default=False, required=False, help='True if you use speech enhancement mode')

    # clean speech detector config
    parser.add_argument("--csd_csv_dir", type=str, default='csd/csv', help="path to experiments directory")

    # speech quality assessment config
    parser.add_argument("--sqa_ssl_model_path", type=str, default='models/sqa_models/wav2vec_small.pt', help="pretrained wav2vec base model path")
    parser.add_argument("--sqa_model_ckpt_path", type=str, default='models/sqa_models/model_noresqa_mos.pth', help="pretrained NORESQA-MOS model path")
    parser.add_argument('--sqa_nmr_wav_dir', type=str, default='/mnt/dataset/daps', required = False, help='path of clean wav file')
    parser.add_argument('--sqa_nmr_feat_path', type=str, default='sqa/noresqa/feat/daps_nmr_embs.pkl', required = False, help='path of nmr embedding pickle file')
    parser.add_argument("--sqa_nmr_chunk_time", type=float, default=3.0, help="nmr wav chunk time")
    parser.add_argument("--sqa_nmr_step_size", type=int, default=75, help="embedding step size")

    # sound classification config
    parser.add_argument('--sc_ontology_file_path', type=str, default='data/BEATs/ontology.json', required=False, help='path of audioset ontology')
    parser.add_argument('--sc_labels_indices_csv', type=str, default='data/BEATs/class_labels_indices.csv', required=False, help='csv file of containing audioset label indices')
    parser.add_argument("--beats_model_ckpt_path", type=str, default='models/sc_models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt', help="pretrained BEATs model path")
    parser.add_argument("--sc_chunk_time", type=float, default=1.0, help="sc chunk time")
    parser.add_argument("--sc_step_ratio", type=float, default=0.1, help="sc step ratio")

    # alignment params
    parser.add_argument("--align_model", default=None, help="Name of phoneme-level ASR model to do alignment")
    parser.add_argument("--align_extend", default=2, type=float, help="Seconds before and after to extend the whisper segments for alignment (if not using VAD).")
    parser.add_argument("--align_from_prev", default=True, type=bool, help="Whether to clip the alignment start time of current segment to the end time of the last aligned word of the previous segment (if not using VAD)")
    parser.add_argument("--interpolate_method", default="nearest", choices=["nearest", "linear", "ignore"], help="For word .srt, method to assign timestamps to non-aligned words, or merge them into neighbouring.")
    parser.add_argument("--no_align", action='store_true', help="Do not perform phoneme alignment")

    # vad params
    parser.add_argument("--hf_token", type=str, default='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', help="Hugging Face Access Token to access PyAnnote gated models")
    parser.add_argument("--vad_tmp_dir", default="vad/tmp_wav", help="Temporary directory to write audio file if input if not .wav format (only for VAD).")
    parser.add_argument("--vad_save_lab_dir", default="vad/lab", help="Temporary directory to write audio file if input if not .wav format (only for VAD).")

    parser.add_argument("--vad_filter", default=True, help="Whether to pre-segment audio with VAD, highly recommended! Produces more accurate alignment + timestamp see WhisperX paper https://arxiv.org/abs/2303.00747")
    parser.add_argument("--vad_onset", type=float, default=0.500, help="Onset threshold for VAD (see pyannote.audio), reduce this if speech is not being detected")
    parser.add_argument("--vad_offset", type=float, default=0.363, help="Offset threshold for VAD (see pyannote.audio), reduce this if speech is not being detected.")
    parser.add_argument("--vad_pad_onset", type=float, default=0.250, help="Padding Onset for VAD (see pyannote.audio)")
    parser.add_argument("--vad_pad_offset", type=float, default=0.250, help="Padding time for VAD (see pyannote.audio)")

    # diarization params
    parser.add_argument("--no_diarize", action="store_false", help="Apply diarization to assign speaker labels to each segment/word")
    parser.add_argument("--min_speakers", default=None, type=int)
    parser.add_argument("--max_speakers", default=None, type=int)

    parser.add_argument("--diar_exp_dir", type=str, default='sd/voxconverse/test', help="path to diarization experiments directory")
    parser.add_argument('--diar_model_name', type=str, default='pyannote/speaker-diarization@2.1', required=False, help='pretrained speaker diarization model name')
    # parser.add_argument('--diar_embedding', type=str, default='fbdp1202/mfa-conformer', required=False, help='pretrained speaker diarization model name')
    parser.add_argument('--diar_embedding', type=str, default='speechbrain/spkrec-ecapa-voxceleb', required=False, help='pretrained speaker diarization model name')

    # whisper params
    parser.add_argument("--temperature", type=float, default=0, help="temperature to use for sampling")
    parser.add_argument("--best_of", type=optional_int, default=5, help="number of candidates when sampling with non-zero temperature")
    parser.add_argument("--beam_size", type=optional_int, default=5, help="number of beams in beam search, only applicable when temperature is zero")
    parser.add_argument("--patience", type=float, default=None, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    parser.add_argument("--length_penalty", type=float, default=None, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")

    parser.add_argument("--suppress_tokens", type=str, default="-1", help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
    parser.add_argument("--initial_prompt", type=str, default=None, help="optional text to provide as a prompt for the first window.")
    parser.add_argument("--condition_on_previous_text", type=str2bool, default=False, help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")
    parser.add_argument("--fp16", type=str2bool, default=True, help="whether to perform inference in fp16; True by default")

    parser.add_argument("--temperature_increment_on_fallback", type=optional_float, default=0.2, help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below")
    parser.add_argument("--compression_ratio_threshold", type=optional_float, default=2.4, help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
    parser.add_argument("--logprob_threshold", type=optional_float, default=-1.0, help="if the average log probability is lower than this value, treat the decoding as failed")
    parser.add_argument("--no_speech_threshold", type=optional_float, default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")
    parser.add_argument("--word_timestamps", type=str2bool, default=False, help="(experimental) extract word-level timestamps and refine the results based on them")
    parser.add_argument("--prepend_punctuations", type=str, default="\"\'“¿([{-", help="if word_timestamps is True, merge these punctuation symbols with the next word")
    parser.add_argument("--append_punctuations", type=str, default="\"\'.。,，!！?？:：”)]}、", help="if word_timestamps is True, merge these punctuation symbols with the previous word")
    parser.add_argument("--threads", type=optional_int, default=0, help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")

    # parser.add_argument("--model_flush", action="store_true", help="Flush memory from each model after use, reduces GPU requirement but slower processing >1 audio file.")

    # custom
    parser.add_argument("--overwrite", action='store_true', help="Extracting features independently of their existence")
    # fmt: on

    args = parser.parse_args().__dict__
    
    args['vad_tmp_dir'] = os.path.join(args['exp_dir'], args['vad_tmp_dir'])
    args['vad_save_lab_dir'] = os.path.join(args['exp_dir'], args['vad_save_lab_dir'])
    args['sqa_nmr_feat_path'] = os.path.join(args['exp_dir'], args['sqa_nmr_feat_path'])
    args['csd_csv_dir'] = os.path.join(args['exp_dir'], args['csd_csv_dir'])

    args['diar_exp_dir'] = os.path.join(args['exp_dir'], args['diar_exp_dir'], args['diar_embedding'].replace('/', '_'))

    return args

def main():

    args = get_args()
    set_seeds(args['seed'])

    overwrite: bool = args.pop("overwrite")
    
    wav_list = sorted(glob.glob('/mnt/labelmaker/labelmaker/data/sd/voxconverse_0_3/wav/test' + "/*.wav"))

    # run speech enhancement
    # use_se: bool = args['use_se']
    use_se: bool = False
    if use_se:
        enhancer = SpeechEnhancer(args)
        se_wav_list = enhancer(wav_list)
        assert(len(se_wav_list) == len(wav_list)),\
            "Not Match Speech Enhancement Wav File Number ({} != {})".format(len(se_wav_list), len(wav_list))
        del enhancer

    # run speaker diarization
    diarizer = SpeakerDiarizer(args)

    diar_annot_list = []
    for wav_path in tqdm.tqdm(wav_list):
        diar_results = diarizer(wav_path)
        diar_annot_list.append(diar_results)
    del diarizer


if __name__ == "__main__":
    main()
import ffmpeg
import os
import warnings
from typing import TYPE_CHECKING, Optional, Tuple, Union
import tempfile
import numpy as np
import pandas as pd
import torch
from whisper import load_model
from whisper.audio import SAMPLE_RATE

from whisperx.alignment import load_align_model, align, check_align_model
from whisperx.asr import transcribe, transcribe_with_vad
from whisperx.diarize import DiarizationPipeline, assign_word_speakers
from whisperx.utils import get_writer
from whisperx.vad import load_vad_model
from whisperx.diarize import Segment as SegmentX

from whisper.audio import (
    N_SAMPLES,
    SAMPLE_RATE,
    CHUNK_LENGTH,
    log_mel_spectrogram,
    load_audio
)

from whisper.utils import (
    exact_div,
    format_timestamp,
    make_safe,
)

def assign_segment_speakers(diarize_df, result_segments, fill_nearest=False):
    for seg in result_segments:
        speakers = []
        # for wdx, wrow in wdf.iterrows():
        if not np.isnan(seg['start']):
            diarize_df['intersection'] = np.minimum(diarize_df['end'], seg['end']) - np.maximum(diarize_df['start'], seg['start'])
            diarize_df['union'] = np.maximum(diarize_df['end'], seg['end']) - np.minimum(diarize_df['start'], seg['start'])
            # remove no hit
            if not fill_nearest:
                dia_tmp = diarize_df[diarize_df['intersection'] > 0]
            else:
                dia_tmp = diarize_df
            if len(dia_tmp) == 0:
                speaker = None
            else:
                speaker = dia_tmp.sort_values("intersection", ascending=False).iloc[0][2]
        else:
            speaker = None
        speakers.append(speaker)

        speaker_count = pd.Series(speakers).value_counts()
        if len(speaker_count) == 0:
            seg["speaker"]= "UNKNOWN"
        else:
            seg["speaker"] = speaker_count.index[0]

    return result_segments

class SpeechRecognizer:
    
    def __init__(self, args):
        asr_model_name: str = args["asr_model"]
        model_dir: str = args["asr_model_dir"]
        
        self.output_dir: str = args["asr_output_dir"]
        self.output_format: str = args["asr_output_format"]
        self.device: str = args["device"]
        # model_flush: bool = args.pop("model_flush")
        os.makedirs(self.output_dir, exist_ok=True)

        self.tmp_dir: str = args["vad_tmp_dir"]
        if self.tmp_dir is not None:
            os.makedirs(self.tmp_dir, exist_ok=True)

        # Align Config
        align_model: str = args["align_model"]
        self.align_extend: float = args["align_extend"]
        self.align_from_prev: bool = args["align_from_prev"]
        self.interpolate_method: str = args["interpolate_method"]
        no_align: bool = args["no_align"]

        # load align model and set language
        if no_align:
            self.align_model, self.align_metadata = None, None
        else:
            align_language = args["language"] if args["language"] is not None else "en" # default to loading english if not specified
            self.align_model, self.align_metadata = load_align_model(align_language, self.device, model_name=align_model)

        # if model_flush:
        #     print(">>Model flushing activated... Only loading model after ASR stage")
        #     del align_model
        #     align_model = ""

        if asr_model_name.endswith(".en") and args["language"] not in {"en", "English"}:
            if args["language"] is not None:
                warnings.warn(
                    f"{asr_model_name} is an English-only model but receipted '{args['language']}'; using English instead."
                )
            args["language"] = "en"

        # set temperature
        temperature = args.pop("temperature")
        if (increment := args.pop("temperature_increment_on_fallback")) is not None:
            self.temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
        else:
            self.temperature = [temperature]

        # set num threads
        if (threads := args.pop("threads")) > 0:
            torch.set_num_threads(threads)

        self.asr_model = load_model(asr_model_name, device=self.device, download_root=model_dir)

        self.args = args

    def asr_merge_chunks(self, segments, chunk_size):
        curr_end = 0
        merged_segments = []
        seg_idxs = []
        speaker_idxs = []

        assert chunk_size > 0

        segments_list = []
        for segment, _, label in segments.itertracks(yield_label=True):
            segments_list.append(SegmentX(segment.start, segment.end, label))

        if len(segments_list) == 0:
            print("No active speech found in audio")
            return []

        # assert segments_list, "segments_list is empty."
        # Make sur the starting point is the start of the segment.
        curr_start = segments_list[0].start

        for seg in segments_list:
            if seg.end - curr_start > chunk_size and curr_end-curr_start > 0:
                merged_segments.append({
                    "start": curr_start,
                    "end": curr_end,
                    "segments": seg_idxs,
                })
                curr_start = seg.start
                seg_idxs = []
                speaker_idxs = []
            curr_end = seg.end
            seg_idxs.append((seg.start, seg.end))
            speaker_idxs.append(seg.speaker)
        # add final
        merged_segments.append({ 
                    "start": curr_start,
                    "end": curr_end,
                    "segments": seg_idxs,
                })    
        return merged_segments
    
    def transcribe_with_vad_info(self, wav_path, diarize_segments, mel = None, verbose = True):
        audio = load_audio(wav_path)
        audio = torch.from_numpy(audio)

        prev = 0
        output = {"segments": []}

        # merge segments to approx 30s inputs to make whisper most appropraite
        diarize_segments = self.asr_merge_chunks(diarize_segments, chunk_size=CHUNK_LENGTH)
        # diarize_segments = self.asr_chunking(diarize_segments, chunk_size=CHUNK_LENGTH)
        if len(diarize_segments) == 0:
            return output

        print(">>Performing transcription...")
        for sdx, seg_t in enumerate(diarize_segments):
            if verbose:
                print(f"~~ Transcribing VAD chunk: ({format_timestamp(seg_t['start'])} --> {format_timestamp(seg_t['end'])}) ~~")
            seg_f_start, seg_f_end = int(seg_t["start"] * SAMPLE_RATE), int(seg_t["end"] * SAMPLE_RATE)
            local_f_start, local_f_end = seg_f_start - prev, seg_f_end - prev
            audio = audio[local_f_start:] # seek forward
            seg_audio = audio[:local_f_end-local_f_start] # seek forward
            prev = seg_f_start
            local_mel = log_mel_spectrogram(seg_audio, padding=N_SAMPLES)
            # need to pad

            decode_kwargs = {}
            for key in ["task", "language", "beam_size", "patience", "length_penalty", "suppress_tokens", "fp16", "prompt", "prefix"]:
                if key in self.args.keys():
                    decode_kwargs[key] = self.args.pop(key)

            result = transcribe(self.asr_model, audio, mel=local_mel, temperature=self.temperature, **decode_kwargs)
            seg_t["text"] = result["text"]
            output["segments"].append(
                {
                    "start": seg_t["start"],
                    "end": seg_t["end"],
                    "language": result["language"],
                    "text": result["text"],
                    "seg-text": [x["text"] for x in result["segments"]],
                    "seg-start": [x["start"] for x in result["segments"]],
                    "seg-end": [x["end"] for x in result["segments"]],
                    }
                )

        output["language"] = output["segments"][0]["language"]

        return output

    def __call__(self, wav_path, diarize_segments, mel = None):

        result = self.transcribe_with_vad_info(wav_path, diarize_segments, mel=mel, verbose=self.args['verbose'])

        # >> Align
        if self.align_model is not None and len(result["segments"]) > 0:
            if check_align_model(result["language"]):
                if result.get("language", "en") != self.align_metadata["language"]:
                    # load new language
                    print(f"New language found ({result['language']})! Previous was ({self.align_metadata['language']}), loading new alignment model for new language...")
                    self.align_model, self.align_metadata = load_align_model(result["language"], self.device)
                print(">>Performing alignment...")
                result = align(result["segments"], self.align_model, self.align_metadata, wav_path, self.device,
                    extend_duration=self.align_extend, start_from_previous=self.align_from_prev, interpolate_method=self.interpolate_method)

        # >> Diarize
        diarize_df = pd.DataFrame(diarize_segments.itertracks(yield_label=True))
        diarize_df['start'] = diarize_df[0].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df[0].apply(lambda x: x.end)

        if not 'word-segments' in result["segments"]:
            word_segments = None
            results_segments = assign_segment_speakers(diarize_df, result["segments"])
        else:
            results_segments, word_segments = assign_word_speakers(diarize_df, result["segments"])

        result = {"segments": results_segments, "word_segments": word_segments}

        return result

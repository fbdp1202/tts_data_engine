import ffmpeg
import os
import warnings
from typing import TYPE_CHECKING, Optional, Tuple, Union
import tempfile
import numpy as np
import torch
from whisper import load_model
from whisper.audio import SAMPLE_RATE

from whisperx.alignment import load_align_model, align
from whisperx.asr import transcribe, transcribe_with_vad
from whisperx.diarize import DiarizationPipeline, assign_word_speakers
from whisperx.utils import get_writer
from whisperx.vad import load_vad_model

class DataEngine:
    def __init__(self, args):
        asr_model_name: str = args.pop("model")
        model_dir: str = args.pop("model_dir")
        
        self.output_dir: str = args.pop("output_dir")
        self.output_format: str = args.pop("output_format")
        self.device: str = args.pop("device")
        # model_flush: bool = args.pop("model_flush")
        os.makedirs(self.output_dir, exist_ok=True)

        self.tmp_dir: str = args.pop("tmp_dir")
        if self.tmp_dir is not None:
            os.makedirs(self.tmp_dir, exist_ok=True)

        # Align Config
        align_model: str = args.pop("align_model")
        self.align_extend: float = args.pop("align_extend")
        self.align_from_prev: bool = args.pop("align_from_prev")
        self.interpolate_method: str = args.pop("interpolate_method")
        no_align: bool = args.pop("no_align")

        # VAD config
        hf_token: str = args.pop("hf_token")
        vad_filter: bool = args.pop("vad_filter")
        self.vad_onset: float = args.pop("vad_onset")
        self.vad_offset: float = args.pop("vad_offset")

        self.diarize: bool = args.pop("no_diarize")
        self.min_speakers: int = args.pop("min_speakers")
        self.max_speakers: int = args.pop("max_speakers")

        # load VAD model
        self.vad_model = None
        if vad_filter:
            from pyannote.audio import Pipeline
            from pyannote.audio import Model, Pipeline
            self.vad_model = load_vad_model(torch.device(self.device), self.vad_onset, self.vad_offset, use_auth_token=hf_token)

        # load diarization model
        self.diarize_model = None
        if self.diarize:
            if hf_token is None:
                print("Warning, no --hf_token used, needs to be saved in environment variable, otherwise will throw error loading diarization model...")
            self.diarize_model = DiarizationPipeline(use_auth_token=hf_token)

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


    def infer(self, audio_path):

        input_audio_path = audio_path
        tfile = None

        # >> VAD & ASR
        if self.vad_model is not None:
            if not audio_path.endswith(".wav"):
                print(">>VAD requires .wav format, converting to wav as a tempfile...")
                # tfile = tempfile.NamedTemporaryFile(delete=True, suffix=".wav")
                audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
                if self.tmp_dir is not None:
                    input_audio_path = os.path.join(self.tmp_dir, audio_basename + ".wav")
                else:
                    input_audio_path = os.path.join(os.path.dirname(audio_path), audio_basename + ".wav")
                ffmpeg.input(audio_path, threads=0).output(input_audio_path, ac=1, ar=SAMPLE_RATE).run(cmd=["ffmpeg"])
            print(">>Performing VAD...")
            result = transcribe_with_vad(self.asr_model, input_audio_path, self.vad_model, temperature=self.temperature, **self.args)
        else:
            print(">>Performing transcription...")
            result = transcribe(self.asr_model, input_audio_path, temperature=self.temperature, **self.args)

        # >> Align
        if self.align_model is not None and len(result["segments"]) > 0:
            if result.get("language", "en") != self.align_metadata["language"]:
                # load new language
                print(f"New language found ({result['language']})! Previous was ({self.align_metadata['language']}), loading new alignment model for new language...")
                self.align_model, self.align_metadata = load_align_model(result["language"], self.device)
            print(">>Performing alignment...")
            result = align(result["segments"], self.align_model, self.align_metadata, input_audio_path, self.device,
                extend_duration=self.align_extend, start_from_previous=self.align_from_prev, interpolate_method=self.interpolate_method)

        # >> Diarize
        if self.diarize_model is not None:
            diarize_segments = self.diarize_model(input_audio_path, min_speakers=self.min_speakers, max_speakers=self.max_speakers)
            results_segments, word_segments = assign_word_speakers(diarize_segments, result["segments"])
            result = {"segments": results_segments, "word_segments": word_segments}

        # cleanup
        if input_audio_path != audio_path:
            os.remove(input_audio_path)
        
        return result

import json

import os
import zlib
from typing import Callable, TextIO, Iterator, Tuple
import pandas as pd
import numpy as np

from whisper.utils import ResultWriter

import pdb

def write_ass(transcript: Iterator[dict],
            file: TextIO,
            color: str = None, underline=True,
            prefmt: str = None, suffmt: str = None,
            font: str = None, font_size: int = 24,
            strip=True, **kwargs):
    """
    Credit: https://github.com/jianfch/stable-ts/blob/ff79549bd01f764427879f07ecd626c46a9a430a/stable_whisper/text_output.py
        Generate Advanced SubStation Alpha (ass) file from results to
    display both phrase-level & word-level timestamp simultaneously by:
     -using segment-level timestamps display phrases as usual
     -using word-level timestamps change formats (e.g. color/underline) of the word in the displayed segment
    Note: ass file is used in the same way as srt, vtt, etc.
    Parameters
    ----------
    transcript: dict
        results from modified model
    file: TextIO
        file object to write to
    color: str
        color code for a word at its corresponding timestamp
        <bbggrr> reverse order hexadecimal RGB value (e.g. FF0000 is full intensity blue. Default: 00FF00)
    underline: bool
        whether to underline a word at its corresponding timestamp
    prefmt: str
        used to specify format for word-level timestamps (must be use with 'suffmt' and overrides 'color'&'underline')
        appears as such in the .ass file:
            Hi, {<prefmt>}how{<suffmt>} are you?
        reference [Appendix A: Style override codes] in http://www.tcax.org/docs/ass-specs.htm
    suffmt: str
        used to specify format for word-level timestamps (must be use with 'prefmt' and overrides 'color'&'underline')
        appears as such in the .ass file:
            Hi, {<prefmt>}how{<suffmt>} are you?
        reference [Appendix A: Style override codes] in http://www.tcax.org/docs/ass-specs.htm
    font: str
        word font (default: Arial)
    font_size: int
        word font size (default: 48)
    kwargs:
        used for format styles:
        'Name', 'Fontname', 'Fontsize', 'PrimaryColour', 'SecondaryColour', 'OutlineColour', 'BackColour', 'Bold',
        'Italic', 'Underline', 'StrikeOut', 'ScaleX', 'ScaleY', 'Spacing', 'Angle', 'BorderStyle', 'Outline',
        'Shadow', 'Alignment', 'MarginL', 'MarginR', 'MarginV', 'Encoding'

    """

    fmt_style_dict = {'Name': 'Default', 'Fontname': 'Arial', 'Fontsize': '48', 'PrimaryColour': '&Hffffff',
                    'SecondaryColour': '&Hffffff', 'OutlineColour': '&H0', 'BackColour': '&H0', 'Bold': '0',
                    'Italic': '0', 'Underline': '0', 'StrikeOut': '0', 'ScaleX': '100', 'ScaleY': '100',
                    'Spacing': '0', 'Angle': '0', 'BorderStyle': '1', 'Outline': '1', 'Shadow': '0',
                    'Alignment': '2', 'MarginL': '10', 'MarginR': '10', 'MarginV': '10', 'Encoding': '0'}

    for k, v in filter(lambda x: 'colour' in x[0].lower() and not str(x[1]).startswith('&H'), kwargs.items()):
        kwargs[k] = f'&H{kwargs[k]}'

    fmt_style_dict.update((k, v) for k, v in kwargs.items() if k in fmt_style_dict)

    if font:
        fmt_style_dict.update(Fontname=font)
    if font_size:
        fmt_style_dict.update(Fontsize=font_size)

    fmts = f'Format: {", ".join(map(str, fmt_style_dict.keys()))}'

    styles = f'Style: {",".join(map(str, fmt_style_dict.values()))}'

    ass_str = f'[Script Info]\nScriptType: v4.00+\nPlayResX: 384\nPlayResY: 288\nScaledBorderAndShadow: yes\n\n' \
            f'[V4+ Styles]\n{fmts}\n{styles}\n\n' \
            f'[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n\n'

    if prefmt or suffmt:
        if suffmt:
            assert prefmt, 'prefmt must be used along with suffmt'
        else:
            suffmt = r'\r'
    else:
        if not color:
            color = 'HFF00'
        underline_code = r'\u1' if underline else ''

        prefmt = r'{\1c&' + f'{color.upper()}&{underline_code}' + '}'
        suffmt = r'{\r}'
    
    def secs_to_hhmmss(secs: Tuple[float, int]):
        mm, ss = divmod(secs, 60)
        hh, mm = divmod(mm, 60)
        return f'{hh:0>1.0f}:{mm:0>2.0f}:{ss:0>2.2f}'


    def dialogue(chars: str, start: float, end: float, idx_0: int, idx_1: int, audio_tag: list = None, sqa_tag: dict = None) -> str:
        if idx_0 == -1:
            text = chars
        else:
            text = f'{chars[:idx_0]}{prefmt}{chars[idx_0:idx_1]}{suffmt}{chars[idx_1:]}'
        if audio_tag is not None:
            text = text + r'{\fs10}\N'
            tag_text_arr = []
            for tag in audio_tag:
                tag_text = f'{tag["name"]}: {tag["pred"]:.2f}'
                tag_text_arr.append(tag_text)
            text = text + ','.join(tag_text_arr)
        
        if sqa_tag is not None:
            text = text + r'{\fs10}\N'
            sqa_text_arr = []
            for key, value in sqa_tag.items():
                sqa_text = f'{key}: {value:.2f}'
                sqa_text_arr.append(sqa_text)
            text = text + ','.join(sqa_text_arr)

        return f"Dialogue: 0,{secs_to_hhmmss(start)},{secs_to_hhmmss(end)}," \
               f"Default,,0,0,0,,{text.strip() if strip else text}"
    
    ass_arr = []

    for segment in transcript:
        # if "12" in segment['text']:
            # import pdb; pdb.set_trace()

        if "spk_id" in segment:
            speaker_str = f"[{segment['spk_id']}]: "
        else:
            speaker_str = ""
            
        uttr_ts = {
            "chars": speaker_str + segment['text'],
            "start": segment['start'],
            "end": segment['end'],
            "idx_0": -1,
            "idx_1": -1,
            "audio_tag": segment['audio_tag'],
            "sqa_tag": segment['sqa_tag'],
        }

        ass_arr.append(uttr_ts)

    ass_str += '\n'.join(map(lambda x: dialogue(**x), ass_arr))

    file.write(ass_str)

class WriteASS(ResultWriter):
    extension: str = "ass"

    def write_result(self, result: dict, file: TextIO):
        write_ass(result["segments"], file)

if __name__ == "__main__":

    json_path = '/mnt/labelmaker/labelmaker/exps/results/M7h4bbv7XeE.json'
    ass_dir = "/mnt/labelmaker/labelmaker/exps/ass/M7h4bbv7XeE"
    wav_path = "/mnt/labelmaker/labelmaker/data/youtube/M7h4bbv7XeE/wav/M7h4bbv7XeE.wav"
    with open(json_path, 'r') as f:
        result = json.load(f)
    
    os.makedirs(ass_dir, exist_ok=True)

    writer = WriteASS(ass_dir)
    writer(result, wav_path)

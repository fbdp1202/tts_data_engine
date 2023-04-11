from pyannote.audio import Pipeline

import pdb


class SpeakerDiarizer:

    def __init__(self):

    


def main():
    use_auth_token="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"
    model_name="pyannote/speaker-diarization@2.1"
    model = Pipeline.from_pretrained(model_name, use_auth_token=use_auth_token)
    model = model.to('cuda:0')

    input_audio_path='/mnt/FRCRN/The_Dark_Knight.wav'
    # input_audio_path='/mnt/FRCRN/The_Dark_Knight_SE_FRCRN.wav'
    result = model(input_audio_path)
    print("Diarization Done.")

if __name__ == '__main__':
    main()
# TTS Data Engine
This project covers the TTS Data Engine for automatically generating TTS training data from YouTube url.

---

## Dependencies
```
requirements.txt

I'm not sure... It works "pip install requirements.txt" command

Please Just you refer if you set up your own environments.
```

## Set up

- `models/embedding/nnet/model.pth` &rarr;
    > "My Custom Embedding model, MFA_Conformer checkpoint path"

    > `https://github.com/zyzisyz/mfa_conformer` this github repository could be helpful for reproducing MFA-Conformer

- `models/sc_models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt` &rarr;
    > visit "https://github.com/microsoft/unilm/tree/master/beats"

- `models/sqa_models/model_noresqa_mos.pth` &rarr;
    > download from "https://github.com/facebookresearch/Noresqa/blob/main/models/model_noresqa_mos.pth"

    > git clone not work. Because model size is so big. you download directly.

- `models/sqa_models/wav2vec_small.pt` &rarr;
    > visit "https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md", download "Wav2Vec 2.0 Base No finetuning"

```

---

## Pipelines

- `main.py`
    - run pipeline and then save results
    - If you want to know the format of the output file, please refer to this file `Wb6Oc1_SdJw-ex.json` which was created from url `https://www.youtube.com/watch?v=Wb6Oc1_SdJw`

- `src/url_loader.py`
    - download mp4, wav, captions from youtube url
    - just use `pytube` package

- `src/diarize.py`
    - I just use `pyannote v2.1 framework`
    - I created `src/custom_pyannote` to just conduct experiments individually, but nothing has changed significantly.

- `src/asr.py`
    - I just use `whisper / whisperX framework`
    - I don't use the VAD module in WhisperX, but simply replaced them with diarization results.
    - I proceed with the word alignment, but I didn't use it in the result.

- `src/collector.py`
    - Sound Classification and Speech Quality Assessment analysis of the found utterances are conducted.
    - If the position of the utterance is not given, it can be used immediately using the VAD module of WhisperX.

- `src/classify.py`
    - Estimation of Sound Classification Results to Determine Acoustic Noise in a Given utterance.
    - For Sound Classification, we use BEATs iter3+ which is finetuned by AS-2M dataset.

- `src/sqa.py`
    - Perform a speech quality assessment subjective for a given utterance.
    - NORESQA-MOS model is used for the Speech Quality Assessment.
    - For Non Matching Reference (NMR), 100 DAPs dataset Clean were randomly cut for 3 seconds.

- `src/vad.py`
    - voice activity detection module.
    - Using the VAD module provided by whisperX.

- `src/enhance.py`
    - Speech Enhancement Module using FRCRN model which is archived SOTA performance in DNS Challenges.
    - we modify `https://github.com/modelscope/modelscope/blob/203a565a3996aa79c80eca34c8409b4334867268/modelscope/pipelines/audio/ans_pipeline.py` to `src/FRCRN/ans_pipeline.py` to boost the speed through decoding in batch mode.

---

## Citation

as well the following works, used in each stage of the pipeline:
```bibtex
@article{bain2022whisperx,
  title={WhisperX: Time-Accurate Speech Transcription of Long-Form Audio},
  author={Bain, Max and Huh, Jaesung and Han, Tengda and Zisserman, Andrew},
  journal={arXiv preprint, arXiv:2303.00747},
  year={2023}
}
```

```bibtex
@article{radford2022robust,
  title={Robust speech recognition via large-scale weak supervision},
  author={Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2212.04356},
  year={2022}
}
```

```bibtex
@article{baevski2020wav2vec,
  title={wav2vec 2.0: A framework for self-supervised learning of speech representations},
  author={Baevski, Alexei and Zhou, Yuhao and Mohamed, Abdelrahman and Auli, Michael},
  journal={Advances in neural information processing systems},
  volume={33},
  pages={12449--12460},
  year={2020}
}
```

```bibtex
@inproceedings{bredin2020pyannote,
  title={Pyannote. audio: neural building blocks for speaker diarization},
  author={Bredin, Herv{\'e} and Yin, Ruiqing and Coria, Juan Manuel and Gelly, Gregory and Korshunov, Pavel and Lavechin, Marvin and Fustes, Diego and Titeux, Hadrien and Bouaziz, Wassim and Gill, Marie-Philippe},
  booktitle={ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7124--7128},
  year={2020},
  organization={IEEE}
}
```

```bibtex
@inproceedings{
noresqa,
title={{NORESQA}: A Framework for Speech Quality Assessment using Non-Matching References},
author={Pranay Manocha and Buye Xu and Anurag Kumar},
booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
year={2021},
url={https://proceedings.neurips.cc/paper/2021/file/bc6d753857fe3dd4275dff707dedf329-Paper.pdf}
}

@inproceedings{
noresqamos,
title={Speech Quality Assessment through MOS using Non-Matching References},
author={Pranay Manocha and Anurag Kumar},
booktitle={Interspeech},
year={2022},
url={https://arxiv.org/abs/2206.12285}
}
```

```bibtex
@misc{kumar2023torchaudiosquim,
      title={TorchAudio-Squim: Reference-less Speech Quality and Intelligibility measures in TorchAudio}, 
      author={Anurag Kumar and Ke Tan and Zhaoheng Ni and Pranay Manocha and Xiaohui Zhang and Ethan Henderson and Buye Xu},
      year={2023},
      eprint={2304.01448},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```

```bibtex
@misc{chen2022beats,
      title={BEATs: Audio Pre-Training with Acoustic Tokenizers}, 
      author={Sanyuan Chen and Yu Wu and Chengyi Wang and Shujie Liu and Daniel Tompkins and Zhuo Chen and Furu Wei},
      year={2022},
      eprint={2212.09058},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```

```bibtex
@misc{zhao2022frcrn,
      title={FRCRN: Boosting Feature Representation using Frequency Recurrence for Monaural Speech Enhancement}, 
      author={Shengkui Zhao and Bin Ma and Karn N. Watcharasupat and Woon-Seng Gan},
      year={2022},
      eprint={2206.07293},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
# Whisper Finetuning with Pytorch Lightning

This code implements finetuning of OpenAI Whisper models using Pytorch Lightning.
Most of the code is inspired by (and partly directly copied from)
[whisper-finetuning](https://github.com/jumon/whisper-finetuning).
However, since the current code is based on Pytorch Lightning, it also support multi-GPU training.
It also supports training with SpecAugment.

The finetuning method implemented here (and also the one in [whisper-finetuning](https://github.com/jumon/whisper-finetuning)
is quite different from the finetuning code in HuggingFace Transformers and ESPNet, and
is more similar to the training method that was actually used for training Whisper (according to the paper): finetuning
is done on 30-second chunks extracted from long audio recordings, with
intra-utterance timestamps. Therefore, the resulting model works very well for 
transcribing long audios, using e.g. [faster-whisper](https://github.com/guillaumekln/faster-whisper/tree/master/faster_whisper).



## Usage

TODO
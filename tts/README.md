
<h2 align="center">
<p>Real-Time State-of-the-art Speech Synthesis for Tensorflow 2
</h2>

## Features
- High performance on Speech Synthesis.
- Be able to fine-tune on other languages.
- Fast, Scalable, and Reliable.
- Suitable for deployment.
- Easy to implement a new model, based-on abstract class.
- Mixed precision to speed-up training if possible.
- Support Single/Multi GPU gradient Accumulate.
- Support both Single/Multi GPU in base trainer class.
- TFlite conversion for all supported models.
- Android/IOS example.
- Support many languages (currently, we support Chinese, Korean, English, French and German)

## Requirements
This repository is tested on Ubuntu 18.04 with:

- Python 3.7+
- Cuda 10.1
- CuDNN 7.6.5
- Tensorflow 2.2/2.3/2.4/2.5/2.6
- [Tensorflow Addons](https://github.com/tensorflow/addons) >= 0.10.0

Different Tensorflow version should be working but not tested yet. This repo will try to work with the latest stable TensorFlow version. **We recommend you install TensorFlow 2.6.0 to training in case you want to use MultiGPU.**

# Supported Model architectures
TensorFlowTTS currently  provides the following architectures:

1. **Tacotron-2** released with the paper [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884) by Jonathan Shen, Ruoming Pang, Ron J. Weiss, Mike Schuster, Navdeep Jaitly, Zongheng Yang, Zhifeng Chen, Yu Zhang, Yuxuan Wang, RJ Skerry-Ryan, Rif A. Saurous, Yannis Agiomyrgiannakis, Yonghui Wu.
2. **Multi-band MelGAN** released with the paper [Multi-band MelGAN: Faster Waveform Generation for High-Quality Text-to-Speech](https://arxiv.org/abs/2005.05106) by Geng Yang, Shan Yang, Kai Liu, Peng Fang, Wei Chen, Lei Xie.
3. **FastSpeech2** released with the paper [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558) by Yi Ren, Chenxu Hu, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, Tie-Yan Liu.

We are also implementing some techniques to improve quality and convergence speed from the following papers:

2. **Guided Attention Loss** released with the paper [Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention
](https://arxiv.org/abs/1710.08969) by Hideyuki Tachibana, Katsuya Uenoyama, Shunsuke Aihara.

# Tutorial End-to-End

## Prepare Dataset

Prepare a dataset in the following format:
```
|- [NAME_DATASET]/
|   |- metadata.csv
|   |- wavs/
|       |- file1.wav
|       |- ...
```

Where `metadata.csv` has the following format: `id|transcription`. This is a ljspeech-like format; you can ignore preprocessing steps if you have other format datasets.

Note that `NAME_DATASET` should be `[ljspeech/kss/baker/libritts/synpaflex]` for example.

## Preprocessing

The preprocessing has two steps:

1. Preprocess audio features
    - Convert characters to IDs
    - Compute mel spectrograms
    - Normalize mel spectrograms to [-1, 1] range
    - Split the dataset into train and validation
    - Compute the mean and standard deviation of multiple features from the **training** split
2. Standardize mel spectrogram based on computed statistics

To reproduce the steps above:
```
tts/bin/preprocess --rootdir ./[ljspeech/kss/baker/libritts/thorsten/synpaflex] --outdir ./dump_[ljspeech/kss/baker/libritts/thorsten/synpaflex] --config preprocess/[ljspeech/kss/baker/thorsten/synpaflex]_preprocess.yaml --dataset [ljspeech/kss/baker/libritts/thorsten/synpaflex]
```

Right now we only support [`ljspeech`](https://keithito.com/LJ-Speech-Dataset/), [`kss`](https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset), [`baker`](https://weixinxcxdb.oss-cn-beijing.aliyuncs.com/gwYinPinKu/BZNSYP.rar), [`libritts`](http://www.openslr.org/60/), [`thorsten`](https://github.com/thorstenMueller/deep-learning-german-tts) and
[`synpaflex`](https://www.ortolang.fr/market/corpora/synpaflex-corpus/) for dataset argument. In the future, we intend to support more datasets.

**Note**: To run `libritts` preprocessing, please first read the instruction in bin/fastspeech2_libritts. We need to reformat it first before run preprocessing.

After preprocessing, the structure of the project folder should be:
```
|- [NAME_DATASET]/
|   |- metadata.csv
|   |- wav/
|       |- file1.wav
|       |- ...
|- dump_[ljspeech/kss/baker/libritts/thorsten]/
|   |- train/
|       |- ids/
|           |- LJ001-0001-ids.npy
|           |- ...
|       |- raw-feats/
|           |- LJ001-0001-raw-feats.npy
|           |- ...
|       |- raw-f0/
|           |- LJ001-0001-raw-f0.npy
|           |- ...
|       |- raw-energies/
|           |- LJ001-0001-raw-energy.npy
|           |- ...
|       |- norm-feats/
|           |- LJ001-0001-norm-feats.npy
|           |- ...
|       |- wavs/
|           |- LJ001-0001-wave.npy
|           |- ...
|   |- valid/
|       |- ids/
|           |- LJ001-0009-ids.npy
|           |- ...
|       |- raw-feats/
|           |- LJ001-0009-raw-feats.npy
|           |- ...
|       |- raw-f0/
|           |- LJ001-0001-raw-f0.npy
|           |- ...
|       |- raw-energies/
|           |- LJ001-0001-raw-energy.npy
|           |- ...
|       |- norm-feats/
|           |- LJ001-0009-norm-feats.npy
|           |- ...
|       |- wavs/
|           |- LJ001-0009-wave.npy
|           |- ...
|   |- stats.npy
|   |- stats_f0.npy
|   |- stats_energy.npy
|   |- train_utt_ids.npy
|   |- valid_utt_ids.npy
|- examples/
|   |- melgan/
|   |- fastspeech/
|   |- tacotron2/
|   ...
```

- `stats.npy` contains the mean and std from the training split mel spectrograms
- `stats_energy.npy` contains the mean and std of energy values from the training split
- `stats_f0.npy` contains the mean and std of F0 values in the training split
- `train_utt_ids.npy` / `valid_utt_ids.npy` contains training and validation utterances IDs respectively

We use suffix (`ids`, `raw-feats`, `raw-energy`, `raw-f0`, `norm-feats`, and `wave`) for each input type.


**IMPORTANT NOTES**:
- This preprocessing step is based on [ESPnet](https://github.com/espnet/espnet) so you can combine all models here with other models from ESPnet repository.
- Regardless of how your dataset is formatted, the final structure of the `dump` folder **SHOULD** follow the above structure to be able to use the training script, or you can modify it by yourself 😄.

## Training models

To know how to train model from scratch or fine-tune with other datasets/languages, please see detail at example directory.

- For Tacotron-2 tutorial, pls see bin/tacotron2
- For FastSpeech2 tutorial, pls see bin/fastspeech2
- For FastSpeech2 + MFA tutorial, pls see bin/fastspeech2_libritts
- For Multiband-MelGAN tutorial, pls see multiband_melgan
# Abstract Class Explaination

## Abstract DataLoader Tensorflow-based dataset

A detail implementation of abstract dataset class from dataset/abstract_dataset. There are some functions you need overide and understand:

1. **get_args**: This function return argumentation for **generator** class, normally is utt_ids.
2. **generator**: This function have an inputs from **get_args** function and return a inputs for models. **Note that we return a dictionary for all generator functions with the keys that exactly match with the model's parameters because base_trainer will use model(\*\*batch) to do forward step.**
3. **get_output_dtypes**: This function need return dtypes for each element from **generator** function.
4. **get_len_dataset**: Return len of datasets, normaly is len(utt_ids).

**IMPORTANT NOTES**:

- A pipeline of creating dataset should be: cache -> shuffle -> map_fn -> get_batch -> prefetch.
- If you do shuffle before cache, the dataset won't shuffle when it re-iterate over datasets.
- You should apply map_fn to make each element return from **generator** function have the same length before getting batch and feed it into a model.

Some examples to use this **abstract_dataset** are tacotron_dataset.py, fastspeech_dataset.py, melgan_dataset.py, fastspeech2_dataset.py


## Abstract Trainer Class

A detail implementation of base_trainer from trainer/base_trainer.py. It include Seq2SeqBasedTrainer and GanBasedTrainer inherit from BasedTrainer. All trainer support both single/multi GPU. There a some functions you **MUST** overide when implement new_trainer:

- **compile**: This function aim to define a models, and losses.
- **generate_and_save_intermediate_result**: This function will save intermediate result such as: plot alignment, save audio generated, plot mel-spectrogram ...
- **compute_per_example_losses**: This function will compute per_example_loss for model, note that all element of the loss **MUST** has shape [batch_size].

All models on this repo are trained based-on **GanBasedTrainer** (see train_melgan.py, train_melgan_stft.py, train_multiband_melgan.py) and **Seq2SeqBasedTrainer** (see train_tacotron2.py, train_fastspeech.py).

# End-to-End Examples
Here is an example code for end2end inference with fastspeech2 and multi-band melgan. 

```python
import numpy as np
import soundfile as sf
import yaml

import tensorflow as tf

from tts.inference import TFAutoModel
from tts.inference import AutoProcessor

# initialize fastspeech2 model.
fastspeech2 = TFAutoModel.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en")


# initialize mb_melgan model
mb_melgan = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-ljspeech-en")


# inference
processor = AutoProcessor.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en")

input_ids = processor.text_to_sequence("Recent research at Harvard has shown meditating for as little as 8 weeks, can actually increase the grey matter in the parts of the brain responsible for emotional regulation, and learning.")
# fastspeech inference

mel_before, mel_after, duration_outputs, _, _ = fastspeech2.inference(
    input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
    speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
    speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
    f0_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
    energy_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
)

# melgan inference
audio_before = mb_melgan.inference(mel_before)[0, :, 0]
audio_after = mb_melgan.inference(mel_after)[0, :, 0]

# save to file
sf.write('./audio_before.wav', audio_before, 22050, "PCM_16")
sf.write('./audio_after.wav', audio_after, 22050, "PCM_16")
```

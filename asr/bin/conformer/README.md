# Conformer: Convolution-augmented Transformer for Speech Recognition

Reference: [https://arxiv.org/abs/2005.08100](https://arxiv.org/abs/2005.08100)

## Example Model YAML Config

Go to [config.yml](./config.yml)

## Usage

Training, see `python bin/conformer/train.py --help`

Testing, see `python bin/conformer/test.py --help`

TFLite Conversion, see `python bin/conformer/inference/gen_tflite_model.py --help`

## Conformer Subwords - Results on LibriSpeech

**Summary**

- Number of subwords: 1031
- Maxium length of a subword: 4
- Subwords corpus: all training sets, dev sets and test-clean
- Number of parameters: 10,341,639
- Positional Encoding Type: sinusoid concatenation
- Trained on: 4 RTX 2080Ti 11G

**Error Rates**

| **Test-clean** | Test batch size |  WER (%)   |  CER (%)   |
| :------------: | :-------------: | :--------: | :--------: |
|    _Greedy_    |        1        | 6.37933683 | 2.4757576  |
|  _Greedy V2_   |        1        | 7.86670732 | 2.82563138 |

| **Test-other** | Test batch size |  WER (%)   |  CER (%)   |
| :------------: | :-------------: | :--------: | :--------: |
|    _Greedy_    |        1        | 15.7308521 | 7.67273521 |

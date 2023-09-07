import os
import tempfile
import wave
import pyaudio
import soundfile
import tensorflow as tf

from tensorflow_tts.inference import AutoConfig, TFAutoModel, AutoProcessor
from tensorflow_tts.processor import LJSpeechProcessor


def play_wav(fn):
    CHUNK = 1024
    wf = wave.open(fn, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(CHUNK)

    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(CHUNK)

    stream.stop_stream()
    stream.close()
    p.terminate()


def save_wav(audio, fn, rate):
    soundfile.write(fn, audio, rate, "PCM_16")


class TTS:

    def language(self):
        raise NotImplementedError

    def txt2wav(self, txt):
        raise NotImplementedError


class Tacotron2MBMelGAN(TTS):

    def __init__(self, txt2mel_conf_fn, txt2mel_model_fn,
                 mel2wav_conf_fn, mel2wav_model_fn,
                 mapper_fn,
                 language):
        print("Loading txt2mel config from", txt2mel_conf_fn)
        txt2mel_conf = AutoConfig.from_pretrained(txt2mel_conf_fn)

        print("Loading txt2mel model from", txt2mel_model_fn)
        self.txt2mel = TFAutoModel.from_pretrained(txt2mel_model_fn, txt2mel_conf)

        print("Loading mel2wav config from", mel2wav_conf_fn)
        mel2wav_conf = AutoConfig.from_pretrained(mel2wav_conf_fn)

        print("Loading mel2wav model from", mel2wav_model_fn)
        self.mel2wav = TFAutoModel.from_pretrained(mel2wav_model_fn, mel2wav_conf)

        print("Loading mapper from", mapper_fn)
        self.processor = AutoProcessor.from_pretrained(mapper_fn)

        self.lang = language

    def language(self):
        return self.lang

    def txt2wav(self, txt):
        if isinstance(self.processor, LJSpeechProcessor):
            txt = self.processor.get_phoneme(txt)

        input_ids = self.processor.text_to_sequence(txt, inference=True)

        print(input_ids)

        print("Text to Mel...")
        mel_outputs, post_mel_outputs, stop_outputs, alignment_historys, = self.txt2mel.inference(
            input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            input_lengths=tf.constant([len(input_ids)], dtype=tf.int32),
            speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
        )

        print("Mel to Wav...")
        post_mel_outputs = post_mel_outputs.numpy()
        post_mel_output = post_mel_outputs[0]
        print(post_mel_output.shape)
        stop_token = tf.math.round(tf.nn.sigmoid(stop_outputs[0]))  # [T]
        real_length = tf.math.reduce_sum(tf.cast(tf.math.equal(stop_token, 0.0), tf.int32), -1)
        post_mel_output = post_mel_output[:real_length, :]
        print(post_mel_output.shape)

        # for i, post_mel_output in enumerate(post_mel_outputs):
        #     print(post_mel_output.shape)
        #     stop_token = tf.math.round(tf.nn.sigmoid(stop_outputs[i]))  # [T]
        #     real_length = tf.math.reduce_sum(tf.cast(tf.math.equal(stop_token, 0.0), tf.int32), -1)
        #     post_mel_output = post_mel_output[:real_length, :]
        #     print(post_mel_output.shape)

        # post_mel_outputs = tf.expand_dims(post_mel_output, 0)

        audio_after = self.mel2wav.inference(post_mel_outputs)[0, :, 0]

        return audio_after


def test_baker():
    txt2mel_conf_fn = r".\bin\tacotron2\conf\tacotron2.baker.v1.yaml"
    txt2mel_model_fn = r"D:\dataset\baker\baker\tacotron2\checkpoints\model-28000.h5"
    mel2wav_conf_fn = r".\bin\mbmelgan\conf\multiband_melgan.baker.v1.yaml"
    mel2wav_model_fn = r"D:\dataset\baker\baker\mbmelgan\checkpoints\generator-80000.h5"
    mapper_fn = r".\processor\pretrained\baker_mapper.json"

    tts = Tacotron2MBMelGAN(txt2mel_conf_fn, txt2mel_model_fn,
                            mel2wav_conf_fn, mel2wav_model_fn,
                            mapper_fn, "zh")

    while True:
        text = input("输入要合成语音的文本: ")

        audio = tts.txt2wav(text)

        print("Saving wav...")
        tmp_wav_fn = os.path.join(tempfile.gettempdir(), str(hash(text)) + ".wav")
        save_wav(audio, tmp_wav_fn, 22050)

        print("Playing wav...")
        play_wav(tmp_wav_fn)


def test_ljspeech():
    txt2mel_conf_fn = r".\bin\tacotron2\conf\tacotron2.v1.yaml"
    txt2mel_model_fn = r"D:\dataset\LJSpeech-1.1\tacotron2\run1\checkpoints\model-20000.h5"
    mel2wav_conf_fn = r".\bin\mbmelgan\conf\multiband_melgan.v1.yaml"
    mel2wav_model_fn = r"D:\dataset\LJSpeech-1.1\mbmelgan\run2\checkpoints\generator-152000.h5"
    mapper_fn = r".\processor\pretrained\ljspeech_mapper.json"

    tts = Tacotron2MBMelGAN(txt2mel_conf_fn, txt2mel_model_fn,
                            mel2wav_conf_fn, mel2wav_model_fn,
                            mapper_fn, "en")

    while True:
        text = input("输入要合成语音的文本: ")

        audio = tts.txt2wav(text)

        print("Saving wav...")
        tmp_wav_fn = os.path.join(tempfile.gettempdir(), str(hash(text)) + ".wav")
        save_wav(audio, tmp_wav_fn, 22050)

        print("Playing wav...")
        play_wav(tmp_wav_fn)


if __name__ == "__main__":
    test_ljspeech()

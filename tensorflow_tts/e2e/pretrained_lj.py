import soundfile as sf

import tensorflow as tf

from tensorflow_tts.inference import AutoProcessor
from tensorflow_tts.inference import TFAutoModel

processor = AutoProcessor.from_pretrained("tensorspeech/tts-tacotron2-ljspeech-en")
tacotron2 = TFAutoModel.from_pretrained("tensorspeech/tts-tacotron2-ljspeech-en")
mb_melgan = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-ljspeech-en")

text = "This is a demo to show how to use our model to generate mel spectrogram from raw text."

input_ids = processor.text_to_sequence(text)

# tacotron2 inference (text-to-mel)
mel_outputs, post_mel_outputs, stop_outputs, alignment_historys = tacotron2.inference(
    input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
    input_lengths=tf.convert_to_tensor([len(input_ids)], tf.int32),
    speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
)

post_mel_outputs = post_mel_outputs.numpy()
for i, post_mel_output in enumerate(post_mel_outputs):
    stop_token = tf.math.round(tf.nn.sigmoid(stop_outputs[i]))  # [T]
    real_length = tf.math.reduce_sum(tf.cast(tf.math.equal(stop_token, 0.0), tf.int32), -1)
    post_mel_output = post_mel_output[:real_length, :]

audio = mb_melgan.inference(post_mel_outputs)[0, :, 0]

# save to file
sf.write('./audio_lj.wav', audio, 22050, "PCM_16")

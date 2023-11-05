import soundfile as sf

import tensorflow as tf

from tts.inference.auto_config import AutoConfig
from tts.inference.auto_model import TFAutoModel
from tts.inference.auto_processor import AutoProcessor

txt2mel_conf_fn = r"D:\kidden\mt\open\github\mt-io\tts\bin\tacotron2\conf\tacotron2.baker.v1.yaml"
print("Loading txt2mel config from", txt2mel_conf_fn)
txt2mel_conf = AutoConfig.from_pretrained(txt2mel_conf_fn)

model_fn = r"D:\kidden\mt\tts\tacotron2\baker\model-24000.h5"
print("Loading txt2mel model from", model_fn)
txt2mel = TFAutoModel.from_pretrained(model_fn, txt2mel_conf)

mel2wav_conf_fn = r"D:\kidden\mt\open\github\mt-io\tts\bin\mbmelgan\conf\multiband_melgan.baker.v1.yaml"
print("Loading mel2wav config from", mel2wav_conf_fn)
mel2wav_conf = AutoConfig.from_pretrained(mel2wav_conf_fn)

model_fn2 = r"D:\dataset\baker\baker\mbmelgan\checkpoints\generator-80000.h5"
print("Loading mel2wav model from", model_fn2)
mel2wav = TFAutoModel.from_pretrained(model_fn2, mel2wav_conf)

mapper_fn = r"D:\kidden\mt\open\github\mt-io\tts\processor\pretrained\baker_mapper.json"
print("Loading mapper from", mapper_fn)
processor = AutoProcessor.from_pretrained(mapper_fn)

text = "这是一个声音合成的测试例子，效果如何？"
input_ids = processor.text_to_sequence(text, inference=True)

print("Text to Mel...")
mel_outputs, post_mel_outputs, stop_outputs, alignment_historys, = txt2mel.inference(
    input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
    input_lengths=tf.constant([len(input_ids)], dtype=tf.int32),
    speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
)

print(post_mel_outputs.shape)

print("Mel to Wav...")
post_mel_outputs = post_mel_outputs.numpy()
for i, post_mel_output in enumerate(post_mel_outputs):
    print(stop_outputs[i][:200])
    print(stop_outputs[i][-200:])
    stop_token = tf.math.round(tf.nn.sigmoid(stop_outputs[i]))  # [T]
    real_length = tf.math.reduce_sum(tf.cast(tf.math.equal(stop_token, 0.0), tf.int32), -1)
    print(real_length)
    post_mel_output = post_mel_output[:real_length, :]

print(post_mel_outputs.shape)

audio_after = mel2wav.inference(post_mel_outputs)[0, :, 0]

wav_fn = './audio_baker_24k_80k.wav'
print("Saving wav file into", wav_fn)
sf.write(wav_fn, audio_after, 24000, "PCM_16")

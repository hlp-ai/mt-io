import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tts.inference import TFAutoModel, AutoConfig

mel2wav_conf_fn = r"D:\kidden\mt\open\github\mt-io\tts\bin\mbmelgan\conf\multiband_melgan.baker.v1.yaml"
print("Loading mel2wav config from", mel2wav_conf_fn)
mel2wav_conf = AutoConfig.from_pretrained(mel2wav_conf_fn)

model_fn2 = r"D:\dataset\baker\baker\mbmelgan\checkpoints\generator-80000.h5"
print("Loading mel2wav model from", model_fn2)
mb_melgan = TFAutoModel.from_pretrained(model_fn2, mel2wav_conf)

# Save to Pb
tf.saved_model.save(mb_melgan, "./mb_melgan", signatures=mb_melgan.inference)

# Load and Inference
mb_melgan = tf.saved_model.load("./mb_melgan")

mels = np.load(r"D:\dataset\baker\baker\dump\valid\norm-feats\4-norm-feats.npy")

audios = mb_melgan.inference(mels[None, ...])

plt.plot(audios[0, :, 0])  # (B, T, C)
plt.show()

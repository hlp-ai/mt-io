import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow_tts.inference import TFAutoModel


mb_melgan = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-ljspeech-en")

# # Save to Pb
tf.saved_model.save(mb_melgan, "./mb_melgan", signatures=mb_melgan.inference)

# # Load and Inference
mb_melgan = tf.saved_model.load("./mb_melgan")

mels = np.load("../dump/valid/norm-feats/LJ001-0009-norm-feats.npy")

audios = mb_melgan.inference(mels[None, ...])

plt.plot(audios[0, :, 0])

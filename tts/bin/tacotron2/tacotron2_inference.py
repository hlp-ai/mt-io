import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tts.inference.auto_config import AutoConfig
from tts.inference.auto_model import TFAutoModel
from tts.inference.auto_processor import AutoProcessor

mapper_fn = r"D:\kidden\mt\open\github\mt-io\tts\processor\pretrained\baker_mapper.json"
print("Loading mapper from", mapper_fn)
processor = AutoProcessor.from_pretrained(mapper_fn)

txt2mel_conf_fn = r"D:\kidden\mt\open\github\mt-io\tts\bin\tacotron2\conf\tacotron2.baker.v1.yaml"
print("Loading txt2mel config from", txt2mel_conf_fn)
txt2mel_conf = AutoConfig.from_pretrained(txt2mel_conf_fn)

model_fn = r"D:\kidden\mt\tts\tacotron2\baker\model-24000.h5"
print("Loading txt2mel model from", model_fn)
tacotron2 = TFAutoModel.from_pretrained(model_fn, txt2mel_conf)

tacotron2.setup_window(win_front=6, win_back=6)
tacotron2.setup_maximum_iterations(3000)

# # Save to Pb
# save model into pb and do inference. Note that signatures should be a tf.function with input_signatures.
tf.saved_model.save(tacotron2, "./test_saved", signatures=tacotron2.inference)

# # Load and Inference
tacotron2 = tf.saved_model.load("./test_saved")

input_text = "深度学习是目前自然语言处理的主流方法。"
input_ids = processor.text_to_sequence(input_text, inference=True)

decoder_output, mel_outputs, stop_token_prediction, alignment_history = tacotron2.inference(
    tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
    tf.convert_to_tensor([len(input_ids)], tf.int32),
    tf.convert_to_tensor([0], dtype=tf.int32)
)


def display_alignment(alignment_history):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_title(f'Alignment steps')
    im = ax.imshow(
        alignment_history[0].numpy(),
        aspect='auto',
        origin='lower',
        interpolation='none')
    fig.colorbar(im, ax=ax)
    plt.xlabel('Decoder timestep')
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    plt.show()
    plt.close()


display_alignment(alignment_history)


def display_mel(mel_outputs):
    mel_outputs = tf.reshape(mel_outputs, [-1, 80]).numpy()
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(311)
    ax1.set_title(f'Predicted Mel-after-Spectrogram')
    im = ax1.imshow(np.rot90(mel_outputs), aspect='auto', interpolation='none')
    fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax1)
    plt.show()
    plt.close()


display_mel(mel_outputs)


# # Let inference other input to check dynamic shape

input_text = "语言合成效果和采用的方法息息相关，大家都是这样认为的吗？"
input_ids = processor.text_to_sequence(input_text, inference=True)

decoder_output, mel_outputs, stop_token_prediction, alignment_history = tacotron2.inference(
    tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
    tf.convert_to_tensor([len(input_ids)], tf.int32),
    tf.convert_to_tensor([0], dtype=tf.int32),
)

display_alignment(alignment_history)

display_mel(mel_outputs)

import logging
import os
import time
import yaml
import pytest
import tensorflow as tf

from tensorflow_tts.configs import Tacotron2Config
from tensorflow_tts.models import TFTacotron2
from tensorflow_tts.utils import return_strategy

from tensorflow_tts.bin.tacotron2.train_tacotron2 import Tacotron2Trainer

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)


@pytest.mark.parametrize(
    "var_train_expr, config_path",
    [
        ("embeddings|decoder_cell", "../tensorflow_tts/bin/tacotron2/conf/tacotron2.v1.yaml"),
        (None, "../tensorflow_tts/bin/tacotron2/conf/tacotron2.v1.yaml"),
        (
            "embeddings|decoder_cell",
            "../tensorflow_tts/bin/tacotron2/conf/tacotron2.baker.v1.yaml",
        ),
    ],
)
def test_tacotron2_train_some_layers(var_train_expr, config_path):
    config = Tacotron2Config(n_speakers=5, reduction_factor=1)
    model = TFTacotron2(config, name="tacotron2")
    model._build()
    optimizer = tf.keras.optimizers.Adam(lr=0.001)

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    config.update({"outdir": "./"})
    config.update({"var_train_expr": var_train_expr})

    STRATEGY = return_strategy()

    trainer = Tacotron2Trainer(
        config=config, strategy=STRATEGY, steps=0, epochs=0, is_mixed_precision=False,
    )
    trainer.compile(model, optimizer)

    len_trainable_vars = len(trainer._trainable_variables)
    all_trainable_vars = len(model.trainable_variables)

    if var_train_expr is None:
        tf.debugging.assert_equal(len_trainable_vars, all_trainable_vars)
    else:
        tf.debugging.assert_less(len_trainable_vars, all_trainable_vars)


@pytest.mark.parametrize(
    "n_speakers, n_chars, max_input_length, max_mel_length, batch_size",
    [(2, 15, 25, 50, 2), (1, 15, 25, 50, 1)],
)
def test_tacotron2_trainable(
    n_speakers, n_chars, max_input_length, max_mel_length, batch_size
):
    config = Tacotron2Config(n_speakers=n_speakers, reduction_factor=1)
    model = TFTacotron2(config, name="tacotron2")
    model._build()

    # fake input
    input_ids = tf.random.uniform(
        [batch_size, max_input_length], maxval=n_chars, dtype=tf.int32
    )
    input_lengths = tf.constant([max_input_length]*batch_size)
    speaker_ids = tf.convert_to_tensor([0] * batch_size, tf.int32)
    mel_gts = tf.random.uniform(shape=[batch_size, max_mel_length, 80])
    mel_lengths = tf.constant([max_mel_length]*batch_size)

    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function(experimental_relax_shapes=True)
    def one_step_training(input_ids, speaker_ids, mel_gts, mel_lengths):
        with tf.GradientTape() as tape:
            mel_preds, post_mel_preds, stop_preds, alignment_history = model(
                input_ids,
                input_lengths,
                speaker_ids,
                mel_gts,
                mel_lengths,
                training=True,
            )
            loss_before = tf.keras.losses.MeanSquaredError()(mel_gts, mel_preds)
            loss_after = tf.keras.losses.MeanSquaredError()(mel_gts, post_mel_preds)

            stop_gts = tf.expand_dims(
                tf.range(tf.reduce_max(mel_lengths), dtype=tf.int32), 0
            )  # [1, max_len]
            stop_gts = tf.tile(stop_gts, [tf.shape(mel_lengths)[0], 1])  # [B, max_len]
            stop_gts = tf.cast(
                tf.math.greater_equal(stop_gts, tf.expand_dims(mel_lengths, 1) - 1),
                tf.float32,
            )

            # calculate stop_token loss
            stop_token_loss = binary_crossentropy(stop_gts, stop_preds)

            loss = stop_token_loss + loss_before + loss_after

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, alignment_history

    for i in range(2):
        if i == 1:
            start = time.time()
        loss, alignment_history = one_step_training(input_ids, speaker_ids, mel_gts, mel_lengths)
        print(f" > loss: {loss}")
    total_runtime = time.time() - start
    print(f" > Total run-time: {total_runtime}")
    print(f" > Avg run-time: {total_runtime/2}")


def test_tflite():
    config = Tacotron2Config(n_speakers=1, reduction_factor=1)
    tacotron2 = TFTacotron2(config, name="tacotron2")
    tacotron2._build()

    # Concrete Function
    tacotron2_concrete_function = tacotron2.inference_tflite.get_concrete_function()

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [tacotron2_concrete_function]
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()

    # Save the TF Lite model.
    with open('tacotron2.tflite', 'wb') as f:
        f.write(tflite_model)

    print('Model size is %f MBs.' % (len(tflite_model) / 1024 / 1024.0))

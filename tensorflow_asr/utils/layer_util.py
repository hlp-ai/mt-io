import tensorflow as tf


def get_rnn(
    rnn_type: str,
):
    assert rnn_type in ["lstm", "gru", "rnn"]
    if rnn_type == "lstm":
        return tf.keras.layers.LSTM
    if rnn_type == "gru":
        return tf.keras.layers.GRU
    return tf.keras.layers.SimpleRNN


def get_conv(
    conv_type: str,
):
    assert conv_type in ["conv1d", "conv2d"]
    if conv_type == "conv1d":
        return tf.keras.layers.Conv1D
    return tf.keras.layers.Conv2D

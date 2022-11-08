
# tf.data.Dataset does not work well for namedtuple so we are using dict

import tensorflow as tf


def create_inputs(
    inputs: tf.Tensor,
    inputs_length: tf.Tensor,
    predictions: tf.Tensor = None,
    predictions_length: tf.Tensor = None,
) -> dict:
    data = {
        "inputs": inputs,
        "inputs_length": inputs_length,
    }
    if predictions is not None:
        data["predictions"] = predictions
    if predictions_length is not None:
        data["predictions_length"] = predictions_length
    return data


def create_logits(
    logits: tf.Tensor,
    logits_length: tf.Tensor,
) -> dict:
    return {"logits": logits, "logits_length": logits_length}


def create_labels(
    labels: tf.Tensor,
    labels_length: tf.Tensor,
) -> dict:
    return {
        "labels": labels,
        "labels_length": labels_length,
    }

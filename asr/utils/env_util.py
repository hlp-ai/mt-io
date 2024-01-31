
import logging
import warnings
from typing import List, Union

import tensorflow as tf

logger = tf.get_logger()


def setup_environment():
    """Setting tensorflow running environment"""
    warnings.simplefilter("ignore")
    logger.setLevel(logging.INFO)
    return logger


def setup_devices(devices: List[int], cpu: bool = False):
    """Setting visible devices

    Args:
        devices (list): list of visible devices' indices
    """
    if cpu:
        cpus = tf.config.list_physical_devices("CPU")
        tf.config.set_visible_devices(cpus, "CPU")
        tf.config.set_visible_devices([], "GPU")
        logger.info(f"Run on {len(cpus)} Physical CPUs")
    else:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            visible_gpus = [gpus[i] for i in devices]
            tf.config.set_visible_devices(visible_gpus, "GPU")
            logger.info(f"Run on {len(visible_gpus)} Physical GPUs")


def setup_strategy(devices: List[int], tpu_address: str = None):
    """Setting mirrored strategy for training

    Args:
        devices (list): list of visible devices' indices
        tpu_address (str): an optional custom tpu address

    Returns:
        tf.distribute.Strategy: MirroredStrategy for training on gpus
    """
    setup_devices(devices)
    return tf.distribute.MirroredStrategy()


def has_devices(devices: Union[List[str], str]):
    if isinstance(devices, list):
        return all([len(tf.config.list_logical_devices(d)) != 0 for d in devices])
    return len(tf.config.list_logical_devices(devices)) != 0

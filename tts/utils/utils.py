"""Utility functions."""

import fnmatch
import os
from pathlib import Path

MODEL_FILE_NAME = "model.h5"
CONFIG_FILE_NAME = "config.yml"
PROCESSOR_FILE_NAME = "processor.json"
LIBRARY_NAME = "tensorflow_tts"
CACHE_DIRECTORY = os.path.join(Path.home(), ".cache", LIBRARY_NAME)


def find_files(root_dir, query="*.wav", include_root_dir=True):
    """Find files recursively.
    Args:
        root_dir (str): Root root_dir to find.
        query (str): Query to find.
        include_root_dir (bool): If False, root_dir name is not included.
    Returns:
        list: List of found filenames.
    """
    files = []
    for root, _, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, query):
            files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files


def save_weights(model, filepath):
    """Save model weights.

    Args:
        model (tf.keras.Model): Model to save.
        filepath (str): Path to save the model weights to.
    """
    model.save_weights(filepath)


def load_weights(model, filepath):
    """Load model weights.

    Args:
        model (tf.keras.Model): Model to load weights to.
        filepath (str): Path to the weights file.
    """
    model.load_weights(filepath)

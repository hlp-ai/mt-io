import os
import fire

from tensorflow_asr.utils import env_util

logger = env_util.setup_environment()
import tensorflow as tf

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")


from tensorflow_asr.featurizers.speech_featurizers import read_raw_audio


def main(
    saved_model: str = None,
    filename: str = None,
):
    tf.keras.backend.clear_session()

    module = tf.saved_model.load(export_dir=saved_model)

    signal = read_raw_audio(filename)
    transcript = module.pred(signal)

    print("Transcript: ", "".join([chr(u) for u in transcript]))


if __name__ == "__main__":
    fire.Fire(main)

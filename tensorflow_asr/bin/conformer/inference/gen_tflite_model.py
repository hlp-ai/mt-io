import os
import fire
from tensorflow_asr.utils import env_util

logger = env_util.setup_environment()
import tensorflow as tf

from tensorflow_asr.configs.config import Config
from tensorflow_asr.helpers import exec_helpers, featurizer_helpers
from tensorflow_asr.models.transducer.conformer import Conformer

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")


def main(
    config: str = DEFAULT_YAML,
    h5: str = None,
    subwords: bool = False,
    sentence_piece: bool = False,
    output: str = None,
):
    assert h5 and output
    tf.keras.backend.clear_session()
    tf.compat.v1.enable_control_flow_v2()

    config = Config(config)
    speech_featurizer, text_featurizer = featurizer_helpers.prepare_featurizers(
        config=config,
        subwords=subwords,
        sentence_piece=sentence_piece,
    )

    conformer = Conformer(**config.model_config, vocabulary_size=text_featurizer.num_classes)
    conformer.make(speech_featurizer.shape)
    conformer.load_weights(h5, by_name=True)
    conformer.summary(line_length=100)
    conformer.add_featurizers(speech_featurizer, text_featurizer)

    exec_helpers.convert_tflite(model=conformer, output=output)


if __name__ == "__main__":
    fire.Fire(main)

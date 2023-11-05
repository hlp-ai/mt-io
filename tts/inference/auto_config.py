"""Tensorflow Auto Config modules."""

import yaml
from collections import OrderedDict

from tts.configs import (
    FastSpeech2Config,
    MelGANGeneratorConfig,
    MultiBandMelGANGeneratorConfig,
    Tacotron2Config,
)

CONFIG_MAPPING = OrderedDict(
    [
        ("fastspeech2", FastSpeech2Config),
        ("multiband_melgan_generator", MultiBandMelGANGeneratorConfig),
        ("melgan_generator", MelGANGeneratorConfig),
        ("tacotron2", Tacotron2Config),
    ]
)


class AutoConfig:
    def __init__(self):
        raise EnvironmentError(
            "AutoConfig is designed to be instantiated "
            "using the `AutoConfig.from_pretrained(pretrained_path)` method."
        )

    @classmethod
    def from_pretrained(cls, pretrained_path, **kwargs):
        with open(pretrained_path) as f:
            config = yaml.load(f, Loader=yaml.Loader)

        try:
            model_type = config["model_type"]
            config_class = CONFIG_MAPPING[model_type]
            config_class = config_class(**config[model_type + "_params"], **kwargs)
            config_class.set_config_params(config)
            return config_class
        except Exception:
            raise ValueError(
                "Unrecognized config in {}. "
                "Should have a `model_type` key in its config.yaml, or contain one of the following strings "
                "in its name: {}".format(
                    pretrained_path, ", ".join(CONFIG_MAPPING.keys())
                )
            )

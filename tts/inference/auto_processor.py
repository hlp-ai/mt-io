"""Tensorflow Auto Processor modules."""

import json
from collections import OrderedDict

from tts.processor import (
    LJSpeechProcessor,
    BakerProcessor,
)

CONFIG_MAPPING = OrderedDict(
    [
        ("LJSpeechProcessor", LJSpeechProcessor),
        ("BakerProcessor", BakerProcessor),
    ]
)


class AutoProcessor:
    def __init__(self):
        raise EnvironmentError(
            "AutoProcessor is designed to be instantiated "
            "using the `AutoProcessor.from_pretrained(pretrained_path)` method."
        )

    @classmethod
    def from_pretrained(cls, pretrained_path, **kwargs):
        with open(pretrained_path, "r") as f:
            config = json.load(f)

        try:
            processor_name = config["processor_name"]
            processor_class = CONFIG_MAPPING[processor_name]
            processor_class = processor_class(data_dir=None, loaded_mapper_path=pretrained_path)
            return processor_class
        except Exception:
            raise ValueError(
                "Unrecognized processor in {}. "
                "Should have a `processor_name` key in its config.json, or contain one of the following strings "
                "in its name: {}".format(
                    pretrained_path, ", ".join(CONFIG_MAPPING.keys())
                )
            )

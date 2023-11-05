"""Tensorflow Auto Model modules."""
from collections import OrderedDict

from tts.configs import (
    FastSpeech2Config,
    MelGANGeneratorConfig,
    MultiBandMelGANGeneratorConfig,
    Tacotron2Config,
)

from tts.models import (
    TFMelGANGenerator,
    TFMBMelGANGenerator,
)

from tts.inference.savable_models import (
    SavableTFFastSpeech2,
    SavableTFTacotron2
)


TF_MODEL_MAPPING = OrderedDict(
    [
        (FastSpeech2Config, SavableTFFastSpeech2),
        (MultiBandMelGANGeneratorConfig, TFMBMelGANGenerator),
        (MelGANGeneratorConfig, TFMelGANGenerator),
        (Tacotron2Config, SavableTFTacotron2),
    ]
)


class TFAutoModel(object):
    """General model class for inferencing."""

    def __init__(self):
        raise EnvironmentError("Cannot be instantiated using `__init__()`")

    @classmethod
    def from_pretrained(cls, pretrained_path=None, config=None, **kwargs):
        assert config is not None, "Please make sure to pass a config along to load a model from a local file"

        for config_class, model_class in TF_MODEL_MAPPING.items():
            if isinstance(config, config_class) and str(config_class.__name__) in str(
                config
            ):
                model = model_class(config=config, **kwargs)
                model.set_config(config)
                model._build()
                if pretrained_path is not None and ".h5" in pretrained_path:
                    try:
                        model.load_weights(pretrained_path)
                    except:
                        model.load_weights(
                            pretrained_path, by_name=True, skip_mismatch=True
                        )
                return model

        raise ValueError(
            "Unrecognized configuration class {} for this kind of TFAutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in TF_MODEL_MAPPING.keys()),
            )
        )

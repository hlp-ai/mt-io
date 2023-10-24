from tts.models.base_model import BaseModel
from tts.models.fastspeech import TFFastSpeech
from tts.models.fastspeech2 import TFFastSpeech2
from tts.models.melgan import (
    TFMelGANDiscriminator,
    TFMelGANGenerator,
    TFMelGANMultiScaleDiscriminator,
)
from tts.models.mb_melgan import TFPQMF
from tts.models.mb_melgan import TFMBMelGANGenerator

from tts.models.tacotron2 import TFTacotron2

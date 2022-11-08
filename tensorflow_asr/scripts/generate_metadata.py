import os
import argparse
from tensorflow_asr.configs.config import Config
from tensorflow_asr.utils.file_util import preprocess_paths
from tensorflow_asr.datasets.asr_dataset import ASRDataset
from tensorflow_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
from tensorflow_asr.featurizers.text_featurizers import SubwordFeaturizer, SentencePieceFeaturizer

parser = argparse.ArgumentParser(prog="Dataset Metadata Creation")

parser.add_argument("--stage", type=str, default="train", help="The stage of dataset")

parser.add_argument("--config", type=str, default=None, help="The file path of model configuration file")

parser.add_argument("--sentence_piece", default=False, action="store_true", help="Whether to use `SentencePiece` model")

parser.add_argument("--metadata", type=str, default=None, help="Path to file containing metadata")

parser.add_argument("--subwords", type=str, default=None, help="Path to file that stores generated subwords")

parser.add_argument("transcripts", nargs="+", type=str, default=None, help="Paths to transcript files")

args = parser.parse_args()

assert args.metadata is not None, "metadata must be defined"

transcripts = preprocess_paths(args.transcripts)

config = Config(args.config)

speech_featurizer = TFSpeechFeaturizer(config.speech_config)

if args.sentence_piece:
    print("Loading SentencePiece model ...")
    text_featurizer = SentencePieceFeaturizer.load_from_file(config.decoder_config, args.subwords)
elif args.subwords and os.path.exists(args.subwords):
    print("Loading subwords ...")
    text_featurizer = SubwordFeaturizer.load_from_file(config.decoder_config, args.subwords)

dataset = ASRDataset(
    data_paths=transcripts,
    speech_featurizer=speech_featurizer, text_featurizer=text_featurizer,
    stage=args.stage, shuffle=False,
)

dataset.update_metadata(args.metadata)

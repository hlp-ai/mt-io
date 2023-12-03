"""Runing mfa to extract textgrids."""

from subprocess import call
from pathlib import Path

import click
import os


@click.command()
@click.option("--mfa_path", default=os.path.join('mfa', 'montreal-forced-aligner', 'bin', 'mfa_align'))
@click.option("--corpus_directory", default="libritts")
@click.option("--lexicon", default=os.path.join('mfa', 'lexicon', 'librispeech-lexicon.txt'))
@click.option("--acoustic_model_path", default=os.path.join('mfa', 'montreal-forced-aligner', 'pretrained_models', 'english.zip'))
@click.option("--output_directory", default=os.path.join('mfa', 'parsed'))
@click.option("--jobs", default="8")
def run_mfa(
    mfa_path: str,
    corpus_directory: str,
    lexicon: str,
    acoustic_model_path: str,
    output_directory: str,
    jobs: str,
):
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    call(
        [
            f".{os.path.sep}{mfa_path}",
            corpus_directory,
            lexicon,
            acoustic_model_path,
            output_directory,
            f"-j {jobs}"
         ]
    )


if __name__ == "__main__":
    run_mfa()

"""1. 从librispeech语料创建转写或元数据文件"""
import os
import glob
import argparse
import librosa
from tqdm.auto import tqdm
import unicodedata

from asr.utils.file_util import preprocess_paths

parser = argparse.ArgumentParser(prog="Setup LibriSpeech Transcripts")

parser.add_argument("--dir", "-d", type=str, default=None, help="Directory of dataset")
parser.add_argument("--max_len", type=int, default=17, help="Maximum length of audio allowed, in sec")

parser.add_argument("output", type=str, default=None, help="The output .tsv transcript file path")

args = parser.parse_args()

assert args.dir and args.output

args.dir = preprocess_paths(args.dir, isdir=True)
args.output = preprocess_paths(args.output)

transcripts = []

threshold_len_secs = args.max_len
max_len_secs = 0
filtered  = 0

text_files = glob.glob(os.path.join(args.dir, "**", "*.txt"), recursive=True)

for text_file in tqdm(text_files, desc="[Loading]"):
    current_dir = os.path.dirname(text_file)
    with open(text_file, "r", encoding="utf-8") as txt:
        lines = txt.read().splitlines()
    for line in lines:
        line = line.split(" ", maxsplit=1)
        audio_file = os.path.join(current_dir, line[0] + ".flac")
        y, sr = librosa.load(audio_file, sr=None)
        duration = librosa.get_duration(y, sr)
        if duration > max_len_secs:
            max_len_secs = duration
        if duration > threshold_len_secs:
            filtered += 1
            continue
        text = unicodedata.normalize("NFC", line[1].lower())
        transcripts.append(f"{audio_file}\t{duration}\t{text}\n")

with open(args.output, "w", encoding="utf-8") as out:
    out.write("PATH\tDURATION\tTRANSCRIPT\n")
    for line in tqdm(transcripts, desc="[Writing]"):
        out.write(line)

print("max_len_secs:", max_len_secs)
print("Filtered:", filtered)

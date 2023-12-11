import argparse
from tqdm.auto import tqdm

parser = argparse.ArgumentParser(prog="Create vocabulary file from transcripts")

parser.add_argument("--output", type=str, default=None, help="The output .txt vocabulary file path")

parser.add_argument("transcripts", nargs="+", type=str, default=None, help="Transcript .tsv files")

args = parser.parse_args()

assert args.output and args.transcripts

lines = []
for trans in args.transcripts:
    with open(trans, "r", encoding="utf-8") as t:
        lines.extend(t.read().splitlines()[1:])

vocab = {}
for line in tqdm(lines, desc="[Processing]"):
    line = line.split("\t")[-1]
    for c in line:
        vocab[c] = 1

with open(args.output, "w", encoding="utf-8") as out:
    for key in vocab.keys():
        out.write(f"{key}\n")

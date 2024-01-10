import os
from collections import Counter

from ocr.syn.clean import clean_text

corpus_dir = [r"D:\dataset\zh-x-val\flores200\flores200_dev",
              r"D:\dataset\zh-x-val\flores200\flores200_devtest"]

script = "Arab"  # Latn, Cyrl, Deva
files = []

for d in corpus_dir:
    fs = os.listdir(d)
    for f in fs:
        if f.find(script) > 0:
            files.append(os.path.join(d, f))


print("# of files:", len(files))


out_file = "./text-{}.txt".format(script)
out_vocab = "./vocab-{}.txt".format(script)

lines = 0
chars = 0

char2count = Counter()

outf = open(out_file, "w", encoding="utf-8")
outv = open(out_vocab, "w", encoding="utf-8")

for file in files:
    print(file)
    with open(file, encoding="utf-8") as f:
        for line in f:
            parts = line.split("\t")
            s = clean_text(parts[1])  # the second column
            char2count.update(s)
            lines += 1
            chars += len(s)
            outf.write(s + "\n")

print(f"# of Lines: {lines}, # of chars: {chars}, # of char type: {len(char2count)}")
for c, n in char2count.items():
    print(c, hex(ord(c)))

for c, n in char2count.items():
    outv.write(c + "\n")

outf.close()
outv.close()

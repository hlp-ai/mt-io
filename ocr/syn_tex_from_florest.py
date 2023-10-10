import os
import unicodedata
from collections import Counter


def is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True

    return False


def is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False


def clean_text(text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or is_control(char):
            continue
        if is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


corpus_dir = [r"D:\dataset\zh-x-val\flores200\flores200_dev",
              r"D:\dataset\zh-x-val\flores200\flores200_devtest"]

script = "Latn"  # Latn, Cyrl, Deva
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
    if c == " ":
        continue
    outv.write(c + "\n")

outf.close()
outv.close()

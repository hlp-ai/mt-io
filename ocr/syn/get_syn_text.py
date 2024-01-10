
in_file = r"D:\dataset\mnmt\tsv\zh\en-zh.tsv.zh"
n = 150000
out_file = "{}-{}".format(in_file, n)

with  open(in_file, encoding="utf-8") as f, open(out_file, "w", encoding="utf-8") as out:
    i = 0
    for line in f:
        line = line.strip()
        out.write(line + "\n")
        i += 1

        if i > n:
            break

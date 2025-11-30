import os
from pathlib import Path
import random

IMAGES = Path("/home/personal/Desktop/mowito/data/images")
MASKS = Path("/home/personal/Desktop/mowito/data/masks")
OUT = Path("/home/personal/Desktop/mowito/data/splits")

OUT.mkdir(parents=True, exist_ok=True)

random.seed(42)

images = sorted([p.name for p in IMAGES.glob("*") if p.is_file()])

good = []
bad = []

for img in images:
    mask_path = MASKS / img
    if mask_path.exists():
        bad.append(img)
    else:
        good.append(img)

print("Good images:", len(good))
print("Bad images:", len(bad))

def split(lst, t=0.8, v=0.1):
    random.shuffle(lst)
    n = len(lst)
    tn = int(n*t)
    vn = int(n*v)
    return lst[:tn], lst[tn:tn+vn], lst[tn+vn:]

g_train, g_val, g_test = split(good)
b_train, b_val, b_test = split(bad)

def write_split(fname, items):
    with open(OUT/fname, "w") as f:
        for x in items:
            f.write(x + "\n")

write_split("train.txt", g_train + b_train)
write_split("val.txt",   g_val   + b_val)
write_split("test.txt",  g_test  + b_test)

print("Done! Splits generated.")

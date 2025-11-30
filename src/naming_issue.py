import os
import shutil
from pathlib import Path

GOOD = Path("/home/personal/Desktop/mowito/good")
BAD = Path("/home/personal/Desktop/mowito/bad")
MASK = Path("/home/personal/Desktop/mowito/masks")     
OUT_IMG = Path("/home/personal/Desktop/mowito/data/images")
OUT_MASK = Path("/home/personal/Desktop/mowito/data/masks")

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_MASK.mkdir(parents=True, exist_ok=True)

def process_folder(folder, prefix, copy_mask=False):
    for f in folder.glob("*"):
        if not f.is_file():
            continue

        new_name = f"{prefix}_{f.name}"
        dest_img = OUT_IMG / new_name
        shutil.copy(f, dest_img)


        if copy_mask:
            mask_file = MASK / f.name
            if mask_file.exists():
                dest_mask = OUT_MASK / new_name
                shutil.copy(mask_file, dest_mask)

print("Processing GOOD images...")
process_folder(GOOD, "good", copy_mask=False)

print("Processing BAD images...")
process_folder(BAD, "bad", copy_mask=True)

print("DONE! Unique names created in:")
print("  data/images/")
print("  data/masks/")

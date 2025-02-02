import os
import json
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

JSON_DIR = "/home/bcxu/nnunet/nnUNet_raw/Dataset001_sampling/labelsTr"             # Folder containing the JSON files
OUTPUT_DIR = "/home/bcxu/nnunet/nnUNet_raw/Dataset001_sampling/labelsTr"       # Where PNGs will be saved
os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(JSON_DIR):
    if not filename.endswith(".json"):
        continue

    json_path = os.path.join(JSON_DIR, filename)
    
    with open(json_path, "r") as f:
        data = json.load(f)

    # We assume the JSON has keys: "labels", "masks", "scores", etc.
    # data["labels"] -> list of integer label IDs
    # data["masks"]  -> list of dicts, each containing {"size": [H,W], "counts": RLE_string}
    labels = data.get("labels", [])
    masks_rle = data.get("masks", [])

    if len(labels) == 0 or len(masks_rle) == 0:
        print(f"Skipping {filename}: No labels or masks found.")
        continue

    # All RLE masks share the same image size => derived from first "masks" entry
    # or each "masks" entry has "size": [height, width]
    height, width = masks_rle[0]["size"]

    # Create an empty multi-class mask
    # 8-bit is usually enough for small label sets. If your label IDs are large, consider uint16.
    multi_class_mask = np.zeros((height, width), dtype=np.uint8)

    # Loop through each RLE mask + label
    for i, rle_obj in enumerate(masks_rle):
        label_value = labels[i]  # The integer label for this mask
        # If label_value == 0, that means "background." Usually we skip or overwrite the background with other classes.
        # But if your dataset uses 0 for an actual class, you'll want to handle it accordingly.

        if "size" not in rle_obj or "counts" not in rle_obj:
            print(f"Skipping an RLE in {filename} due to missing 'size'/'counts'.")
            continue

        # Decode the RLE to a binary mask
        decoded = coco_mask.decode({
            "size": rle_obj["size"],   # e.g. [1080, 1920]
            "counts": rle_obj["counts"]
        })  # decoded is a 2D numpy array of 0s and 1s

        # Write label_value into multi_class_mask wherever decoded == 1
        # Make sure your label_value won't exceed 255 if you use uint8!
        # If you have many classes, consider using uint16 for the mask.
        multi_class_mask[decoded == 1] = label_value

    # Save the PNG
    base_name = os.path.splitext(filename)[0]  # e.g. "417_VID003A_1020"
    out_path = os.path.join(OUTPUT_DIR, base_name + ".png")
    Image.fromarray(multi_class_mask).save(out_path)
    print(f"Saved multi-class mask: {out_path}")

print("Done converting JSON -> multi-class PNG masks.")

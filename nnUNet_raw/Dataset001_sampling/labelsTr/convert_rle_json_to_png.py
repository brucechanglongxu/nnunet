import os
import json
import numpy as np
from PIL import Image
from pycocotools import mask

# Define input/output directories
json_dir = "/home/bcxu/nnunet/nnUNet_raw/Dataset001_sampling/labelsTr"
output_dir = "/home/bcxu/nnunet/nnUNet_raw/Dataset001_sampling/labelsTr_png"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each JSON file
for json_file in os.listdir(json_dir):
    if not json_file.endswith(".json"):
        continue

    json_path = os.path.join(json_dir, json_file)

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        # Check JSON format
        if not isinstance(data, list):
            print(f"Skipping {json_file}: Expected a list, got {type(data)}")
            continue

        for obj in data:
            if "size" not in obj or "counts" not in obj:
                print(f"Skipping {json_file}: Missing 'size' or 'counts' field")
                continue

            mask_size = obj["size"]
            rle_counts = obj["counts"]

            print(f"Processing {json_file} with size {mask_size}")

            # Decode RLE mask
            decoded_mask = mask.decode({"size": mask_size, "counts": rle_counts})

            # Convert mask to uint8 (0 = background, 255 = foreground)
            binary_mask = (decoded_mask * 255).astype(np.uint8)

            # Save the PNG file
            png_filename = os.path.splitext(json_file)[0] + ".png"
            png_path = os.path.join(output_dir, png_filename)

            Image.fromarray(binary_mask).save(png_path)
            print(f"Converted {json_file} -> {png_filename}")

    except Exception as e:
        print(f"Error processing {json_file}: {e}")

print("Conversion complete.")

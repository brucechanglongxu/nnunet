import os
from PIL import Image

input_dir = "imagesTr"
output_dir = "imagesTr_png"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".jpg"):
        img = Image.open(os.path.join(input_dir, filename))
        img.save(os.path.join(output_dir, filename.replace(".jpg", ".png")), "PNG")


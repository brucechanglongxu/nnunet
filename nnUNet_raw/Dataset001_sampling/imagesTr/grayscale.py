from PIL import Image
import os

images_dir = "/home/bcxu/nnunet/nnUNet_raw/Dataset001_sampling/imagesTr"
for fname in os.listdir(images_dir):
    if fname.endswith(".png"):
        img_path = os.path.join(images_dir, fname)
        img = Image.open(img_path).convert("L")  # grayscale
        img.save(img_path)

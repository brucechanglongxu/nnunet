import json
import os

COCO_JSON_PATH = "/home/bcxu/nnunet/nnUNet_raw/Dataset001_sampling/dataset.json"  # Original COCO-like file
OUTPUT_JSON_PATH = "/home/bcxu/nnunet/nnUNet_raw/Dataset001_sampling/newdataset.json"

# If your training images are .png and located in imagesTr
FILE_ENDING = ".png"

# Define channel names. For single-RGB images, we only need channel 0
CHANNEL_NAMES = {
    "0": "RGB"
}

# Provide a name for the background label (ID = 0)
BACKGROUND_LABEL_NAME = "background"

def main():
    with open(COCO_JSON_PATH, "r") as f:
        coco_data = json.load(f)

    # 1) Build the "labels" dict from the "categories" list.
    #    COCO categories often look like: [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}, ...]
    #    nnU-Net needs: {"background": 0, "cat": 1, "dog": 2, ...}
    labels_dict = {BACKGROUND_LABEL_NAME: 0}  # We'll start with background=0

    categories = coco_data.get("categories", [])
    for cat in categories:
        cat_id = cat["id"]   # e.g. 1
        cat_name = cat["name"]  # e.g. "dog"
        # We assume cat_id starts at 1, so each category_id becomes a label index
        labels_dict[cat_name] = cat_id  

    # 2) Count how many training images ( = length of the "images" list or however you store them)
    #    If you're only labeling a subset, adapt accordingly. For typical COCO, "images" is the full list.
    images_list = coco_data.get("images", [])
    num_training = len(images_list)

    # 3) Construct the minimal dataset.json structure for nnU-Net
    dataset_json = {
        "channel_names": CHANNEL_NAMES,      # e.g. {"0": "RGB"}
        "labels": labels_dict,               # e.g. {"background":0, "dog":1, "cat":2, ...}
        "numTraining": num_training,         # how many imagesTr you actually have
        "file_ending": FILE_ENDING          # e.g. ".png"
        # optionally add "overwrite_image_reader_writer": "NaturalImage2DIO" for 2D PNG
    }

    # 4) Write out the minimal JSON
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w") as out_f:
        json.dump(dataset_json, out_f, indent=4)

    print(f"Wrote nnU-Net dataset.json to: {OUTPUT_JSON_PATH}")
    print("Keys in new dataset.json:", dataset_json.keys())

if __name__ == "__main__":
    main()


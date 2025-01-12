import os 
import sys
import json
import math
import random
import argparse
from pathlib import Path
from copy import deepcopy
sys.path.insert(0, os.getcwd())

from PIL import Image
from tqdm import tqdm


def aug_transform(pil_image):
    W, H = pil_image.size
    max_pixel = random.uniform(args.min_pixel, args.min_pixel+1)
    if W*H > max_pixel:
        resize_factor = math.sqrt(max_pixel / (pil_image.width * pil_image.height))
        width, height = int(pil_image.width * resize_factor), int(pil_image.height * resize_factor)
        pil_image = pil_image.resize((width, height), resample=Image.NEAREST)
    return pil_image


def main():
    with open(args.json_file, "r") as f:
        records = json.load(f)

    new_records = []
    for epoch in range(args.epochs):
        Path(os.path.join(args.image_dst, f"epoch_{epoch}")).mkdir(exist_ok=True, parents=True)
        for record in tqdm(records):
            for file in record["image"]:
                for image_src in args.image_src_list:
                    path = os.path.join(image_src, file)
                    if os.path.isfile(path):
                        break
                
                image = Image.open(path)
                image = aug_transform(image)
                image.save(os.path.join(args.image_dst, f"epoch_{epoch}", file))
            new_record = deepcopy(record)
            new_record["image"] = [os.path.join(f"epoch_{epoch}", ele) for ele in record["image"]]
            new_records.append(new_record)
        
        image_count = len(os.listdir(os.path.join(args.image_dst, f"epoch_{epoch}")))
        print(f"Epoch {epoch} has {image_count} images")
    
    with open("../data/mire/final/train_aug.json", "w", encoding="utf-8") as f:
        json.dump(new_records, f, ensure_ascii=False, indent=4)
    print("Save new records at '../data/mire/final/train_aug.json'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--json_file", type=str, required=True)
    parser.add_argument("--image_dst", type=str, required=True)
    parser.add_argument("--image_src_list", type=str, nargs="+")
    parser.add_argument("--max_pixel", type=str, default=1000000)
    parser.add_argument("--min_pixel", type=str, default=600000)
    args = parser.parse_args()
    main() 
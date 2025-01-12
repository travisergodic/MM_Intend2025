import os
import sys
import math
import json
import argparse
sys.path.insert(0, os.getcwd())

from PIL import Image

from src.utils import *

max_side = 616
pixel_margin=None

def main():
    src = args.src
    dst = args.dst

    with open(args.raw_file, "r") as f:
        records = json.load(f)

    for record in records:
        instruction = record["instruction"]
        if not is_intend_instruction(instruction):
            continue
        
        image_paths = [os.path.join(src, file) for file in record["image"]]
        images = [Image.open(path) for path in image_paths]

        for image in images:
            if pixel_margin is not None:
                pixel_sum = sum([image.size[0]*image.size[1] for image in images])
                if pixel_margin >= pixel_sum:
                    continue
                resize_ratio = math.sqrt(pixel_margin / pixel_sum)
                
            if max_side is not None:
                W, H = image.size
                if max_side >= max(W, H):
                    continue
                resize_ratio = max_side / max(W, H)
                
    
            for image, filename in zip(images, record["image"]):
                W, H = image.size
                W, H = int(resize_ratio*W), int(resize_ratio*H)
                image.resize((W, H)).save(os.path.join(dst, filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--raw_file", type=str, required=True)
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)
    args = parser.parse_args()
    main() 
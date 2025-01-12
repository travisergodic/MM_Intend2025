import os
import sys
import json
import argparse
sys.path.append("./PaddleOCR")

from PIL import Image
from tqdm import tqdm
from paddleocr import PaddleOCR


def main():
    ocr = PaddleOCR(use_angle_cls=args.use_angle_cls, lang='ch', ocr_version='PP-OCRv3', use_gpu=args.use_gpu)

    with open(args.json_file, 'r') as f:
        data = json.load(f)

    records = []
    for ele in tqdm(data):
        for filename in ele["image"]:
            image_path = os.path.join(args.image_root, os.path.basename(filename))
            res = ocr.ocr(image_path, cls=True)[0]
            try:
                record = {
                    "image": filename,
                    "text": [ele[1][0] for ele in res],
                    "bbox": [ele[0] for ele in res]
                }
            except:
                record = {
                    "image": filename,
                    "text": [],
                    "bbox": []
                }
            records.append(record)

    with open(args.save_path, "w", encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=4) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--json_file", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--use_angle_cls", action="store_true")
    args = parser.parse_args()
    main()
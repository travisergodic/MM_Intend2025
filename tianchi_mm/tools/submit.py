import os
import sys
import argparse
sys.path.insert(0, os.getcwd())

import pandas as pd

from src.utils import *
from src.constants import IMAGE_SCENCE_CLASSES, INTEND_CLASSES
from src.transform import TRANSFORM
from src.logger_helper import setup_logger


logger = setup_logger()


def main():
    # read
    test_data = read_json(args.raw_file)
    predict_data = [json.loads(line) for line in open(args.predict_file, "r")]

    logger.info(f"Found {len(test_data)} test records!")
    logger.info(f"Found {len(predict_data)} prediction records!") 
        
    exceed_max_length_idxs = [record["id"] for record in test_data if len(record["image"]) > args.max_image_length]
    assert len(test_data)-len(predict_data) - len(exceed_max_length_idxs) == 0

    # transform
    transformer = TRANSFORM.build(type=args.type)

    pred_curr_idx = 0
    new_records = []
    for i, example in enumerate(test_data):
        if len(example["image"]) > args.max_image_length: 
            example["predict"] = "包装区别"
        else:
            instruction = example["instruction"]
            label = predict_data[pred_curr_idx]["predict"].strip().strip(".").strip("。")
            example["predict"] = transformer.label_to_name(instruction, label)
            pred_curr_idx += 1
        new_records.append(example)

    assert len(new_records) == len(test_data)
    
    df = pd.DataFrame(new_records)

    print(df["predict"].value_counts())

    invalid_predictions = df.loc[~df["predict"].isin(IMAGE_SCENCE_CLASSES + INTEND_CLASSES), "predict"].tolist()
    logger.info(f"Found {len(invalid_predictions)} invalid predictions: \n {invalid_predictions}")

    save_path = os.path.join(args.save_dir, os.path.basename(args.predict_file).rsplit(".")[0] + ".csv")
    df.to_csv(save_path, index=None, encoding="utf-8-sig")
    logger.info(f"Save submission file at {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--raw_file", type=str, required=True)
    parser.add_argument("--predict_file", type=str, required=True)
    parser.add_argument("--type", type=str, choices=["identity", "split", "idx"], default="identity")
    parser.add_argument("--max_image_length", type=int, default=100)
    parser.add_argument("--save_dir", type=str)
    args = parser.parse_args()
    main() 
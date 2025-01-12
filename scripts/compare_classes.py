import os
import sys
import json
import argparse
sys.path.insert(0, os.getcwd())

from tianchi_mm.src.utils import *
from tianchi_mm.src.backend import access_client
from tianchi_mm.src.agent import ClassComparisonAgent
from tianchi_mm.src.constants import IMAGE_SCENE_LABEL_TO_DESC


def main():
    with open(args.json_file, "r") as f:
        data = json.load(f)

    logger.info(f"Found {len(data)} data")

    openai_client = access_client(api_key=args.api_key, backend=args.backend)

    # agent
    comparision_agent = ClassComparisonAgent(
        openai_client, args.model_name, temperature=0.3, retry=2
    )
    args.class1 = args.class1.strip()
    args.class2 = args.class2.strip()
    
    if (args.class1 in IMAGE_SCENCE_CLASSES) and (args.class1 in IMAGE_SCENCE_CLASSES):
        class1_desc = IMAGE_SCENE_LABEL_TO_DESC[args.class1]
        class2_desc = IMAGE_SCENE_LABEL_TO_DESC[args.class2]
    else:
        raise ValueError()

    logger.info(f"{args.class1}: {class1_desc}")
    logger.info(f"{args.class2}: {class2_desc}")
     
    class1_image_paths = [
       os.path.join(args.image_root, os.path.basename(record["image"][0])) for record in data if record["output"]==args.class1
    ]
    class2_image_paths = [
       os.path.join(args.image_root, os.path.basename(record["image"][0])) for record in data if record["output"]==args.class2
    ]
    logger.info(f"Found {len(class1_image_paths)} images with class {args.class1}")
    logger.info(f"Found {len(class2_image_paths)} images with class {args.class2}")

    result = comparision_agent.do(
        class1_desc=class1_desc, 
        class2_desc=class2_desc, 
        class1_image_paths=class1_image_paths,
        class2_image_paths=class2_image_paths
    )
    logger.info(str({args.class1: result["class1"], args.class2: result["class2"]}))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--backend", type=str, choices=["openai", "ali", "llama"])
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--json_file", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--class1", type=str, required=True)
    parser.add_argument("--class2", type=str, required=True)
    args = parser.parse_args()
    main()
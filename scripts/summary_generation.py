import os
import sys
import json
import argparse
sys.path.insert(0, os.getcwd())

from tianchi_mm.src.utils import *
from tianchi_mm.src.backend import access_client
from tianchi_mm.src.agent import ClassSummaryAgent
from tianchi_mm.src.constants import IMAGE_SCENE_LABEL_TO_DESC


def main():
    with open(args.json_file, "r") as f:
        data = json.load(f)

    logger.info(f"Found {len(data)} data")

    openai_client = access_client(api_key=args.api_key, backend=args.backend)

    # agent
    summary_agent = ClassSummaryAgent(
        openai_client, args.model_name, temperature=0.3, retry=2
    )
    args.cls_name = args.cls_name.strip()
    
    if args.cls_name in IMAGE_SCENCE_CLASSES:
        cls_desc = IMAGE_SCENE_LABEL_TO_DESC[args.cls_name]
    else:
        raise ValueError()

    image_paths = [
       os.path.join(args.image_root, os.path.basename(record["image"][0])) for record in data if record["output"]==args.cls_name
    ][:args.n]

    logger.info(f"'{args.cls_name}' description: {cls_desc}")
    logger.info(f"Collect {len(image_paths)} images of class='{args.cls_name}'")

    summary_text = summary_agent.do(
        class_name=args.cls_name, 
        class_desc=cls_desc, 
        image_paths=image_paths
    )
    logger.info(f"summary text: {summary_text}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--backend", type=str, choices=["openai", "ali", "llama"])
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--json_file", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--cls_name", type=str, required=True)
    parser.add_argument("--n", type=int, required=True)
    args = parser.parse_args()
    main()
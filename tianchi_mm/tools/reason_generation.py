import os
import sys
import json
import argparse
from copy import deepcopy
sys.path.insert(0, os.getcwd())

from src.utils import *
from src.backend import access_client
from src.agent import ImageSceneSBSAgent, IntendSBSAgent, CriticAgent
from src.constants import IMAGE_SCENE_LABEL_TO_NUMBER, INTEND_LABEL_TO_NUMBER, IMAGE_SCENCE_CLASSES, INTEND_CLASSES



def main():
    with open(args.json_file, "r") as f:
        data = json.load(f)

    logger.info(f"Found {len(data)} data")

    openai_client = access_client(api_key=args.api_key, backend=args.backend)

    # agent
    image_scene_cot_agent = ImageSceneSBSAgent(
        client=openai_client, 
        model_name=args.model_name, 
        temperature=0.3
    )
    intend_cot_agent = IntendSBSAgent(
        client=openai_client, 
        model_name=args.model_name, 
        temperature=0.3
    )
    critic_agent = CriticAgent(client=openai_client, model_name=args.model_name, temperature=0.1)
    
    fail_idxs, sbs_records = [], []
    for record in data:
        instruction = record["instruction"]
        label = record["output"]

        if is_image_scene_instruction(instruction):
            sbs_agent = image_scene_cot_agent
            label_to_number = IMAGE_SCENE_LABEL_TO_NUMBER
            label_list = IMAGE_SCENCE_CLASSES

        elif is_intend_instruction(instruction):
            sbs_agent = intend_cot_agent
            label_to_number = INTEND_LABEL_TO_NUMBER
            label_list = INTEND_CLASSES
        else:
            continue

        curr_record = deepcopy(record)
        image_paths = [os.path.join(args.image_root, os.path.basename(ele)) for ele in record["image"]] 
        response = sbs_agent.do(instruction, image_paths, label)

        if response is not None:
            response_number = critic_agent.do(response)
        else:
            response_number = None
            response = ""

        status = "fail" if (response_number != label_to_number[label]) else "success"

        if status == "fail":
            fail_idxs.append(curr_record["id"])

        curr_record.update(
            {
                "cot": response, 
                "gt": label, 
                "pred": label_list[response_number-1] if isinstance(response_number, int) else "null", 
                "status": status
            }
        )

        sbs_records.append(curr_record)
        # save
        with open(args.save_path, "w", encoding="utf-8") as f:
            json.dump(sbs_records, f, ensure_ascii=False, indent=4)

    logger.info(f"Number of fail case: {len(fail_idxs)}")
    logger.info(f"Fail idxs: {fail_idxs}")
    logger.info(f"Error rate: {len(fail_idxs)/len(data)}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--backend", type=str, choices=["openai", "ali", "llama", "nvidia", "gemini", "claude"])
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--json_file", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()
    main()
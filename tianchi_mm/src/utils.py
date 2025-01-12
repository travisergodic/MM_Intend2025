import re
import yaml
import logging
import base64
import json
from copy import deepcopy

from src.prompt import *
from src.constants import IMAGE_SCENCE_CLASSES, INTEND_CLASSES
from src.prompt import *


logger = logging.getLogger(__name__)


def extract_classification_text(text):
    return text.split("<用户与客服的对话 END>")[-1].strip()

    
def get_label_list_start_end_index(classification_text):
    match_indices = [(m.start(), m.end()) for m in re.finditer(r'\[.*?\]', classification_text)]
    assert len(match_indices)==1
    return match_indices[0]


def load_yaml(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_intend_records(records):
    intend_records = []
    for record in records:
        if record["output"] in INTEND_CLASSES:
            intend_records.append(deepcopy(record))
    return intend_records


def get_image_scene_records(records):
    intend_records = []
    for record in records:
        if record["output"] in IMAGE_SCENCE_CLASSES:
            intend_records.append(deepcopy(record))
    return intend_records


def is_intend_instruction(instruction):
    if "你是一个电商客服专家" in instruction:
        return True
    return False


def is_image_scene_instruction(instruction):
    if "你是一个电商领域识图专家" in instruction: 
        return True
    return False


def find_matching_bracket(input_string):
    # Initialize variables to track bracket count and index
    bracket_count = 0
    first_bracket_index = -1
    corresponding_closing_bracket_index = -1

    # Loop through the string to find the matching closing bracket
    for i, char in enumerate(input_string):
        if char == '{':
            if bracket_count == 0:
                first_bracket_index = i
            bracket_count += 1
        elif char == '}':
            bracket_count -= 1
            if bracket_count == 0 and first_bracket_index != -1:
                corresponding_closing_bracket_index = i
                break

    return first_bracket_index, corresponding_closing_bracket_index


def decode_json(raw):
    start, end = find_matching_bracket(raw)
    raw = raw[start:end+1]
    try:
        return json.loads(raw)
    except:
        try:
            return eval(raw)
        except:
            pass

    raw = raw.replace("'", '"')
    try:
        return json.loads(raw)
    except:
        return eval(raw)


def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

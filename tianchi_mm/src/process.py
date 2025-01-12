import re
import time
import random
import logging
from copy import deepcopy

from src.prompt import *
from src.constants import *
from src.utils import is_intend_instruction, is_image_scene_instruction


logger = logging.getLogger(__name__)


def extract_dialogue(text):
    pattern = r"<用户与客服的对话 START>(.*?)<用户与客服的对话 END>"
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        return match.strip()
    return


def augment_dialogue(client, dialogue, label, max_try=3):    
    # 格式化用戶輸入
    user_prompt = AUGMENT_USER_PROMPT_TEMPLATE.format(
        dialogue=dialogue,
        label=label
    )

    number_of_images = [ele.group() for ele in re.finditer("<image>", dialogue)]
    for _ in range(max_try):
        try:
            time.sleep(1)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": AUGMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    # {"role": "user", "content": f"#### 圖像(base64)\n{encoded_image}"}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            new_dialogue = response.choices[0].message.content
            
            if number_of_images == len([ele.group() for ele in re.finditer("<image>", new_dialogue)]):
                return new_dialogue
            else:
                raise ValueError

        except Exception as e:
            print(e)
    return None


def augment_intend_instruction(client, instruction, label, do_augment_dialogue=False, do_augment_labels=False):
    dialogue = extract_dialogue(instruction)
    
    if do_augment_dialogue:
        new_dialogue = augment_dialogue(client, dialogue, label)
        if new_dialogue:
            if dialogue == new_dialogue:
                logger.info(f"Genrate the same dialogue: {dialogue}")
            dialogue = new_dialogue
    
    classes = deepcopy(INTEND_CLASSES)
    if do_augment_labels:
        random.shuffle(classes)
    class_label_string = "[" + ",".join([f'"{category}"' for category in classes]) + "]"

    return INTEND_DATA_PROMPT_TEMPLATE.format(dialogue=dialogue, class_labels_string=class_label_string)


def augment_image_scene_instruction(instruction, do_augment_labels=False):
    classes = deepcopy(IMAGE_SCENCE_CLASSES)
    if do_augment_labels:
        random.shuffle(classes)
    class_label_string = "[" + ",".join([f'"{category}"' for category in classes]) + "]"
    return IMAGE_SCENE_DATA_PROMPT_TEMPLATE.format(class_labels_string=class_label_string)


def mix_intend_instruction(instruction1, instruction2, do_augment_labels=False):
    dialogue1 = extract_dialogue(instruction1)
    dialogue2 = extract_dialogue(instruction2)

    # shuffle
    classes = deepcopy(INTEND_CLASSES)
    if do_augment_labels:
        random.shuffle(classes)
    class_label_string = "[" + ",".join([f'"{category}"' for category in classes]) + "]"

    return CONCAT_INTEND_DATA_PROMPT_TEMPLATE.format(dialogue1=dialogue1, dialogue2=dialogue2, class_labels_string=class_label_string)


def mix_image_scene_instruction(instruction1, instruction2, do_augment_labels=False):
    classes = deepcopy(IMAGE_SCENCE_CLASSES)
    if do_augment_labels:
        random.shuffle(classes)
    class_label_string = "[" + ",".join([f'"{category}"' for category in classes]) + "]"
    return CONCAT_IMAGE_SCNENE_DATA_PROMPT_TEMPLATE.format(class_labels_string=class_label_string)


def mix_labels(label1, label2):
    return f"{label1},{label2}"


def convert_to_detail_desc_label(instruction, intend_label_to_desc, image_scene_label_to_desc):
    if is_intend_instruction(instruction):
        res = instruction.split("以下是可以参考的分类标签为")[0] + "以下是可以参考的分类标签与对应的描述：\n"
        res += "\n".join(
            [f"+ 标籤：{cls}，描述：{intend_label_to_desc[cls]}" for cls in INTEND_CLASSES]
        )

    elif is_image_scene_instruction(instruction):
        res = instruction.split("以下是可以参考的分类标签,分类标签")[0] + "以下是可以参考的分类标签与对应的描述：\n"
        res += "\n".join(
            [f"+ 标籤：{cls}，描述：{image_scene_label_to_desc[cls]}" for cls in IMAGE_SCENCE_CLASSES]
        )
    else:
        raise ValueError(f"Incorrect instruction: {instruction}")
    return res


def convert_to_cot_image_scene_instruction(instruction):
    instruction = IMAGE_SCENE_CHAIN_OF_THOUGHT_PROMPT_TEMPLATE
    return instruction

def convert_to_cot_intend_instruction(instruction):
    dialogue = extract_dialogue(instruction)
    instruction = INTEND_CHAIN_OF_THOUGHT_PROMPT_TEMPLATE.format(dialogue=dialogue)
    return instruction


def convert_to_standard_image_scene_instruction(instruction, is_sbs=False):
    prompt = IMAGE_SCENE_STANDARD_PROMPT_TEMPLATE
    if is_sbs:
        prompt += "\n\nLet's think step by step"
    else:
        prompt += "\n\n请直接输出分类编号，不要有其他说明。"
    return prompt

def convert_to_standard_intend_instruction(instruction, is_sbs=False):
    dialogue = extract_dialogue(instruction)
    prompt = INTEND_STANDARD_PROMPT_TEMPLATE.format(dialogue=dialogue)
    if is_sbs:
        prompt += "\n\nLet's think step by step"
    else:
        prompt += "\n\n请直接输出分类编号，不要有其他说明。"
    return prompt


def convert_idx_instruction_to_mapping(instruction):
    match = re.findall(r'([a-zA-Z])\.\s*([^\n]+)', instruction)
    if not match:
        raise ValueError("未找到分类标签编号部分，请检查 instruction 格式是否正确。")
    result = {item[0].strip(): item[1].strip() for item in match}

    for idx, name in result.items():
        if name == "各种非淘宝、菜鸟APP的截图":
            result[idx] = "外部APP截图"

    print(result)
    return result
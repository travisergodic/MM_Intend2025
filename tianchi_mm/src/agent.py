import re
import time
import logging

from src.utils import encode_image, decode_json
from src.process import extract_dialogue
from src.prompt import *
from src.constants import *


logger = logging.getLogger(__name__)



class ImageSceneCoTAgentWithLabel:
    def __init__(self, client, model_name, temperature=0.3, retry=2):
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.retry = retry
        
    def do(self, instrction, image_paths, label):
        base64_image = encode_image(image_paths[0])
        for _ in range(self.retry):
            try:
                time.sleep(1)
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system", 
                            "content": IMAGE_SCENE_CHAIN_OF_THOUGHT_AGENT_SYSTEM_PROMPT_WITH_LABEL
                        },
                        {
                            "role": "user", 
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": IMAGE_SCENE_CHAIN_OF_THOUGHT_AGENT_USER_PROMPT_TEMPLATE_WITH_LABEL.format(label=label)
                                }
                            ]
                        }
                    ],
                    temperature=self.temperature,
                )
                text = response.choices[0].message.content
                res = re.search(r"(?<=图像标籤[:：]).*", text)
                if res:
                    extract_label = res.group().strip("。").strip(".")
                
                if extract_label != label:
                    raise ValueError(f"Label {label} and extracted label {extract_label} mismatch")

                logger.info(f"label: {label}, cot: {text}")
                return {"cot": text, "status": "success", "gt": label, "pred": extract_label}
            
            except Exception as e:
                logger.info(e)
        return {"cot": text, "status": "fail", "gt": label, "pred": extract_label}



class IntendCoTAgentWithLabel:
    def __init__(self, client, model_name, temperature=0.3, retry=2):
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.retry = retry
        
    def do(self, instruction, image_paths, label):
        base64_image_list = [encode_image(image_path) for image_path in image_paths]
        user_prompt = INTEND_CHAIN_OF_THOUGHT_AGENT_USER_PROMPT_TEMPLATE_WITH_LABEL.format(
            dialogue=extract_dialogue(instruction), label=label
        )
        user_content = []
        curr_image_idx = 0 
        for segment in re.split(r'(<image>)', user_prompt):
            if segment == "<image>":
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image_list[curr_image_idx]}"
                        }
                    }
                )
                curr_image_idx += 1
            else:
                user_content.append(
                    {"type": "text", "text": segment}
                )

        for _ in range(self.retry):
            try:
                time.sleep(1)
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system", 
                            "content": INTEND_CHAIN_OF_THOUGHT_AGENT_SYSTEM_PROMPT_WITH_LABEL
                        },
                        {
                            "role": "user", 
                            "content": user_content
                        }
                    ],
                    temperature=self.temperature,
                )
                text = response.choices[0].message.content
                res = re.search(r"(?<=用户意图[:：]).*", text)
                if res:
                    extract_label = res.group().strip("。").strip(".")
                
                if extract_label != label:
                    raise ValueError(f"Label {label} and extracted label {extract_label} mismatch")

                logger.info(f"label: {label}, cot: {text}")
                return {"cot": text, "status": "success", "gt": label, "pred": extract_label}
            
            except Exception as e:
                logger.info(e)
        return {"text": text, "status": "fail", "gt": label, "pred": extract_label}
    

class ClassComparisonAgent:
    def __init__(self, client, model_name, temperature=0.3, retry=2):
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.retry = retry

    def do(self, class1_desc, class2_desc, class1_image_paths, class2_image_paths):
        class1_base64_images = [encode_image(image_path) for image_path in class1_image_paths]
        class2_base64_images = [encode_image(image_path) for image_path in class2_image_paths]

        class1_image_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            } for base64_image in class1_base64_images
        ]

        class2_image_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            } for base64_image in class2_base64_images
        ]

        user_prompt =CLASS_COMPARSION_USER_PROMPT_TEMPLATE.format(
            class1_desc=class1_desc, class2_desc=class2_desc
        )

        user_content = []
        user_prompt_segments = re.split(r'(<class1_images>|<class2_images>)', user_prompt)

        for segment in user_prompt_segments:
            if segment == "<class1_images>":
                user_content += class1_image_content
            elif segment == "<class2_images>":
                user_content += class2_image_content
            else:
                user_content.append({"type": "text", "text": segment})

        try:
            time.sleep(1)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": CLASS_COMPARSION_SYSTEM_PROMPT
                    },
                    {
                        "role": "user", 
                        "content": user_content
                    }
                ],
                temperature=self.temperature
            )
            text = response.choices[0].message.content
            res = decode_json(text)
            if ("class1" not in res) or ("class2" not in res):
                raise ValueError()
            return res

        except Exception as e:
            logger.info(f"{e}: {text}")


class ClassSummaryAgent:
    def __init__(self, client, model_name, temperature=0.3, retry=2):
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.retry = retry

    def do(self, class_name, class_desc, image_paths):
        base64_images = [encode_image(image_path) for image_path in image_paths] 
        user_prompt =CLASS_SUMMARY_USER_PROMPT_TEMPLATE.format(
            class_name=class_name, 
            class_desc=class_desc
        )

        image_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            } for base64_image in base64_images
        ]

        user_content = []

        for segment in re.split(r'(<images>)', user_prompt):
            if segment == "<images>":
                user_content += image_content
            else:
                user_content.append({"type": "text", "text": segment})

        try:
            time.sleep(1)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": CLASS_SUMMARY_SYSTEM_PROMPT
                    },
                    {
                        "role": "user", 
                        "content": user_content
                    }
                ],
                temperature=self.temperature
            )
            return response.choices[0].message.content

        except Exception as e:
            logger.info(e)



class IntendSBSAgent:
    def __init__(self, client, model_name, temperature=0.3, retry=3):
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.retry = retry
        self.sbs_prompt = "Rationale:\nLet's think step by step."

    def do(self, instruction, image_paths, label):
        base64_image_list = [encode_image(image_path) for image_path in image_paths]
        answer_number = INTEND_LABEL_TO_NUMBER[label]

        user_prompt = INTEND_SBS_PROMPT_TEMPLATE_WITH_LABEL.format(
            dialogue=extract_dialogue(instruction), answer_number=answer_number, sbs_prompt=self.sbs_prompt
        )
        
        user_content = []
        curr_image_idx = 0 
        for segment in re.split(r'(<image>)', user_prompt):
            if segment == "<image>":
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image_list[curr_image_idx]}"
                        }
                    }
                )
                curr_image_idx += 1
            else:
                user_content.append(
                    {"type": "text", "text": segment}
                )

        for _ in range(self.retry):
            text = None
            try:
                time.sleep(1)
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": user_content}],
                    temperature=self.temperature,
                )
                text = response.choices[0].message.content
                logger.info(f"label: {label}({INTEND_LABEL_TO_NUMBER[label]}), cot: {text}")

                digits = sorted(set(re.findall(r'\d+', text)))
                if (len(digits) == 1) and int(digits[0])==answer_number:
                    return text

            except Exception as e:
                logger.info(e)
        return text


class ImageSceneSBSAgent:
    def __init__(self, client, model_name, temperature=0.3, retry=3):
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.retry = retry
        self.sbs_prompt = "Rationale:\nLet's think step by step."

    def do(self, instruction, image_paths, label):
        base64_image_list = [encode_image(image_path) for image_path in image_paths]
        answer_number = IMAGE_SCENE_LABEL_TO_NUMBER[label]

        user_prompt = IMAGE_SCENE_SBS_PROMPT_TEMPLATE_WITH_LABEL.format(
            answer_number=answer_number, sbs_prompt=self.sbs_prompt
        )
        
        user_content = []
        curr_image_idx = 0 
        for segment in re.split(r'(<image>)', user_prompt):
            if segment == "<image>":
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image_list[curr_image_idx]}"
                        }
                    }
                )
                curr_image_idx += 1
            else:
                user_content.append(
                    {"type": "text", "text": segment}
                )
        
        text = None
        for _ in range(self.retry):
            
            try:
                time.sleep(1)
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": user_content}],
                    temperature=self.temperature,
                )
                text = response.choices[0].message.content
                logger.info(f"label: {label}({IMAGE_SCENE_LABEL_TO_NUMBER[label]}), cot: {text}")

                digits = sorted(set(re.findall(r'\d+', text)))
                if (len(digits) == 1) and int(digits[0])==answer_number:
                    return text

            except Exception as e:
                logger.info(e)
        return text
    

class CriticAgent:
    def __init__(self, client, model_name, temperature=0.3, retry=3):
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.retry = retry

    def do(self, response):
        user_content = GIVE_ANSWER_PROMPT.format(response=response)
        text = None
        for _ in range(self.retry):
            try:
                time.sleep(1)
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": user_content}],
                    temperature=self.temperature,
                )
                text = response.choices[0].message.content

            except Exception as e:
                logger.info(e)

            if text is None:
                continue
            
            digits = sorted(set(re.findall(r'\d+', text)))
            if len(digits) == 1:
                return int(digits[0])
        return None
    


class ImageSceneLabeler:
    def __init__(self, client, model_name, temperature=0.3, retry=3):
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.retry = retry

    def do(self, instruction, image_paths):
        base64_image_list = [encode_image(image_path) for image_path in image_paths]
        
        user_content = []
        curr_image_idx = 0 
        for segment in re.split(r'(<image>)', IMAGE_SCENE_LABELER_USER_PROMPT):
            if segment == "<image>":
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image_list[curr_image_idx]}"
                        }
                    }
                )
                curr_image_idx += 1
            else:
                user_content.append(
                    {"type": "text", "text": segment}
                )

        for _ in range(self.retry):
            
            try:
                time.sleep(1)
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": IMAGE_SCENE_LABELER_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=self.temperature,
                )
                text = response.choices[0].message.content
                result = decode_json(text)
                if all([k in result for k in ("pred", "cot")]):
                    return result

            except Exception as e:
                logger.info(e)
                print(text)
        return None
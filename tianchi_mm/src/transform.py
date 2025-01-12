import logging

from src.utils import is_image_scene_instruction, is_intend_instruction
from src.process import extract_dialogue, convert_idx_instruction_to_mapping
from src.prompt import *
from src.registry import TRANSFORM


logger = logging.getLogger(__name__)


def get_reverae_mapping(a_dict):
    vals = list(a_dict.values())
    if len(vals) != len(set(vals)):
        raise ValueError("Duplicate values!") 
    return {v:k for k, v in a_dict.items()}


@TRANSFORM.register("identity")
class IdentityInstructionTrasnform:
    def __init__(self, relabel_dict=None, **kwargs):
        self.relabel_dict = relabel_dict or dict()
        self.relabel_cnt = 0 

    def transform(self, record):
        if self.relabel_dict and (record["id"] in self.relabel_dict):
            record["output"] = self.relabel_dict[record["output"]]
            self.relabel_cnt += 1
        return record

    def name_to_label(self, instrcution, name):
        return name
    
    def label_to_name(self, instruction, label):
        return label
    

@TRANSFORM.register("identity_system")
class IdentityInstructionTrasnform:
    def __init__(self, relabel_dict=None, **kwargs):
        self.relabel_dict = relabel_dict or dict()
        self.relabel_cnt = 0 

    def transform(self, record):
        if self.relabel_dict and (record["id"] in self.relabel_dict):
            record["output"] = self.relabel_dict[record["output"]]
            self.relabel_cnt += 1

        instruction = record["instruction"]
        
        if is_intend_instruction(instruction):
            instruction = f"<用户与客服的对话 START>{extract_dialogue(instruction)}<用户与客服的对话 END>"
            system = "你是一个电商客服专家，请根据用户与客服的多轮对话判断用户的意图分类标签。请直接只输出分类标签结果，不需要其他多余的话。以下是可以参考的分类标签为：[\"反馈密封性不好\",\"是否好用\",\"是否会生锈\",\"排水方式\",\"包装区别\",\"发货数量\",\"反馈用后症状\",\"商品材质\",\"功效功能\",\"是否易褪色\",\"适用季节\",\"能否调光\",\"版本款型区别\",\"单品推荐\",\"用法用量\",\"控制方式\",\"上市时间\",\"商品规格\",\"信号情况\",\"养护方法\",\"套装推荐\",\"何时上货\",\"气泡\"]"
        
        elif is_image_scene_instruction(instruction):
            instruction = "Picture 1: <image>"
            system = "你是一个电商领域识图专家,可以理解消费者上传的软件截图或实物拍摄图。现在,请你对消费者上传的图片进行分类。你只需要回答图片分类结果,不需要其他多余的话。以下是可以参考的分类标签,分类标签:[\"实物拍摄(含售后)\",\"商品分类选项\",\"商品头图\",\"商品详情页截图\",\"下单过程中出现异常（显示购买失败浮窗）\",\"订单详情页面\",\"支付页面\",\"消费者与客服聊天页面\",\"评论区截图页面\",\"物流页面-物流列表页面\",\"物流页面-物流跟踪页面\",\"物流页面-物流异常页面\",\"退款页面\",\"退货页面\",\"换货页面\",\"购物车页面\",\"店铺页面\",\"活动页面\",\"优惠券领取页面\",\"账单/账户页面\",\"个人信息页面\",\"投诉举报页面\",\"平台介入页面\",\"外部APP截图\",\"其他类别图片\"]"

        record["instruction"] = instruction
        record["system"] = system
        return record

    def name_to_label(self, instrcution, name):
        return name
    
    def label_to_name(self, instruction, label):
        return label
    

@TRANSFORM.register("sharegpt")
class ShareGPTInstrcutionTransform:
    def __init__(self, idx_to_reason=None, relabel_dict=None, **kwargs):
        self.idx_to_reason = idx_to_reason or dict()
        self.relabel_dict = relabel_dict
        self.relabel_cnt = 0 

    def transform(self, record):
        if self.relabel_dict and (record["id"] in self.relabel_dict):
            record["output"] = self.relabel_dict[record["output"]]
            self.relabel_cnt += 1

        instruction = record["instruction"]
        output = record["output"]
        image = record["image"]
        idx = record["id"]

        conversations = [
            {"from": "human", "value": instruction},
            {"from": "gpt", "value": output}
        ]
        if idx in self.idx_to_reason:
            reason = self.idx_to_reason[idx]
            conversations += [
                {"from": "human", "value": f"原因为何？"},
                {"from": "gpt", "value": reason}
            ]
        return {
            "conversations": conversations,
            "image": image
        }
    
    def name_to_label(self, instrcution, name):
        return name
    
    def label_to_name(self, instruction, label):
        return label
    

@TRANSFORM.register("sharegpt_system")
class ShareGPTInstrcutionTransform:
    def __init__(self, idx_to_reason=None, relabel_dict=None, **kwargs):
        self.idx_to_reason = idx_to_reason or dict()
        self.relabel_dict = relabel_dict
        self.relabel_cnt = 0 

    def transform(self, record):
        if self.relabel_dict and (record["id"] in self.relabel_dict):
            record["output"] = self.relabel_dict[record["output"]]
            self.relabel_cnt += 1

        instruction = record["instruction"]
        image = record["image"]
        idx = record["id"]

        if is_intend_instruction(instruction):
            instruction = f"<用户与客服的对话 START>{extract_dialogue(instruction)}<用户与客服的对话 END>"
            system = "你是一个电商客服专家，请根据用户与客服的多轮对话判断用户的意图分类标签。请直接只输出分类标签结果，不需要其他多余的话。以下是可以参考的分类标签为：[\"反馈密封性不好\",\"是否好用\",\"是否会生锈\",\"排水方式\",\"包装区别\",\"发货数量\",\"反馈用后症状\",\"商品材质\",\"功效功能\",\"是否易褪色\",\"适用季节\",\"能否调光\",\"版本款型区别\",\"单品推荐\",\"用法用量\",\"控制方式\",\"上市时间\",\"商品规格\",\"信号情况\",\"养护方法\",\"套装推荐\",\"何时上货\",\"气泡\"]"
        
        elif is_image_scene_instruction(instruction):
            instruction = "Picture 1: <image>"
            system = "你是一个电商领域识图专家,可以理解消费者上传的软件截图或实物拍摄图。现在,请你对消费者上传的图片进行分类。你只需要回答图片分类结果,不需要其他多余的话。以下是可以参考的分类标签,分类标签:[\"实物拍摄(含售后)\",\"商品分类选项\",\"商品头图\",\"商品详情页截图\",\"下单过程中出现异常（显示购买失败浮窗）\",\"订单详情页面\",\"支付页面\",\"消费者与客服聊天页面\",\"评论区截图页面\",\"物流页面-物流列表页面\",\"物流页面-物流跟踪页面\",\"物流页面-物流异常页面\",\"退款页面\",\"退货页面\",\"换货页面\",\"购物车页面\",\"店铺页面\",\"活动页面\",\"优惠券领取页面\",\"账单/账户页面\",\"个人信息页面\",\"投诉举报页面\",\"平台介入页面\",\"外部APP截图\",\"其他类别图片\"]"


        conversations = [
            {"from": "human", "value": instruction}
        ]

        if "output" in record:
            output = record["output"]
            conversations += [{"from": "gpt", "value": output}]
            if idx in self.idx_to_reason:
                reason = self.idx_to_reason[idx]
                conversations += [
                    {"from": "human", "value": f"原因为何？"},
                    {"from": "gpt", "value": reason}
                ]

        return {
            "conversations": conversations,
            "image": image,
            "system": system
        }
    
    def name_to_label(self, instrcution, name):
        return name
    
    def label_to_name(self, instruction, label):
        return label
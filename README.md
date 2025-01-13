# [阿里天池-WWW2025 多模态对话系统意图识别挑战赛](https://tianchi.aliyun.com/competition/entrance/532277?spm=a2c22.29524729.0.0.1eed3b74lsL7m2)

## 团队信息
1. **团队名称**：baseline
2. **团队成员**：胡太维、朱成斌
3. **初赛排名**：82 / 1643
4. **最终成绩**：84.94 (基准分数：78.82)

## 环境
1. **Python 版本**: 3.10.12
2. **GPU**: L40S x 2 (45G x 2 = 90GB)
3. **CUDA 版本**: 12.4

## 工具
1. **生成分类原因**
   ```bash
   python tools/reason_generation.py --backend ${backend} \
                                     --api_key ${api_key} \
                                     --model_name ${model_name} \
                                     --json_file ${json_file} \
                                     --image_root ${image_root} \
                                     --save_path ${save_path}                  
   ```
   + **backend** (必填): 所使用的大模型平台，目前支援 `openai`, `ali`, `llama`, `nvidia` 选项。以下是这些平台的模型哭连结：
     + [openai](https://platform.openai.com/docs/models)
     + [ali](https://bailian.console.aliyun.com/?spm=5176.29619931.J__Z58Z6CX7MY__Ll8p1ZOR.1.136959fc6GbJTE#/model-market)
     + [llama](https://console.llamaapi.com/zh/384a2ea8-4f96-4d89-82f9-ec3bb38ee717/credits)
     + [nvidia](https://docs.api.nvidia.com/nim/reference/llm-apis)
     + [gemini](https://ai.google.dev/gemini-api/docs/models/gemini?hl=zh-tw)
     + [claude](https://docs.anthropic.com/en/docs/about-claude/models)
   + **api_key** (必填): API 钥匙。
   + **model_name** (必填): 大模型名称，例如 `gpt-4o`, `qwen-vl-max`。
   + **json_file** (必填): 原始档案，每笔纪录必须包含 `instruction`, `output`, `image` 字段。
   + **image** (必填): 图像所在目录。
   + **save_path** (必填): 储存路径 (JSON 格式)。

2. **提交格式**
   ```bash
   python tools/submit.py --raw_file ${raw_file} \
                          --predict_file ${predict_file} \
                          --max_image_length ${max_image_length} \
                          --save_dir ${save_path} 
   ```
   + **raw_file** (必填): 原始未转换的档案路径 (`test1.json`)。
   + **predict_file** (必填): 大模型预测结果的档案。
   + **max_image_length** (必填): 模型推理阶段，单笔数据可容忍的最大图像数量，此数值必须与 `examples/predict_*.yaml` 的 `max_image_length` 一致。
   + **save_dir** (必填): 最终提交档案的储存目录。

3. **数据前处理**：将用户意图数据的图像进行缩放，确保最大边不超过 616。    
   ```bash
   python tools/preprocess.py --raw_file ${raw_file} \
                              --src ${src} \
                              --dst ${dst} 
   ```
   + **raw_file** (必填): 原始未转换的档案路径 (`train.json`, `test1.json`)。
   + **src** (必填): 原始图像的储存目录。
   + **dst** (必填): 缩放后图像的储存目录。

## Llama-Factory 配置 yaml 档新增参数
1. `image_dir`: 图像储存的路径，以 `list of directory` 形式呈现，读取图像时，会依序搜索列表内的目录是否有对应的图像名称，直至搜索到为止。
2. `max_image_length`: 单笔样本可容忍的最大图像数量。
3. `max_pixel`: 可容忍图像最大的总像素，若图像总像素超过此设定值，会对图像进行（维持宽高比）缩放，确保缩放后图像总像素小于此设定值。若设定此参数，`image_resolution` 參數将不起作用。
4. `grayscale_prob`: 随机灰阶概率值。
5. `brightness_prob`: 随机亮度概率值。
6. `brightness_delta`: 随机亮度的区间介于 `(1-brightness_delta, 1+brightness_delta)`。
7. `do_jpeg_compression`: 是否进行 JPEG compression 数据增强。
8. `max_quality`: JPEG compression 的 max_quality。
9. `min_quality`: JPEG compression 的 min_quality。

## 最终方案流程
1. **环境安装**
   ```bash
   cd ${project_dir}
   bash commands/install.sh
   ```
2. **数据下载**：请创建 `tianchi_data/` 目录，并将训练、测试数据放置于目录中，目录结构如下
    ```
    mm_intend/
    ├── tianchi_data/
    │   ├── train/
    │   │   ├── images/
    │   │   └── train.json
    │   ├── test1/
    │   │   ├── images/
    │   │   └── test1.json
    ```
2. **前处理**
   ```bash
   cd ${project_dir}/tianchi_mm/
   mkdir ../tianchi_data/train/intend_images_616
   mkdir ../tianchi_data/test1/intend_images_616

   python tools/preprocess.py --raw_file ../tianchi_data/train/train.json \
                              --src ../tianchi_data/train/images \
                              --dst ../tianchi_data/train/intend_images_616

   python tools/preprocess.py --raw_file ../tianchi_data/test1/test1.json \
                              --src ../tianchi_data/test1/images \
                              --dst ../tianchi_data/test1/intend_images_616
   ```
3. **模型训练**：请根据环境调整 `per_device_train_batch_size`, `gradient_accumulation_steps` 使得最终的 `batch_size` 为 12，预设为双卡环境。为确保训练不会出现 O.O.M 情况，可适当调整 `max_image_length` 的值。
   ```bash
   cd ${project_dir}
   FORCE_TORCHRUN=1 WANDB_DISABLED=true llamafactory-cli train examples/qwen2_vl_full_sft_final.yaml
   ```

4. **模型推理**
   ```bash
   cd ${project_dir}
   WANDB_DISABLED=true llamafactory-cli train examples/predict_full_final.yaml
   ```

5. **转换为提交格式**: 提交档案将储存在 `${project_dir}/generated_predictions.csv`。
   ```bash
   cd ${project_dir}/tianchi_mm/
   python tools/submit.py --raw_file ../tianchi_data/test1/test1.json \
                          --predict_file saves/qwen2_vl-7b/full/freeze_25_b12_900k-infer-225_max5/generated_predictions.jsonl \
                          --max_image_length 5 \
                          --save_dir ..
   ```

## 最佳方案权重
1. **下载模型权重**
   ```bash
   cd ${project_dir}
   wget -O part_aa https://travisergodic-ai-models.oss-cn-shanghai.aliyuncs.com/mm_intend/part_aa
   wget -O part_ab https://travisergodic-ai-models.oss-cn-shanghai.aliyuncs.com/mm_intend/part_ab
   wget -O part_ac https://travisergodic-ai-models.oss-cn-shanghai.aliyuncs.com/mm_intend/part_ac
   wget -O part_ad https://travisergodic-ai-models.oss-cn-shanghai.aliyuncs.com/mm_intend/part_ad
   cat part_aa part_ab part_ac part_ad > checkpoint-225_partial.tar.gz
   tar -xzvf checkpoint-225_partial.tar.gz -C .
   ```
4. **模型推理**：请先将 `examples/predict_full_final.yaml` 的变数 `model_name_or_path` 改为 `checkpoint-225_partial/`
   ```bash
   cd ${project_dir}
   WANDB_DISABLED=true llamafactory-cli train examples/predict_full_final.yaml
   ```
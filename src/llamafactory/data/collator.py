# Copyright 2024 OpenAccess AI Collective and the LlamaFactory team.
#
# This code is inspired by the OpenAccess AI Collective's axolotl library.
# https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/monkeypatch/utils.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Sequence


import torch
from transformers import DataCollatorForSeq2Seq
from transformers.models.llava_onevision import image_processing_llava_onevision


if TYPE_CHECKING:
   from transformers import ProcessorMixin
   from .template import Template


def calculate_consecutive_lengths(input_ids, image_token_marker):
   """
   Calculate the lengths of all consecutive image token segments in the input_id list.


   :param input_ids: List of input IDs.
   :param is_image_token: Function to determine if an ID is an image token.
   :return: List of lengths of consecutive image token sequences.
   """
   lengths = []
   current_length = 0
   for token_id in input_ids:
       if token_id == image_token_marker:
           current_length += 1
       else:
           if current_length > 0:
               lengths.append(current_length)
               current_length = 0

   if current_length > 0:
       lengths.append(current_length)
       
   return lengths


def adjust_token_list(token_ids, labels, attention_mask, image_token_marker, new_image_token_lengths):
   """
   Adjust the token id list, labels, and attention mask to fit new image token lengths.


   Parameters:
       token_ids (list of int): The original list of token ids.
       labels (list of int): The original list of labels.
       attention_mask (list of int): The original attention mask.
       image_token_marker (int): The marker representing the start of image tokens (e.g., 151655).
       new_image_token_lengths (list of int): The list of new lengths for image tokens.


   Returns:
       tuple: Adjusted (token_ids, labels, attention_mask).
   """
   adjusted_tokens = []
   adjusted_labels = []
   adjusted_attention_mask = []
   new_image_token_idx = 0  # Index for new_image_token_lengths
   i = 0

   while i < len(token_ids):
       if token_ids[i] == image_token_marker:
           # Find the end of the current image token sequence in the original list
           j = i
           while j < len(token_ids) and token_ids[j] == image_token_marker:
               j += 1

           # Replace the original image token sequence with the new one
           if new_image_token_idx < len(new_image_token_lengths):
               new_length = new_image_token_lengths[new_image_token_idx]
               adjusted_tokens.extend([image_token_marker] * new_length)
               adjusted_labels.extend([-100] * new_length)
               adjusted_attention_mask.extend([1] * new_length)
               new_image_token_idx += 1
           else:
               raise ValueError("New image token lengths list is shorter than the number of image token sequences in the original list.")


           # Move index to the end of the current image token sequence
           i = j
       else:
           # Non-image token, directly append to the adjusted lists
           adjusted_tokens.append(token_ids[i])
           adjusted_labels.append(labels[i])
           adjusted_attention_mask.append(attention_mask[i])
           i += 1

   if new_image_token_idx < len(new_image_token_lengths):
       raise ValueError("New image token lengths list is longer than the number of image token sequences in the original list.")
   return adjusted_tokens, adjusted_labels, adjusted_attention_mask



def prepare_4d_attention_mask(attention_mask_with_indices: "torch.Tensor", dtype: "torch.dtype") -> "torch.Tensor":
   r"""
   Expands the attention mask with indices from (batch_size, seq_len) to (batch_size, 1, seq_len, seq_len),
   while handles packed sequences and transforms the mask to lower triangular form to prevent future peeking.
   e.g. 
   python
   # input
   [[1, 1, 2, 2, 2, 0]]
   # output
   [
       [
           [
               [o, x, x, x, x, x],
               [o, o, x, x, x, x],
               [x, x, o, x, x, x],
               [x, x, o, o, x, x],
               [x, x, o, o, o, x],
               [x, x, x, x, x, x],
           ]
       ]
   ]
   where o equals to `0.0`, x equals to `min_dtype`.
   """
   bsz, seq_len = attention_mask_with_indices.size()
   min_dtype = torch.finfo(dtype).min
   expanded_mask = attention_mask_with_indices[:, None, None, :].expand(bsz, 1, seq_len, seq_len)
   # Create a binary mask from the original mask where zeros remain zeros and all other values are set to one
   padding_mask = torch.where(expanded_mask != 0, 1, 0)
   # Create a block-diagonal mask.
   attention_mask_4d = torch.eq(expanded_mask, expanded_mask.transpose(-1, -2)).int() * padding_mask
   # Use the lower triangular mask to zero out the upper triangular part
   attention_mask_4d *= torch.tril(torch.ones((seq_len, seq_len), dtype=torch.long))
   # Invert the attention mask.
   attention_mask_4d = torch.where(attention_mask_4d != 0, torch.tensor(0, dtype=dtype), min_dtype)
   return attention_mask_4d


@dataclass
class MultiModalDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
   r"""
   Data collator that supports VLMs.


   Features should contain input_ids, attention_mask, labels and images.
   """


   template: Optional["Template"] = None
   processor: Optional["ProcessorMixin"] = None


   def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
       batch_images, batch_videos, batch_imglens, batch_vidlens, batch_seqlens = [], [], [], [], []
       for feature in features:
           images = feature.pop("images", None) or []
           videos = feature.pop("videos", None) or []
           batch_images.extend(images)
           batch_videos.extend(videos)
           batch_imglens.append(len(images))
           batch_vidlens.append(len(videos))
           batch_seqlens.append(len(feature["input_ids"]))


       mm_inputs = self.template.mm_plugin.get_mm_inputs(
           batch_images, batch_videos, batch_imglens, batch_vidlens, batch_seqlens, self.processor
       )
       if "token_type_ids" in mm_inputs:
           token_type_ids = mm_inputs.pop("token_type_ids")
           for i, feature in enumerate(features):
               feature["token_type_ids"] = token_type_ids[i]
      
       if "image_grid_thw" in mm_inputs:
           image_grid_thw = mm_inputs["image_grid_thw"]
           image_idx = 0

           for feature in features:
               input_ids = feature["input_ids"]
               attention_mask = feature["attention_mask"]
               labels = feature["labels"]
            
               image_token_length_list = calculate_consecutive_lengths(input_ids, image_token_marker=151655)
                
               new_image_token_length_list = []
               for image_token_length in image_token_length_list:
                   new_image_token_length = image_grid_thw[image_idx].prod().item() // 4  # image_grid_thw --> image_token_length
                   new_image_token_length_list.append(new_image_token_length)
                   image_idx += 1
                   print("old image token length:", image_token_length)
                   print("new image token length:", new_image_token_length)
                    
               new_input_ids, new_labels, new_attention_mask = adjust_token_list(
                   input_ids, 
                   labels, 
                   attention_mask, 
                   image_token_marker=151655, 
                   new_image_token_lengths=new_image_token_length_list
               )
               feature["input_ids"] = new_input_ids
               feature["labels"] = new_labels 
               feature["attention_mask"] = new_attention_mask

       features: Dict[str, "torch.Tensor"] = super().__call__(features)
       features.update(mm_inputs)
       if isinstance(features.get("pixel_values"), list):  # for pixtral inputs
           features = features.data  # use default_collate() instead of BatchEncoding.to()
       return features



@dataclass
class SFTDataCollatorWith4DAttentionMask(MultiModalDataCollatorForSeq2Seq):
   r"""
   Data collator for 4d attention mask.
   """
   block_diag_attn: bool = False
   attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager"
   compute_dtype: "torch.dtype" = torch.float32


   def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
       features = super().__call__(features)
       if self.block_diag_attn and self.attn_implementation != "flash_attention_2":
           features["attention_mask"] = prepare_4d_attention_mask(features["attention_mask"], self.compute_dtype)
       return features


@dataclass
class PairwiseDataCollatorWithPadding(MultiModalDataCollatorForSeq2Seq):
   r"""
   Data collator for pairwise data.
   """
   def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
       r"""
       Pads batched data to the longest sequence in the batch.


       We generate 2 * n examples where the first n examples represent chosen examples and
       the last n examples represent rejected examples.
       """
       concatenated_features = []
       for key in ("chosen", "rejected"):
           for feature in features:
               target_feature = {
                   "input_ids": feature[f"{key}_input_ids"],
                   "attention_mask": feature[f"{key}_attention_mask"],
                   "labels": feature[f"{key}_labels"],
                   "images": feature["images"],
                   "videos": feature["videos"],
               }
               concatenated_features.append(target_feature)
       return super().__call__(concatenated_features)




@dataclass
class KTODataCollatorWithPadding(MultiModalDataCollatorForSeq2Seq):
   r"""
   Data collator for KTO data.
   """
   def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
       target_features = []
       kl_features = []
       kto_tags = []
       for feature in features:
           target_feature = {
               "input_ids": feature["input_ids"],
               "attention_mask": feature["attention_mask"],
               "labels": feature["labels"],
               "images": feature["images"],
               "videos": feature["videos"],
           }
           kl_feature = {
               "input_ids": feature["kl_input_ids"],
               "attention_mask": feature["kl_attention_mask"],
               "labels": feature["kl_labels"],
               "images": feature["images"],
               "videos": feature["videos"],
           }
           target_features.append(target_feature)
           kl_features.append(kl_feature)
           kto_tags.append(feature["kto_tags"])


       batch = super().__call__(target_features)
       kl_batch = super().__call__(kl_features)
       batch["kl_input_ids"] = kl_batch["input_ids"]
       batch["kl_attention_mask"] = kl_batch["attention_mask"]
       batch["kl_labels"] = kl_batch["labels"]
       if "token_type_ids" in kl_batch:
           batch["kl_token_type_ids"] = kl_batch["token_type_ids"]

       batch["kto_tags"] = torch.tensor(kto_tags)
       return batch
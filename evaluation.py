"""
This script contains functions for CLIP evaluation
- top@k
"""

import os
import json
import argparse
import numpy as np
import torch
import imageio.v3 as imageio
from tqdm.auto import tqdm
from typing import Optional, Union, Literal
from torch.utils.data import DataLoader

from transformers import CLIPProcessor, CLIPModel
from peft import PeftConfig, PeftModel

from data.dataloader import CLIPDataset

class Evaluator:
    
    def __init__(
        self,
        clip_processor: CLIPProcessor,
        clip_model: CLIPModel,
        # caption_list: list[str],
        backend_image_features: Optional[torch.Tensor] = None,
        backend_text_features: Optional[torch.Tensor] = None,
    ):
        self.processor = clip_processor
        self.model = clip_model
        # self.captions = caption_list

        self.device = self.model.device
        self.dtype = self.model.dtype

        self.backend_image_features = backend_image_features
        self.backend_text_features = backend_text_features

    def extract_image_features(
        self,
        images = None,
        pixel_values: Optional[torch.FloatTensor] = None,
    ):
        if images is None and pixel_values is None:
            if self.backend_image_features is None:
                raise ValueError("`backend_image_features` is None, either `images` or `pixel_values` should be given!")
            else:
                image_features = self.backend_image_features
        else:
            if images is not None:
                if pixel_values is not None:
                    print("`images` are given, ignore given `pixel_values`.")
                pixel_values = self.processor.image_processor(images, return_tensors='pt')
                
            image_features = self.model.get_image_features(
                pixel_values.to(self.device, dtype=self.dtype),
            )

        return image_features
    
    def extract_text_features(
        self,
        captions = None,
        text_encoding: Optional[dict[str, torch.LongTensor]] = None
    ):
        if captions is None and text_encoding is None:
            if self.backend_text_features is None:
                raise ValueError("`backend_text_features` is None, either `captions` or `text_encoding` should be given!")
            else:
                text_features = self.backend_text_features
        else:
            if captions is not None:
                if text_encoding is not None:
                    print("`captions` are given, ignore given `text_encoding`.")
                text_encoding = self.processor.tokenizer(
                    captions, 
                    return_tensors='pt',
                    padding='max_length',
                    truncation='longest_first'
                )

            text_features = self.model.get_text_features(
                text_encoding['input_ids'].to(self.device),
                text_encoding['attention_mask'].to(self.device),
            )

        return text_features

        
    def compute_logits(
        self, 
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ):
        """ 
        Returns:
            logits_per_image (`torch.Tensor`):
                The logits on every given captions for each image.
            logits_per_text (`torch.Tensor`):
                The logits on every given images for each caption
        """

        logits_per_image = torch.matmul(image_features, text_features.t())
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text
    

    def top_k(self, logits, k: int = 10, return_logits: bool = False):
        # Sort the logits
        logits_sorted, indices = torch.sort(logits, dim=-1, descending=True)
        #Choose the first k indices
        logits_topk = logits_sorted[..., :k]
        indices_topk = indices[..., :k]
        if return_logits:
            return indices_topk, logits_topk
        else:
            return indices_topk

    def hit_ratio_k(
        self, 
        logits: torch.Tensor, 
        gt_labels: Union[list[int], torch.LongTensor], 
        ks: Union[list[int], torch.LongTensor]
    ):
        """
        Args:
            logits (`torch.Tensor`):
                logits of each sample on all labels, should be of size (n_samples, n_labels).
            gt_labels (`list[int]` or `torch.LongTensor`):
                ground truth labels for each sample, a list of size (n_samples,)
            ks (`list[int]` or `torch.LongTensor`):
                the desired k in hit ratio @ k. If multiple ks are given, they should be in ascending order

        Returns:
            hit_ratio_k
        """
        device = logits.device
        
        ks = torch.tensor(ks).long().to(device)
        if not isinstance(gt_labels, torch.Tensor):
            gt_labels = torch.tensor(gt_labels)
        gt_labels = gt_labels.long().to(device)

        k_max = ks.max()
        assert k_max == ks[-1], "The given ks should be in ascending order"

        indices_topk = self.top_k(logits, k_max)

        hit_counter = torch.zeros_like(ks)

        for il, label in enumerate(gt_labels):  # for each sample and its gt label
            for ik, k in enumerate(ks):
                last_k = 0 if ik == 0 else ks[ik-1]
                if label in indices_topk[il, last_k:k]:   # found it
                    hit_counter[ik:] += 1   # all the counter corresponding to the current and bigger k plus 1
                    break
        
        hit_ratio_k = hit_counter.float() / len(gt_labels)

        return hit_ratio_k


    def evaluate(
        self, 
        gt_labels: Union[list[int], torch.LongTensor], 
        ks: Union[list[int], torch.LongTensor],
        images = None,
        captions = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        text_encoding: Optional[dict[str, torch.LongTensor]] = None,
        use_backend_image_features: bool = False,
        use_backend_text_features: bool = False,
        replace_backend_image_features: bool = False,
        replace_backend_text_features: bool = False,
        for_image: bool = True,
    ):
        """
        Main funtion for evaluation.

        Args:
            gt_labels (`list[int]` or `torch.LongTensor`):
                ground truth labels for each sample, a list of size (n_samples,)
            ks (`list[int]` or `torch.LongTensor`):
                the desired k in hit ratio @ k. If multiple ks are given, they should be in ascending order
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width. And the values should be scaled to (0, 1).
            captions (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            pixel_values (`torch.FloatTensor`):
                image features preprocessed by self.processor.image_processor
            input_ids|attention_mask (`torch.LongTensor`):
                text features preprocessed by self.processor.tokenizer
            for_image (bool):
                whether a sample is an image or a caption, defautly True.

        Returns:
            hit_ratio_k (torch.FloatTensor):
                The hit ratio @ k on the given data for each k.
        """
        # Extract features
        if self.backend_image_features is None and images is None and pixel_values is None:
            raise ValueError("`backend_image_features` is None, and no images or pixel values given!")
        else:
            if use_backend_image_features:
                if self.backend_image_features is None:
                    if images is not None or pixel_values is not None:
                        print("`backend_image_features` is None, use the features of given images or pixel values instead.")
                        image_features = self.extract_image_features(images, pixel_values)
                        # self.backend_image_features = image_features
                else:
                    image_features = self.backend_image_features
            else:
                image_features = self.extract_image_features(images, pixel_values)
        
        if replace_backend_image_features:
            self.backend_image_features = image_features
            
        if self.backend_text_features is None and captions is None and text_encoding is None:
            raise ValueError("`backend_text_features` is None, and no captions or text_encoding given!")
        else:
            if use_backend_text_features:
                if self.backend_text_features is None:
                    if captions is not None or text_encoding is not None:
                        print("`backend_text_features` is None, use the features of given captions or text_encodings instead.")
                        text_features = self.extract_text_features(captions, text_encoding)
                        # self.backend_image_features = image_features
                else:
                    text_features = self.backend_text_features
            else:
                text_features = self.extract_text_features(captions, text_encoding)
        
        if replace_backend_text_features:
            self.backend_text_features = text_features

        # Compute logits
        logits_per_image, logits_per_text = self.compute_logits(
            image_features, text_features
        )
        logits = logits_per_image if for_image else logits_per_text

        # Compute hit ration @ k
        hit_ratio_k = self.hit_ratio_k(logits, gt_labels, ks)

        return hit_ratio_k
        
 


def main(
    data_root: str = './dataset',
    pretrained_clip_path: str = None,
    pretrained_lora_path: str = None,
    batch_size: int = 1000,
    ks: Union[int, list[int]] = 1,
    retrieval_type: Literal['image', 'text'] = 'image',
):

    dtype = torch.float16   # Use fp16 for evaluation
    device = torch.device('cuda:0') # Use single GPU for evaluation

    if pretrained_lora_path is not None:    # LoRA first
        peft_config = PeftConfig.from_pretrained(pretrained_lora_path)
        pretrained_clip_path = peft_config.base_model_name_or_path
    
    # Load pretrained CLIP model
    processor = CLIPProcessor.from_pretrained(pretrained_clip_path)
    model = CLIPModel.from_pretrained(pretrained_clip_path)

    if pretrained_lora_path is not None:
        model = PeftModel.from_pretrained(model, pretrained_lora_path, is_trainable=False)

    model.to(device, dtype=dtype)
    model.requires_grad_(False)
    
    evaluator = Evaluator(processor, model,)
    
    # Load validation dataset
    eval_dset = CLIPDataset(root=data_root, split='val', processor=processor)
    if retrieval_type == 'text':
        print("Performing text retrieval, load all text features of the dataset as backend.")
        all_captions = eval_dset.captions
        text_features = evaluator.extract_text_features(all_captions)
        evaluator.backend_text_features = text_features
    elif retrieval_type == 'image':
        print("Performing image retrieval, load all image features of the dataset as backend.")
        pixel_values = []
        for i in range(len(eval_dset)):
            pixel_value = eval_dset[i]['pixel_values']
            pixel_values.append(pixel_value)
        pixel_values = torch.stack(pixel_values)
        image_features = evaluator.extract_image_features(pixel_values=pixel_values)
        evaluator.backend_image_features = image_features
    else:
        raise ValueError(f"Unsupported retrieval type {retrieval_type}.")

    eval_dloader = DataLoader(
        eval_dset, batch_size=batch_size,
    )

    # Evaluate
    indices = torch.arange(len(eval_dset))
    
    for i, batch in enumerate(tqdm(eval_dloader, desc='Evaluation')):
        if (len(eval_dset) - batch_size*i) >= batch_size:
            gt_labels = indices[batch_size*i : batch_size*(i+1)]
        else:
            gt_labels = indices[batch_size*i]
        pixel_values = batch['pixel_values']
        text_encoding = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask']
        }
        hit_ratio_k = evaluator.evaluate(
            pixel_values=pixel_values,
            text_encoding=text_encoding,
            gt_labels=gt_labels,
            ks=ks,
            use_backend_text_features=retrieval_type=='text',
            use_backend_image_features=retrieval_type=='image',
            for_image=retrieval_type=='text',
        )
        hit_ratio_k *= len(gt_labels)

        hit_ratio_k_ = hit_ratio_k if i == 0 else hit_ratio_k_ + hit_ratio_k

    hit_ratio_k = hit_ratio_k_ / len(eval_dset)

    # for i, k in enumerate(ks):
    results = {f'hit@{k}': hit_ratio_k[i].item() for i, k in enumerate(ks)}
    print(results)

    # save results
    json_str = json.dumps(results, indent=4)
    with open(os.path.join(pretrained_lora_path or pretrained_clip_path, 'eval_result.json'), 'w') as f:
        f.write(json_str)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLIP evaluation')
    parser.add_argument("--data_root", type=str, default='./dataset')
    parser.add_argument("--pretrained_clip_path", type=str, default="openai/clip-vit-base-patch16")
    parser.add_argument("--pretrained_lora_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1000,)
    parser.add_argument("--ks", type=int, nargs='+')
    parser.add_argument("--retrieval_type", type=str, default='image',
                        help="The target of retrieval, `image` or `text`.")
    args = parser.parse_args()

    main(**vars(args))



    
    
"""
This script contains functions for CLIP evaluation
- top@k
"""


import numpy as np
import torch

from transformers import CLIPProcessor, CLIPModel

from data.dataloader import CLIPDataset

data_root = './dataset'


class Evaluator:
    
    def __init__(
        self,
        clip_processor: CLIPProcessor,
        clip_model: CLIPModel,
        # caption_list: list[str],
    ):
        self.processor = clip_processor
        self.model = clip_model
        # self.captions = caption_list
        
    def compute_logits(self, images, captions):
        """
        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width. And the values should be scaled to (0, 1).
            captions (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
        
        Returns:
            logits_per_image (`torch.Tensor`):
                The logits on every given captions for each image.
            logits_per_text (`torch.Tensor`):
                The logits on every given images for each caption
        """
        device = self.model.device
        dtype = self.model.dtype
        
        inputs = self.processor(
            text=captions,
            images=images,
            return_tensors='pt',
            padding='max_length',
            truncation='longest_first'
        )
        
        image_features = self.model.get_image_features(
            inputs['pixel_values'].to(device, dtype=dtype),
        )
        text_features = self.model.get_text_features(
            inputs['input_ids'].to(device),
            inputs['attention_mask'].to(device),
        )
        logits_per_image = torch.matmul(image_features, text_features.t())
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text
    
    
    def top_k(self, ):
        ...
        
        
    def evaluation(self,):
        ...
        
        
        
        


def main(
    pretrained_clip_path: str,
    
):
    
    dtype = torch.float16   # Use fp16 for evaluation
    device = torch.device('cuda:0') # Use single GPU for evaluation
    
    # Load pretrained CLIP model
    processor = CLIPProcessor.from_pretrained(pretrained_clip_path)
    model = CLIPModel.from_pretrained(pretrained_clip_path)
    model.to(device, dtype=dtype)
    model.requires_grad_(False)
    
    evaluator = Evaluator(processor, model,)
    
    # Load validation dataset
    eval_dset = CLIPDataset(root=data_root, split='val', processor=processor)
    
    
"""
Define the pytorch dataset for image-text pairs
"""

import os
import json
import numpy as np
import imageio.v3 as imageio
import torch

from torch.utils.data import Dataset
from transformers import CLIPProcessor

class CLIPDataset(Dataset):
    
    def __init__(
        self,
        root,
        processor: CLIPProcessor,
        split: str = 'train',
    ):
        
        super().__init__()
        self.split = split
        self.data_dir = os.path.join(root, split)
        
        pairs_path = os.path.join(self.data_dir, f'img_text_pair_{split}.json')
        with open(pairs_path, 'r') as f:
            pairs = json.load(f)
        # self.img_text_pairs = pairs
        
        img_names = []
        captions = []
        for pair in pairs:
            img_names.append(pair['img'])
            captions.append(pair['caption'])
        self.img_names = img_names
        self.captions = captions
        
        self.processor = processor
        
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, index):
        img_name = self.img_names[index]
        caption = self.captions[index]
        
        img_path = os.path.join(self.data_dir, 'imgs', img_name)
        image = imageio.imread(img_path).astype(np.float32) / 255.0
        
        # To CHW
        if image.ndim == 2: # gray scale or binary
            image = np.stack([image]*3)
        else:
            image = image.transpose(2, 0, 1)
        # image = torch.from_numpy(image) / 255.0 # scale to (0, 1)
        
        inputs = self.processor(
            text=caption,
            images=image,
            return_tensors='pt',
            padding='max_length',
            truncation='longest_first'
        )
        inputs = {k: v[0] for k, v in inputs.items()}

        
        return inputs
        
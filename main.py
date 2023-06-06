"""
main script to fine-tune CLIP
"""

import os
import inspect
import argparse
import numpy as np
import torch
import torch.nn as nn
import imageio.v2 as imageio
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdmaW
from einops import rearrange

from accelerate import Accelerator
from transformers import CLIPProcessor, CLIPModel


def main(
    pretrained_clip_path: str,
):
    # Load pretrained clip model
    model = CLIPModel.from_pretrained(pretrained_clip_path)
    processor = CLIPProcessor.from_pretrained(pretrained_clip_path)

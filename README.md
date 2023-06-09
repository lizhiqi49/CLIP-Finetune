# CLIP Fine-tuning on Cytopathologic Image-caption Pairs

This is a project of course "Image Processing and Analysis" of Westlake University, and this repository seeks to fine-tune CLIP model on a dataset contains 40,000 cytopathologic image-caption pairs.


## Quickstart

### Setup environment

1. Install Pytorch

This project is experimented on Pytorch-2.0, please refer to [Pytorch's official webpage](https://pytorch.org/) for installation.

2. Install dependency packages

```bash
git clone https://github.com/lizhiqi49/CLIP-Finetune
cd CLIP-Finetune
pip install -r requirements.txt
```

### Download pretrained CLIP models

The pretrained CLIP model we used is OpenAI's official checkpointing on [huggingface-hub](https://huggingface.co/), and the CLIP version is [openai/clip-vit-base-patch16](https://huggingface.co/openai/clip-vit-base-patch16). You can choose other CLIP versions under OpenAI's repositories. For more details about usage of huggingface's model please refer to [this page](https://huggingface.co/docs/transformers/model_doc/clip).


### Setup dataset

Your dataset should be in this structure:

```
- data_root
    - train
        - imgs
            - xxx.jpg/png
            - xxx.jpg/png
            ...
        - img_text_pair.json
    - val
        - imgs
        - img_text_pair.json
```

Here `img_text_pair.json` should contains a list of dictionaries where each dictionary represents a image-text pair.

```
[
    {
        "img": "xxx.jpg/png",
        "caption": "..."
    },
    {
        ...
    },
    ...
]
```

### Start training

1. Configure hyper-parameters

Configure your own training hyper-parameters under `configs/{exp_name}.yaml`.

2. Configure Accelerate

This project uses library [Accelerate](https://github.com/huggingface/accelerate) for mixed-precision and distributed training, before training start, you need configure your accelerate using `accelerate config` on your shell. 

3. Train!

```
accelerate launch train.py --config configs/{exp_name}.yaml
```

### Evaluation

```
python evaluation.py --data_root {data_root} --batch_size {batch_size} --ks 1 10 50 \
    --pretrained_clip_path {pretrained_clip_path} \
    --pretrained_lora_path {pretrained_lora_path}
```

This command will compute metrics of `hit@1,10,50` on the validation dataset of given `data_root` with `batch_size`,  `pretrained_clip_path`, `pretrained_lora_path` specified. When both two pretrained model path provided, `pretrained_lora_path` first.

Defaultly it will compute `hit@k` for images on all captions in the validation dataset. Finally, a .json file contains evaluation results will be saved in the provided pretrained path.




## Experiment logs

### clip_ft_full_raw_data

directly using the raw data (unprocessed) to fine-tune the whole model, all captions are padding or truncated to the model_max_length, which is 77.

3090 GPU x 4, each with batch size as 128, learning rate as 1e-5 with 100 steps warmup, training steps as 5000.

"""
Time consuming: ~2.5h
GPU memory consuming: 22G
Number of Parameters: 100%

hit@10: 0.0954
hit@25: 0.1625
hit@50: 0.2343
hit@100: 0.3254
"""


### clip_ft_lora_raw_data

Fine-tune CLIP model with LoRA and all attention projection layers (Q, K, V) are applied with LoRA.
LoRA rank is set as 8 and it's scaling factor is set to 1.
The dataset and training setup is the same as `clip_ft_full_raw_data`.

"""
Time consuming: ~1h
GPU memory consuming: 16G
Number of Parameters: 0.49%
"""





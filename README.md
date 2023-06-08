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





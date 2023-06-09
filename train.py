"""
main script to fine-tune CLIP
"""

import os
import math
import inspect
import logging
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import imageio.v2 as imageio
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange
from typing import Optional, Literal
from omegaconf import OmegaConf

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger

import transformers
from transformers import CLIPProcessor, CLIPModel

from data.dataloader import CLIPDataset

logger = get_logger(__name__, log_level="INFO")


# TODO: tokenizer's parallelism issue
# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
# 	- Avoid using `tokenizers` before the fork if possible
# 	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


def get_scheduler(optimizer: Optimizer, num_warmup_steps: int = None, last_epoch: int = -1):
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase. 
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    if num_warmup_steps is not None:
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1.0, num_warmup_steps))
            return 1.0
    else:
        lr_lamdba = lambda _: 1
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
        


def main(
    *,
    exp_name: Optional[str] = None,
    data_root: str = './dataset',
    pretrained_clip_path: str = 'openai/clip-vit-base-patch16',
    seed: Optional[int] = None,
    learning_rate: float = 1e-5,
    train_batch_size: int = 1,
    val_batch_size: int = 1,
    num_workers: int = 96,
    max_train_steps: int = 1000,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-3,
    adam_epsilon: float = 1e-08,
    num_warmup_steps: Optional[int] = None,
    last_epoch: int = -1,
    max_grad_norm: float = 1.0,
    mixed_precision: Literal['no', 'fp16'] = 'fp16',
    gradient_accumulation_steps: int = 1,
    checkpointing_step_interv: int = 1000,
    validation_step_interv: int = 500,
    resume_from_checkpoint: Optional[str] = None,
    output_dir: str = './output',
):  
    # Get config
    *_, config = inspect.getargvalues(inspect.currentframe())
    exp_name = exp_name if exp_name is not None else datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, exp_name)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with='tensorboard',
        project_dir=os.path.join(output_dir, 'logs')
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
    else:
        transformers.utils.logging.set_verbosity_error()
        
    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)
        
    # Handle the output folder creation
    if accelerator.is_main_process:
        # now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        # output_dir = os.path.join(output_dir, now)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/validation", exist_ok=True)
        os.makedirs(f"{output_dir}/pretrained", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))
    
    # Load pretrained clip model
    model = CLIPModel.from_pretrained(pretrained_clip_path)
    processor = CLIPProcessor.from_pretrained(pretrained_clip_path)
    logger.info(f"  Load pretrained CLIP model from {pretrained_clip_path}.")
    
    
    # Load dataset
    train_dset = CLIPDataset(
        root=data_root,
        split='train',
        processor=processor
    )
    train_dloader = DataLoader(
        train_dset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers
    )
    val_dset = CLIPDataset(root=data_root, split='val', processor=processor)
    val_dloader = DataLoader(
        val_dset, batch_size=val_batch_size
    )
    
    # Create optimizer
    params = model.parameters()
    optimizer = torch.optim.AdamW(
        params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon
    )
    # lr scheduler
    lr_scheduler = get_scheduler(
        optimizer, num_warmup_steps, last_epoch
    )
    
    # Prepare everything with `accelerator`
    model, optimizer, train_dloader, val_dloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dloader, val_dloader, lr_scheduler
    )
    
    # # mixed precision
    # weight_dtype = torch.float32
    # if accelerator.mixed_precision == "fp16":
    #     weight_dtype = torch.float16
    # elif accelerator.mixed_precision == "bf16":
    #     weight_dtype = torch.bfloat16
        
    # Move model to device and cast to weight_dtype
    model.to(accelerator.device)
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(exp_name)

    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0
    
    # Potentially load in the weights and states from a previous save
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            if len(dirs) > 0:
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1]
            else:
                path = None
        
        if path is None:
                accelerator.print(
                    f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                resume_from_checkpoint = False
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(output_dir, path))
            global_step = int(path.split("-")[1])

            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = global_step % num_update_steps_per_epoch
            lr_scheduler.last_epoch = resume_step
            
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    model.eval()    # frozen dropout and layer norms
    for epoch in range(first_epoch, num_train_epochs):
        # model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dloader):
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(model):
                # Get input
                inputs = batch
                # Forward
                inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
                inputs.update({'return_loss': True})
                outputs = model(**inputs)
                # Get loss
                loss = outputs.loss

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params, max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                
                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if global_step % checkpointing_step_interv == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                if global_step % validation_step_interv == 0:
                    # if accelerator.is_main_process:
                    val_pbar = tqdm(val_dloader, leave=False, disable=not accelerator.is_local_main_process)
                    val_pbar.set_description("Val")
                    for batch in val_dloader:    # only one batch
                        # Get input
                        inputs = batch

                        # Forward
                        inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
                        inputs.update({'return_loss': True})
                        with torch.no_grad():
                            outputs = model(**inputs)
                        loss = outputs.loss
                        
                        # Gather the losses across all processes for logging (if we use distributed training).
                        avg_loss = accelerator.gather(loss.repeat(val_batch_size)).mean()
                        val_loss = avg_loss.item() 
                        
                        # Logging
                        accelerator.log({"val_loss": val_loss}, step=global_step)
                        val_pbar.update(1)
                        val_pbar.set_postfix({"val_step_loss": val_loss})
                    

            if global_step >= max_train_steps:
                break
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.save_pretrained(f"{output_dir}/pretrained")
    accelerator.end_training()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/clip_ft.yaml")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))

    
    
    

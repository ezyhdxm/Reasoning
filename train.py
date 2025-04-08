import torch
from torch import nn
from typing import Tuple, Callable, Dict, Any, Optional
from optimizer import get_optimizer_and_lr_schedule
from utils import tabulate_model
from torch import optim
import os
from absl import logging

import wandb

from tqdm.notebook import trange

from ml_collections import ConfigDict

import ipywidgets as widgets
from IPython.display import display
import gc


import hashlib
import json




def on_stop_clicked(b):
    stop_flag["stop"] = True
    print("Stop button clicked.")

def get_hash(config: ConfigDict) -> str:
    return hashlib.md5(config.to_json(sort_keys=True).encode("utf-8")).hexdigest()


def _init_log() -> dict:
    """
    Initialize log dictionary for evaluation metrics.
    """
    log = {
        "eval/step": [], "train/lr": [], "train/loss": [], "eval/loss": [], 
        "eval/accuracy": [], "train/accuracy": [], "train/op-accuracy": [], "eval/op-accuracy": []
        }
    return log



def train(model, sampler, config: ConfigDict, verbose=True) -> None:
    exp_name = f"train_{get_hash(config)}"
    exp_dir = os.path.join(config.work_dir, exp_name)   
    logging.info(f"Train Experiment\nNAME: {exp_name}\nCONFIG:\n{config}")
    
    print("Results are saved in: ", exp_dir)

    pad_ignore = config.training.pad_ignore

    stop_button = widgets.Button(description="Stop Training")
    stop_flag = {"stop": False}
    stop_button.on_click(on_stop_clicked)
    display(stop_button)
    
    SEQ_LEN = config.task.max_variables * 8 - 14
    
    if config.device == "cpu":
        MAX_SIZE = 500 * (32 * 1024 * 1024 // (config.batch_size * SEQ_LEN) // 500)  # 500 MB
    else:
        MAX_SIZE = 500 * (64 * 1024 * 1024 // (config.batch_size * SEQ_LEN) // 500)
    
    criterion = (
        nn.CrossEntropyLoss(label_smoothing=config["training"]["label_smoothing"], reduction='none') 
        if config["training"]["label_smoothing"] > 0 else nn.CrossEntropyLoss(reduction='none')
    )

    # Skip if already completed
    log_path = os.path.join(exp_dir, "log.json")

    if os.path.exists(log_path):
        print(f"{exp_name} already completed")
        return
    
    # Save config
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        f.write(config.to_json())
    
    model.to(config.device)

    checkpoint_path = os.path.join(exp_dir, f"checkpoints")

    print(tabulate_model(model, SEQ_LEN, config.batch_size, config.device))

    optimizer, scheduler = get_optimizer_and_lr_schedule(**config.training, params=model.parameters())
    
    print("Initialized model, optimizer, and train state")

    # Data samplers
    
    print("Initialized data samplers")
    

    # Logging
    log = _init_log()
    
    step = 0

    attn_maps = {}

    if pad_ignore:
        test_data, test_mask, test_attn_mask = sampler.generate(config.test_size, get_attn_mask=True)
    else:
        test_data, test_mask = sampler.generate(config.test_size)
    
    test_data = test_data.to(config.device)
    test_target_flat = test_data[:, 1:].reshape(-1) # shape (B*T,)
    

    test_mask_flat = test_mask[:, 1:].reshape(-1) # shape (B*T,)    
    
    # Training loop
    
    epochs = min(config.training.total_steps, MAX_SIZE)
    while (config.training.total_steps % epochs != 0) and (epochs > 0):
        epochs -= 1
    
    tot_iters = config.training.total_steps // epochs
    
    wandb.init(config=config, name=exp_name, **config["wandb"])

    # causal_mask = sampler.causal_mask
    
    print("Start training...")
    for iters in trange(tot_iters):
        if pad_ignore:
            data, mask, attn_mask = sampler.generate(num_samples=config.batch_size * epochs, get_attn_mask=True)
            attn_shape = attn_mask.shape # (num_samples, 1, T, T)
            attn_mask = attn_mask.reshape(epochs, config.batch_size, *attn_shape[1:])
        else:
            data, mask = sampler.generate(num_samples=config.batch_size * epochs)
        
        data = data.reshape(epochs, config.batch_size, -1)
        mask = mask.reshape(epochs, config.batch_size, -1)
        

        for i in trange(epochs, leave=False):
            step += 1

            if stop_flag["stop"]:
                print("Training stopped by button.")
                break
            
            if config.training.get_checkpoints > 0 and ((step % config.training.get_checkpoints == 0) 
                                                        or (step < min(config.training.get_checkpoints, 200) and step % 5 == 0)):
                
                os.makedirs(checkpoint_path, exist_ok=True)
                torch.save({
                    "model": model.state_dict(), 
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    }, os.path.join(checkpoint_path, f"model_{step}.pt"))

            batch = data[i].to(config.device)
            batch_mask = mask[i].to(config.device)
            
            batch_attn_mask = attn_mask[i].to(config.device) if pad_ignore else None
            
            model.train()
            optimizer.zero_grad()
            

            if (config.training.get_attn) > 0 and (step % config.training.get_attn == 0): # and get_attn_flag:
                preds, attn = model(batch, get_attn=True, mask=batch_attn_mask)
                attn_maps[step] = {l: v.clone() for l, v in attn.items()}
            else:
                preds, _ = model(batch, get_attn=False, mask=batch_attn_mask)

            preds = preds[:, :-1]
            targets = batch[:, 1:]
            batch_mask = batch_mask[:, 1:]
            
            preds_flat = preds.reshape(-1, config.vocab_size)        # shape (B*T, N)
            targets_flat = targets.reshape(-1)       # shape (B*T,)
            mask_flat = batch_mask.reshape(-1)   # shape (B*T,)
            

            # Compute per-position loss
            loss_all = criterion(preds_flat, targets_flat)  # shape (B*T,)

            # Apply mask
            masked_loss = loss_all[mask_flat>0]

            # Final loss
            loss = masked_loss.mean()
            
            log["train/loss"].append(loss.item())
            wandb.log({"train/loss": loss}, step=step)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            accuracy_all = (preds_flat.argmax(dim=-1) == targets_flat).float()
            masked_accuracy = accuracy_all[mask_flat>0]
            op_accuracy = accuracy_all[mask_flat==2].mean()
            accuracy = masked_accuracy.mean()
            log["train/accuracy"].append(accuracy.item())
            log["train/op-accuracy"].append(op_accuracy.item())
            wandb.log({"train/accuracy": accuracy}, step=step)
            wandb.log({"train/op-accuracy": op_accuracy}, step=step)

            # Evaluation
            if (step % config.training.eval_iter == 0):
                if verbose:
                    print(f"Step: {step}")
                log["eval/step"].append(step)
                lr_val = scheduler.get_last_lr()[0]
                log["train/lr"].append(lr_val)
                wandb.log({"train/lr": lr_val}, step=step)

                model.eval()
                with torch.no_grad():
                    test_attn_mask_device = test_attn_mask.to(config.device) if pad_ignore else None

                    preds, _ = model(test_data, get_attn=False, mask=test_attn_mask_device)
                    
                    if pad_ignore:
                        del test_attn_mask_device
                        gc.collect()
                        torch.cuda.empty_cache()

                    preds = preds[:, :-1]
                    preds_flat = preds.reshape(-1, config.vocab_size)        # shape (B*T, N)  

                    
                    # Compute per-position loss
                    loss_all = criterion(preds_flat, test_target_flat).cpu()  # shape (B*T,)
                    # Apply mask
                    masked_loss = loss_all[test_mask_flat>0]
                    
                    accuracy_all = (preds_flat.argmax(dim=-1) == test_target_flat).float().cpu()
                    masked_accuracy = accuracy_all[test_mask_flat>0]
                    op_accuracy = accuracy_all[test_mask_flat==2].mean()
                    accuracy = masked_accuracy.mean()
                    log["eval/accuracy"].append(accuracy.item())
                    log["eval/op-accuracy"].append(op_accuracy.item())
                    wandb.log({"eval/accuracy": accuracy}, step=step)
                    wandb.log({"eval/op-accuracy": op_accuracy}, step=step)

                    # Final loss
                    eval_loss = masked_loss.mean()
                    log["eval/loss"].append(eval_loss.item())
                    wandb.log({"eval/loss": eval_loss}, step=step)

    # Save final checkpoint
    os.makedirs(checkpoint_path, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step
    }, os.path.join(checkpoint_path, f"model_final_{step}.pt"))

    # Save logs
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    
    torch.save(attn_maps, os.path.join(exp_dir,"attn_maps.pt"))

    print("Training complete.")


    
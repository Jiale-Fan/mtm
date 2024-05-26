# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Main script for training a policy given a dataset.
"""
import os
import pprint
import random
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Sequence, Tuple

import hydra
import numpy as np
import torch
import torch.distributed
import torch.multiprocessing
import torch.nn.functional as F
import torch.nn.parallel
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

from research.logger import WandBLogger, WandBLoggerConfig, logger, stopwatch
from research.mtm.datasets.base import DatasetProtocol
from research.mtm.datasets.softgym import SoftgymDataset
from research.mtm.distributed_utils import DistributedParams, get_distributed_params
from research.mtm.masks import (
    MaskType,
    create_bc_mask,
    create_forward_dynamics_mask,
    create_full_random_masks,
    create_goal_n_reaching_masks,
    create_goal_reaching_masks,
    create_inverse_dynamics_mask,
    create_random_autoregressize_mask,
    create_random_bc_masks,
    create_random_mask,
    create_random_masks,
    create_rcbc_mask,
    maybe_add_rew_to_mask,
)
from research.mtm.models.mtm_model import MTM, make_plots_with_masks
from research.mtm.tokenizers.base import Tokenizer, TokenizerManager
from research.mtm.tokenizers.patchify import PatchifyTokenizer
from research.mtm.tokenizers.continuous import ContinuousTokenizer
from research.mtm.utils import (
    get_cfg_hash,
    get_ckpt_path_from_folder,
    get_git_dirty,
    get_git_hash,
    set_seed_everywhere,
)

dir_path = os.path.dirname(os.path.realpath(__file__))

# export PYTHONPATH=$PYTHOPATH:$PWD

@dataclass
class RunConfig:
    seed: int = 0
    """RNG seed."""

    batch_size: int = 64
    """Batch size used during training."""

    n_workers: int = 8
    """Number of workers for loading data."""

    log_every: int = 100
    """Print training loss every N steps."""

    print_every: int = 1000
    """Print training loss every N steps."""

    eval_every: int = 5000
    """Evaluate model every N steps."""

    save_every: int = 5000
    """Evaluate model every N steps."""

    device: str = "cuda"
    """Device to use for training."""

    mask_ratios: Sequence[float] = (0.15, 0.35, 0.55, 0.75, 0.85, 0.95)

    mask_patterns: Sequence[str] = ("RANDOM",)
    """Indices of masks to use for evaluation."""

    warmup_steps: int = 1_000
    """Number of warmup steps for learning rate scheduler."""

    num_train_steps: int = 5_000_000
    """Number of training steps."""

    learning_rate: float = 1e-3
    """Learning rate."""

    weight_decay: float = 1e-5
    """Weight decay."""

    traj_length: int = 1
    """Trajectory length."""

    mode_weights: Tuple[int, int, int] = (0.2, 0.1, 0.7)
    """State action return."""

    tsp_ratio: int = 1
    """Train steps per state only train steps ratio.

    1 means train state only every step.
    2 means train state only every other step, etc.
    """


def train_one_batch(
    model: MTM,
    optimizer: torch.optim.Optimizer,
    scheduler: Callable,
    tokenizer_manager: TokenizerManager,
    discrete_map: Dict[str, bool],
    batch: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    loss_keys: Sequence[str] = None,
) -> Dict[str, Any]:
    encoded_batch = tokenizer_manager.encode(batch) # [1024, 16, 1] -> [1024, 16, 1, 1]

    # train the model
    predicted_trajectories = model(encoded_batch, masks)

    # compute the loss
    model_without_ddp = model.module if hasattr(model, "module") else model
    if loss_keys is None:
        loss_keys = model_without_ddp.config.loss_keys

    loss, losses_dict, masked_losses, masked_c_losses = MTM.forward_loss(
        encoded_batch,
        predicted_trajectories,
        masks,
        discrete_map,
        norm=model_without_ddp.norm,
        reduce_use_sum=model_without_ddp.config.reduce_use_sum,
        loss_keys=loss_keys,
    )
    # create a dictionary to log all of the losses
    log_dict = {"train/train_loss": loss.item()}
    log_dict["train/lr"] = scheduler.get_last_lr()[0]
    for k, v in losses_dict.items():
        log_dict[f"train/full_loss_{k}"] = v.item()
    for k, v in masked_losses.items():
        log_dict[f"train/masked_loss_{k}"] = v.item()
    for k, v in masked_c_losses.items():
        log_dict[f"train/masked_c_loss_{k}"] = v.item()

    # backprop
    model.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()

    with torch.no_grad():
        mse_loss = 0
        predictions = tokenizer_manager.decode(predicted_trajectories)
        for k, v in predictions.items():
            _mse = F.mse_loss(v.to(torch.float32), batch[k].to(torch.float32)).item()
            log_dict[f"train/mse_{k}"] = _mse
            mse_loss += _mse
        log_dict["train/mse_sum"] = mse_loss
    return log_dict


def main(hydra_cfg):
    _main(hydra_cfg)
    # p = Profiler()
    # with p:
    #     _main(hydra_cfg)
    # p.print()


def _main(hydra_cfg):
    cfg: RunConfig = hydra.utils.instantiate(hydra_cfg.args)
    dp: DistributedParams = get_distributed_params()

    # make dir
    # get time
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
    save_dir = "./saved_models/"+time_str
    os.makedirs(save_dir, exist_ok=True)

    torch.cuda.set_device(dp.local_rank)
    distributed = dp.world_size > 1
    if distributed:
        logger.info(
            f"Initializing rank {dp.rank} (local rank {dp.local_rank}) in total world size {dp.world_size} (local world size {dp.local_world_size}) with master addr:port {dp.master_addr}:{dp.master_port}"
        )
        torch.distributed.init_process_group(
            backend="nccl", rank=dp.rank, world_size=dp.world_size
        )

    set_seed_everywhere(cfg.seed)
    pprint.pp(cfg)

    logger.info(f"Working directory: {os.getcwd()}")

    with open("config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(hydra_cfg))

    train_dataset: DatasetProtocol
    val_dataset: DatasetProtocol

    train_dataset, val_dataset = hydra.utils.call(
        hydra_cfg.dataset, seq_steps=cfg.traj_length
    )
    logger.info(f"Train set size = {len(train_dataset)}")
    logger.info(f"Validation set size = {len(val_dataset)}")

    ### for debugging
    # train_dataset.trajectory_statistics()

    if hydra_cfg.state_only_dataset is not None:
        state_only_train_dataset, state_only_val_dataset = hydra.utils.call(
            hydra_cfg.state_only_dataset,
            seq_steps=cfg.traj_length,
        )
        logger.info(f"State Only Train set size = {len(state_only_train_dataset)}")
        logger.info(f"State Only Validation set size = {len(state_only_val_dataset)}")

    if "tokenizers" in hydra_cfg:
        tokenizers: Dict[str, Tokenizer] = {
            k: hydra.utils.call(v, key=k, train_dataset=train_dataset)
            for k, v in hydra_cfg.tokenizers.items()
        }

    # tokenizers = {"states": PatchifyTokenizer(patch_size=16)} # TODO: each modality (states, actions, reward etc.) should have its own tokenizer
    tokenizer_manager = TokenizerManager(tokenizers).to(cfg.device)

    discrete_map: Dict[str, bool] = {}
    for k, v in tokenizers.items():
        discrete_map[k] = v.discrete
    logger.info(f"Tokenizers: {tokenizers}")

    if distributed:
        train_sampler = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=dp.world_size, rank=dp.rank, shuffle=True
        )
        val_sampler = torch.utils.data.DistributedSampler(
            val_dataset, num_replicas=dp.world_size, rank=dp.rank, shuffle=False
        )
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    with stopwatch("data loader"):
        train_loader = DataLoader(
            train_dataset,
            # shuffle=True,
            pin_memory=True,
            batch_size=cfg.batch_size,
            num_workers=cfg.n_workers,
            sampler=train_sampler,
        )
        val_loader = DataLoader(
            val_dataset,
            # shuffle=False,
            batch_size=cfg.batch_size,
            num_workers=1,
            sampler=val_sampler,
        )


    train_batch = next(iter(train_loader))
    tokenized = tokenizer_manager.encode(train_batch)
    data_shapes = {}
    for k, v in tokenized.items():
        data_shapes[k] = v.shape[-2:]
    logger.info(f"Data shapes: {data_shapes}")

    # create the model

    model_config = hydra.utils.instantiate(hydra_cfg.model_config)
    model = model_config.create(data_shapes, cfg.traj_length)
    model.to(cfg.device)
    model.train()
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[dp.local_rank], output_device=dp.local_rank
        )

    optimizer = MTM.configure_optimizers(
        model,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),  # following BERT
    )

    def _schedule(step):
        # warmp for 1000 steps
        if step < cfg.warmup_steps:
            return step / cfg.warmup_steps

        # then cosine decay
        assert cfg.num_train_steps > cfg.warmup_steps
        step = step - cfg.warmup_steps
        return 0.5 * (
            1 + np.cos(step / (cfg.num_train_steps - cfg.warmup_steps) * np.pi)
        )

    scheduler = LambdaLR(optimizer, lr_lambda=_schedule)

    # create a wandb logger and log params of interest
    wandb_cfg_log_dict = OmegaConf.to_container(hydra_cfg)
    wandb_cfg_log_dict["*discrete_map"] = discrete_map
    wandb_cfg_log_dict["*path"] = str(os.getcwd())
    wandb_cfg_log_dict["*mp"] = cfg.mask_patterns
    wandb_cfg_log_dict["*git_hash"] = get_git_hash()
    wandb_cfg_log_dict["*git_dirty"] = get_git_dirty()
    wandb_cfg_log_dict["*hydra_cfg_hash"] = get_cfg_hash(hydra_cfg)
    wandb_cfg_log_dict["*num_parameters"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    wandb_cfg_log = WandBLoggerConfig(
        experiment_id=f"{dp.job_id}-{dp.rank}",
        project=hydra_cfg.wandb.project,
        entity=hydra_cfg.wandb.entity or None,
        resume=hydra_cfg.wandb.resume,
        group=dp.job_id,
    )

    if wandb_cfg_log.resume:
        exp_id = wandb_cfg_log_dict["*hydra_cfg_hash"]
        wandb_cfg_log = replace(
            wandb_cfg_log,
            experiment_id=exp_id,
        )
    # wandb_logger = WandBLogger(wandb_cfg_log, wandb_cfg_log_dict, enable=False)
    wandb_logger = WandBLogger(wandb_cfg_log, wandb_cfg_log_dict)

    # train the model
    vis_batch = next(iter(val_loader))  # keep this batch for visualization
    vis_batch = {k: v.to(cfg.device) for k, v in vis_batch.items()}

    has_rew = "rewards" in vis_batch
    has_ret = "returns" in vis_batch
    has_img = "images" in vis_batch # TODO??
    mask_functions_map = {
        MaskType.RANDOM: lambda: create_random_masks(
            data_shapes, cfg.mask_ratios, cfg.traj_length, cfg.device
        ),
        MaskType.FULL_RANDOM: lambda: create_full_random_masks(
            data_shapes, cfg.mask_ratios, cfg.traj_length, cfg.device
        ),
        MaskType.AUTO_MASK: lambda: create_random_autoregressize_mask(
            data_shapes, cfg.mask_ratios, cfg.traj_length, cfg.device, cfg.mode_weights
        ),
        MaskType.RCBC: lambda: create_rcbc_mask(cfg.traj_length, cfg.device),
        MaskType.GOAL: lambda: maybe_add_rew_to_mask(
            cfg.traj_length,
            cfg.device,
            create_goal_reaching_masks,
            has_rew,
            has_img,
            has_ret,
        ),
        MaskType.GOAL_N: lambda: maybe_add_rew_to_mask(
            cfg.traj_length,
            cfg.device,
            create_goal_n_reaching_masks,
            has_rew,
            has_img,
            has_ret,
        ),
        MaskType.ID: lambda: maybe_add_rew_to_mask(
            cfg.traj_length,
            cfg.device,
            create_inverse_dynamics_mask,
            has_rew,
            has_img,
            has_ret,
        ),
        MaskType.FD: lambda: maybe_add_rew_to_mask(
            cfg.traj_length,
            cfg.device,
            create_forward_dynamics_mask,
            has_rew,
            has_img,
            has_ret,
        ),
        MaskType.BC: lambda: maybe_add_rew_to_mask(
            cfg.traj_length,
            cfg.device,
            create_bc_mask,
            has_rew,
            has_img,
            has_ret,
        ),
        MaskType.BC_RANDOM: lambda: maybe_add_rew_to_mask(
            cfg.traj_length,
            cfg.device,
            lambda l, d: create_random_bc_masks(l, d, data_shapes, p=0.5),
            has_rew,
            has_img,
            has_ret,
        ),
    }

    mask_functions = [mask_functions_map[MaskType[i]] for i in cfg.mask_patterns]

    epoch = 0
    step = 0

    batch_iter = iter(train_loader)
    while True:
        B = time.time()
        log_dict = {}
        log_dict["train/epochs"] = epoch

        start_time = time.time()
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(train_loader)
            batch = next(batch_iter)
            epoch += 1

        # cycle between different types

        # ranodmly select mask
        masks = random.choice(mask_functions)()

        if "images" in batch and "images" not in masks:
            masks["images"] = masks["states"]

        batch = {k: v.to(cfg.device, non_blocking=True) for k, v in batch.items()}
        _log_dict = train_one_batch(
            model,
            optimizer,
            scheduler,
            tokenizer_manager,
            discrete_map,
            batch,
            masks,
        )
        log_dict.update(_log_dict)
        # log train step time = time to process a batch
        log_dict["time/train_step"] = time.time() - start_time

        if step % cfg.print_every == 0:
            try:
                train_loss = log_dict["train/train_loss"]
            except:
                train_loss = -1
            logger.info(f"Step: {step}, Train Loss: {train_loss}")

        # save the model

        if dp.rank == 0 and step % cfg.save_every == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                },
                f"model_{step}.pt",
            )

        if step % cfg.log_every == 0:
        # if random.randint(0, cfg.log_every) == 0:
            logger.info(f"Step {step}")
            wandb_logger.log(log_dict, step=step)

        step += 1
        if step >= cfg.num_train_steps:
            break

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        save_dir+f"/model_{step}.pt",
    )


@hydra.main(config_path=".", config_name="config_soft_train", version_base="1.1")
def configure_jobs(hydra_data: DictConfig) -> None:
    logger.info(hydra_data)
    main(hydra_data)


if __name__ == "__main__":
    configure_jobs()

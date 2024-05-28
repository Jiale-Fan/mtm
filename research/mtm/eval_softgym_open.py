# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
JF: script to evaluate the model in closed-loop simulation environment
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
    create_random_autoregressive_mask,
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
from envs.env import Env
from collections import deque
from research.mtm.train_softgym import evaluate, eval_fd, eval_id

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

def main(hydra_cfg):
    _main(hydra_cfg)
    # p = Profiler()
    # with p:
    #     _main(hydra_cfg)
    # p.print()

def _main(hydra_cfg):
    cfg: RunConfig = hydra.utils.instantiate(hydra_cfg.args)

    model_config = hydra.utils.instantiate(hydra_cfg.model_config)
    model = MTM(hydra_cfg["shapes"][hydra_cfg["image_encoder"]], cfg.traj_length, model_config)
    model.load_state_dict(torch.load(hydra_cfg["ckpt_path"])["model"])
    model.eval()

    # create the tokenizer
    train_dataset: DatasetProtocol
    val_dataset: DatasetProtocol
    train_dataset, val_dataset = hydra.utils.call(
        hydra_cfg.dataset, seq_steps=cfg.traj_length
    )
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

    # evaluate the model
    # encoded_trajectories = tokenizer_manager.encode(trajectories)
    # decoded_gt_trajectories = tokenizer_manager.decode(encoded_trajectories)
    # predictions = predict_fn(encoded_trajectories, masks)
    # decoded_trajs = tokenizer_manager.decode(predictions)

    # val_sampler = torch.utils.data.RandomSampler(val_dataset)
    # val_loader = DataLoader(
    #         val_dataset,
    #         # shuffle=False,
    #         batch_size=cfg.batch_size,
    #         num_workers=1,
    #         sampler=val_sampler,
    #     )
    val_sampler = torch.utils.data.SequentialSampler(train_dataset)

    val_loader = DataLoader(
            train_dataset,
            # shuffle=False,
            batch_size=cfg.batch_size,
            num_workers=1,
            sampler=val_sampler,
        )

        # train the model
    vis_batch = next(iter(val_loader))  # keep this batch for visualization
    vis_batch = {k: v.to(cfg.device) for k, v in vis_batch.items()}

    has_rew = "rewards" in vis_batch
    has_ret = "returns" in vis_batch
    has_img = "images" in vis_batch # TODO??

    data_shapes = hydra_cfg["shapes"][hydra_cfg["image_encoder"]]
    cfg.device = "cpu"

    mask_functions_map = {
        MaskType.RANDOM: lambda: create_random_masks(
            data_shapes, cfg.mask_ratios, cfg.traj_length, cfg.device
        ),
        MaskType.FULL_RANDOM: lambda: create_full_random_masks(
            data_shapes, cfg.mask_ratios, cfg.traj_length, cfg.device
        ),
        MaskType.AUTO_MASK: lambda: create_random_autoregressive_mask(
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
    batch_iter = iter(val_loader)

    while True:
        # evaluate the model
        start_time = time.time()
        model.eval()
        try:
            val_batch = next(batch_iter)
        except StopIteration:
            break

        val_batch = {
            k: v.to('cpu', non_blocking=True) for k, v in val_batch.items()
        }
        # val_batch = {
        #     k: v.to(cfg.device, non_blocking=True) for k, v in val_batch.items()
        # }

        # log_dict = val_dataset.eval_logs(model, tokenizer_manager)
        log_dict = {}

        eval_masks = random.choice(mask_functions)()

        _val_dict = evaluate(
                model,
                tokenizer_manager,
                discrete_map,
                val_batch,
                vis_batch,
                eval_masks,
            )
        log_dict.update(_val_dict)


        # for everything with eval prefix keep the max
        max_log = {}
        log_dict["time/eval_time"] = time.time() - start_time

        if cfg.traj_length >= 2:
            log_dict.update(
                eval_fd(model, None, val_batch, tokenizer_manager)
            )
            # log_dict.update(
            #     eval_id(model, None, val_batch, tokenizer_manager)
            # )

        # wandb_logger.log(log_dict, step=step)
        # val_loss = log_dict["val/val_loss"]
        # logger.info(f"Step: {step}, Val Loss: {val_loss}")

def create_env(env_args):
    # symbolic = False #?
    env = Env(**env_args)
    env.seed(env_args.seed)
    return env



@hydra.main(config_path=".", config_name="config_soft_eval", version_base="1.1")
def configure_jobs(hydra_data: DictConfig) -> None:
    logger.info(hydra_data)
    main(hydra_data)


if __name__ == "__main__":
    configure_jobs()

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

class ShortMemory:
    def __init__(self, seq_len, shapes: Dict[str, Tuple[int, int]], init_rtg = 8, expected_rps = 0.3, obs_type = "rgb"):
        self.seq_len = seq_len
        assert seq_len >= 2
        self.shapes = shapes
        self.states_memory = deque([torch.zeros(shapes["states"])] * seq_len, maxlen=seq_len)
        self.actions_memory = deque([torch.zeros(shapes["actions"])] * (seq_len-1), maxlen=seq_len-1) # maximum length of seq_len -1
        self.rewards_memory = deque([torch.zeros((1))] * (seq_len-1), maxlen=seq_len-1)
        self.returns_memory = deque([torch.zeros((1))] * seq_len, maxlen=seq_len)
        self.returns_memory.append(torch.tensor([init_rtg]))
        self.expected_rps = expected_rps
        self.num_valid_states = 0
        self.obs_type = obs_type

    def append_reward(self, reward):
        self.rewards_memory.append(reward)
        self.returns_memory[-1] = self.returns_memory[-2] - reward

    def get_action(
        self,
        model: MTM,
        obs: torch.Tensor,
        tokenizer_manager,
        ratio: int = 1,
        no_prev_action: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate the model on the forward dynamics task.
        Args:
            obs: [1, 128, 128, 3]
            tokenizer_manager (TokenizerManager): tokenizer_manager
        """
        if self.num_valid_states < self.seq_len:
            self.num_valid_states += 1

        if self.obs_type == "rgb":
            self.states_memory.append(obs.squeeze(0).permute(1,2,0))
        elif self.obs_type == "keypoints":
            self.states_memory.append(obs)
        else:
            raise ValueError("obs_type should be one of rgb, keypoints")

        self.returns_memory.append(self.returns_memory[-1] - torch.tensor([self.expected_rps]) )

        eval_batch = {
            "states": torch.stack(list(self.states_memory)).unsqueeze(0),
            "actions": torch.stack(list(self.actions_memory)+[torch.zeros(self.shapes["actions"])]).unsqueeze(0),
            "returns": torch.stack(list(self.returns_memory)).unsqueeze(0), # TODO: how to give the return to illicit the best action?
        }

        device = eval_batch["states"].device
        
        obs_mask1 = torch.ones(self.seq_len, device=device)
        obs_mask1[:self.seq_len-self.num_valid_states] = 0
        if no_prev_action:
            actions_mask1 = torch.zeros(self.seq_len, device=device)
        else:
            actions_mask1 = torch.ones(self.seq_len, device=device)
            actions_mask1[:self.seq_len-self.num_valid_states] = 0
        returns_mask = torch.ones(self.seq_len, device=device)
        returns_mask[:self.seq_len-self.num_valid_states] = 0

        masks = {
            "states": obs_mask1,
            "actions": actions_mask1,
            "returns": returns_mask,
        }

        predictions = model.mask_git_forward(
            tokenizer_manager.encode(eval_batch),
            masks,
            ratio=ratio,
        )
        predicted_next_action = tokenizer_manager.decode(predictions)["actions"][0, -1]

        self.actions_memory.append(predicted_next_action)

        return predicted_next_action


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

    env = create_env(hydra_cfg.env_args)
    
    # img_size = 720
    # all_frames, all_rewards, all_actions, all_keypoints = [], [], [], []
    # sample_stochastically = False

    num_episodes = 8

    for i in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        # short_memory = ShortMemory(seq_len=4, shapes={"states": (128, 128, 3), "actions": (4)})
        short_memory = ShortMemory(seq_len=4, shapes={"states": (33), "actions": (4)}, obs_type="keypoints")
        frames, rewards, actions, keypoints = [], [], [], []
        while not done:
            keypoints.append(obs.to('cpu').numpy())
            frames.append(env.get_image(128, 128))
            action = short_memory.get_action(model, obs, tokenizer_manager, ratio=1, no_prev_action=False)
            action = action.detach().to('cpu').numpy()
            obs, reward, done, info = env.step(action)

            short_memory.append_reward(reward)

            actions.append(action)
            rewards.append(reward)
            
        
        # all_frames.append(frames)
        # all_rewards.append(rewards)
        # all_actions.append(actions)
        # all_keypoints.append(keypoints)


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

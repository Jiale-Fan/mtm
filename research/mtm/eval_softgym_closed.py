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
from softgym.utils.visualization import save_numpy_as_gif


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

def deepcopy_dict_of_tensors(dict_of_tensors):
    # Create a new dictionary to hold the deep copies
    deep_copied_dict = {}
    for key, tensor in dict_of_tensors.items():
        # Use tensor.clone() to create a deep copy of each tensor
        deep_copied_dict[key] = tensor.clone()
    return deep_copied_dict

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
    
    def get_action_beam_search(
        self,
        model: MTM,
        obs: torch.Tensor,
        tokenizer_manager,
        ratio: int = 1,
        no_prev_action: bool = False,
        horizon: int = 5,
        beam_size: int = 5,
    ) -> Dict[str, Any]:

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



        encoded_batch = tokenizer_manager.encode(eval_batch)
        encoded_batch["actions"] = encoded_batch["actions"].to('cpu')
        new_nodes = [[encoded_batch, 0]]
        for i in range(horizon):
            to_expand = new_nodes
            new_nodes = []
            for node, r in to_expand:
                # utilize behavior cloning to get the top k actions for each node
                node_actions = self.forward_behavior_cloning(model, tokenizer_manager, node, level=i, top_k=beam_size) # top_k does not necessarily need to equal beam_size
                for node_action in node_actions:
                    # utilize the reward prediction to get the reward for each node
                    new_node, pred_reward = self.forward_reward_prediction(model, tokenizer_manager, node_action, level=i)
                    new_nodes.append([new_node, r+pred_reward])
                # choose the top k nodes
                new_nodes = sorted(new_nodes, key=lambda x: x[1], reverse=True)[:beam_size]
            
            # forecasting the next state using the forward dynamics
            if i != horizon-1:
                new_nodes = [[self.forward_forward_dynamics(model, tokenizer_manager, node, level=i+1), r] for node, r in new_nodes]
            else:
                break

        action_to_execute = new_nodes[0][0]["actions"][0, -1, 0, :] # this action is in token form
        decoded_action = tokenizer_manager.tokenizers["actions"].decode(action_to_execute.reshape(1, 1, 1, -1).to('cuda'))

        squeezed_action = decoded_action.reshape(-1)
        self.actions_memory.append(squeezed_action.to('cpu'))

        return squeezed_action
    
    def forward_behavior_cloning(self, model, tokenizer_manager, encoded_batch, level, top_k=5, ):

        device = encoded_batch["states"].device

        masks = {
            "states": torch.ones(self.seq_len, device=device),
            "actions": torch.ones(self.seq_len, device=device),
            "rewards": torch.zeros(self.seq_len, device=device),
        }
        masks["actions"][-1] = 0
        if self.num_valid_states+level < self.seq_len:
            masks["states"][:self.seq_len-(self.num_valid_states+level)] = 0
            masks["actions"][:self.seq_len-(self.num_valid_states+level)] = 0
            # masks["rewards"][:self.seq_len-(self.num_valid_states+level)] = 0

        pred_encoded = model.mask_git_forward(
            encoded_batch,
            masks,
            ratio=1,
        )
        pred_reward = tokenizer_manager.tokenizers["actions"].get_probs(pred_encoded["actions"].to('cuda')) # (1, n_clusters)
        top_k_actions = torch.topk(pred_reward, top_k, dim=-1).indices[0, -1]
        most_possible_action_tokens = tokenizer_manager.tokenizers["actions"].look_up_token(top_k_actions)
        ret = []
        for i in range(top_k):
            new_encoded_batch = deepcopy_dict_of_tensors(encoded_batch)
            new_encoded_batch["actions"][:,-1,0,:] = most_possible_action_tokens[i].unsqueeze(0)
            ret.append(new_encoded_batch)

        return ret


    def forward_reward_prediction(self, model, tokenizer_manager, encoded_batch, level, ):
        """
            masks: Dict["states": torch.Tensor [T], "actions": torch.Tensor [T], "rewards": torch.Tensor [T]
            eval_batch: Dict["states": torch.Tensor [B, T, N, dim], "actions": torch.Tensor [B, T, 1, dim], "rewards": torch.Tensor [B, T, 1, dim]
        
        """

        device = encoded_batch["states"].device
        encoded_batch["rewards"] = torch.zeros([1, self.seq_len, 1, 1])

        masks = {
            "states": torch.ones(self.seq_len, device=device),
            "actions": torch.ones(self.seq_len, device=device),
            "rewards": torch.zeros(self.seq_len, device=device), # ? maybe should be 1
        }
        if self.num_valid_states+level < self.seq_len:
            masks["states"][:self.seq_len-(self.num_valid_states+level)] = 0
            masks["actions"][:self.seq_len-(self.num_valid_states+level)] = 0
            # masks["rewards"][:self.seq_len-(self.num_valid_states+level)] = 0


        pred_encoded = model.mask_git_forward(
            encoded_batch,
            masks,
            ratio=1,
        )
        pred_reward = tokenizer_manager.tokenizers["rewards"].decode(pred_encoded["rewards"])[0, -1]

        encoded_batch["rewards"][:, -1] = pred_reward

        return encoded_batch, pred_reward

    def forward_forward_dynamics(self, model, tokenizer_manager, encoded_batch, level):
        device = encoded_batch["states"].device
        masks = {
            "states": torch.ones(self.seq_len, device=device),
            "actions": torch.ones(self.seq_len, device=device),
            "rewards": torch.zeros(self.seq_len, device=device), # ? maybe should be 1
        }
        masks["actions"][-1] = 0
        masks["states"][-1] = 0
        if self.num_valid_states+level < self.seq_len:
            masks["states"][:self.seq_len-(self.num_valid_states+level)] = 0
            masks["actions"][:self.seq_len-(self.num_valid_states+level)] = 0
            # masks["rewards"][:self.seq_len-(self.num_valid_states+level)] = 0

        shifted_encoded_batch = {
            "states": torch.cat([encoded_batch["states"][:, 1:], torch.zeros_like(encoded_batch["states"][:, -1:])], dim=1),
            "actions": torch.cat([encoded_batch["actions"][:, 1:], torch.zeros_like(encoded_batch["actions"][:, -1:])], dim=1),
            "rewards": torch.cat([encoded_batch["rewards"][:, 1:], torch.zeros_like(encoded_batch["rewards"][:, -1:])], dim=1),
        }
        pred_encoded = model.mask_git_forward(
            shifted_encoded_batch,
            masks,
            ratio=1,
        )
        new_node = {
            "states": torch.cat([encoded_batch["states"][:, :-1], pred_encoded["states"][:, -1:]], dim=1),
            "actions": torch.cat([encoded_batch["actions"][:, :-1], pred_encoded["actions"][:, -1:]], dim=1),
            "rewards": torch.cat([encoded_batch["rewards"][:, :-1], pred_encoded["rewards"][:, -1:]], dim=1),
        }

        return new_node
    


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
    all_frames, all_rewards, all_actions, all_keypoints = [], [], [], []
    # sample_stochastically = False

    num_episodes = 100

    for i in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        short_memory = ShortMemory(seq_len=4, shapes={"states": (128, 128, 3), "actions": (4)})
        # short_memory = ShortMemory(seq_len=4, shapes={"states": (33), "actions": (4)}, obs_type="rgb")
        frames, rewards, actions, keypoints = [], [], [], []
        frames.append(env.get_image(128, 128))
        while not done:
            keypoints.append(obs.to('cpu').numpy())
            # action = short_memory.get_action(model, obs, tokenizer_manager, ratio=1, no_prev_action=False)
            with torch.no_grad():
                action = short_memory.get_action_beam_search(model, obs, tokenizer_manager, ratio=1, no_prev_action=False)
            action = action.detach().to('cpu').numpy()
            obs, reward, done, info = env.step(action)
            frames.append(env.get_image(128, 128))
            short_memory.append_reward(reward)
            actions.append(action)
            rewards.append(reward)

        print(f"Episode {i} rewards: {np.array(rewards).sum()}")

                
        save_numpy_as_gif(np.array(frames), f"Episode_{i}.gif")
        
    print(f"Average reward: {np.array(all_rewards).sum()/num_episodes}")    
        
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

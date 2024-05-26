# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Union

import numpy as np
import wandb
from torch.utils.data import Dataset

from research.mtm.datasets.base import DatasetProtocol, DataStatistics
from research.mtm.tokenizers.base import TokenizerManager
import os
import glob


def load_episodes(fn):
    print(f"Loading episodes from {fn}")
    with fn.open("rb") as f:
        episodes = pickle.load(f)
        return episodes
    
def get_datasets(
    dataset_path: Union[str, Path],
    seq_steps: int = 4,
    discount: float = 0.99,
):
    """
    **Unfinished**
    Get the training and validation datasets for softgym environment.
    """
    # home = Path.home()
    # dataset_path = home / "softagent/data/curl/data.pkl"
    if not isinstance(dataset_path, Path):
        # join the path
        dataset_path = Path(dataset_path)
    train_dataset = SoftgymDataset(
        dataset_path=Path.joinpath(dataset_path, 'train_dataset'),
        traj_length=seq_steps,
        discount=discount,
    )
    val_dataset = SoftgymDataset(
        dataset_path=Path.joinpath(dataset_path, 'val_dataset'),
        traj_length=seq_steps,
        discount=discount,
    )
    return train_dataset, val_dataset



class SoftgymDataset(Dataset, DatasetProtocol):
    """

    """
    def __init__(
        self,
        dataset_path: str,
        traj_length: int = 4,
        discount: float = 0.99,
    ):
        self._traj_length = traj_length

        data_objs = glob.glob(str(dataset_path) + '/*.pkl')
        merged_data_obj = {}
        for data_obj in data_objs:
            data_pickle_obj = load_episodes(Path(data_obj))
            for key in data_pickle_obj:
                if key not in merged_data_obj:
                    merged_data_obj[key] = data_pickle_obj[key]
                else:
                    merged_data_obj[key].extend(data_pickle_obj[key])

        # compute path lengths
        self._path_lengths = [len(episode) for episode in merged_data_obj['rewards']]
        self.max_path_length = max(self._path_lengths)
        # check that all path lengths are the same -- since later we will stack them into numpy arrays
        assert all(
            [path_len == self.max_path_length for path_len in self._path_lengths]
        )

        # data buffer structures:
        # prepare actions, states, rewards, returns
        self.actions = np.array(merged_data_obj['actions']) # [n_paths x n_timesteps x action_dim]
        self.states = np.array(merged_data_obj['states'])
        self.rewards = np.array(merged_data_obj['rewards'])
        self.keypoints = np.array(merged_data_obj['keypoints'])
        # TODO: more modality? such as depth?


        self.returns = np.zeros(self.rewards.shape)

        if discount > 1.0: # no discount, and the return for each timestep is the mean of rewards from that timestep onwards
            self.discount = 1.0
            self.use_avg = True
        else:
            self.discount = discount # with discount, the return for each timestep is the sum of discounted rewards from that timestep onwards
            self.use_avg = False
        discounts = (self.discount ** np.arange(self.max_path_length))[None, :]
        for t in range(self.max_path_length):
            ## [ n_paths x 1 ]

            if False:
                ret = (self.rewards[:, t + 1 :] * discounts[:, : -t - 1]).sum(axis=1)
            else:
                dis = discounts if t == 0 else discounts[:, :-t]
                ret = (self.rewards[:, t:] * dis).sum(axis=1)
            self.returns[:, t] = ret
        _, Max_Path_Len = self.returns.shape
        if self.use_avg:
            divisor = np.arange(1, Max_Path_Len + 1)[::-1][None, :]
            self.returns = self.returns / divisor

        # create index map; map from an integer sample index to a tuple of (traj_index, start_timestep_index)
        index_map = {}
        count = 0
        traj_count = 0
        for idx, pl in enumerate(self._path_lengths):
            for i in range(pl - self._traj_length + 1):
                index_map[count] = (traj_count, i)
                count += 1
            traj_count += 1
        self.index_map = index_map

        self.num_trajectories = self.returns.shape[0]
        self.returns = self.returns[..., None]
        self.rewards = self.rewards[..., None]


    def trajectory_statistics(self) -> Dict[str, DataStatistics]:

        return {
            "actions": DataStatistics(
                mean=self.actions.mean(axis=(0, 1), keepdims=True),
                std=self.actions.std(axis=(0, 1), keepdims=True),
                min=self.actions.min(axis=(0, 1), keepdims=True),
                max=self.actions.max(axis=(0, 1), keepdims=True),
            ),
            "rewards": DataStatistics(
                mean=self.rewards.mean(axis=(0, 1), keepdims=True),
                std=self.rewards.std(axis=(0, 1), keepdims=True),
                min=self.rewards.min(axis=(0, 1), keepdims=True),
                max=self.rewards.max(axis=(0, 1), keepdims=True),
            ),
            "returns": DataStatistics(
                mean=self.returns.mean(axis=(0, 1), keepdims=True),
                std=self.returns.std(axis=(0, 1), keepdims=True),
                min=self.returns.min(axis=(0, 1), keepdims=True),
                max=self.returns.max(axis=(0, 1), keepdims=True),
            ),
        }
        # raise NotImplementedError("This function should not be called. \
        #                           This function needs an override implementation by the requirements of the base class. \
        #                           The statistics of the trajectories are typically used for normalization purposes, \
        #                           which is not needed for this dataset. ")


    def __len__(self) -> int:
        # return self.num_trajectories
        return len(self.index_map)

    def get_trajectory(self, traj_index: int) -> Dict[str, np.ndarray]:
        return {
            "states": self.states[traj_index],
            "actions": self.actions[traj_index],
            "rewards": self.rewards[traj_index],
            "returns": self.returns[traj_index],
        }

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """Return a trajectories of the form (observations, actions, rewards, values).

        A random trajectory with self.sequence_length is returned.
        """
        idx, start_idx = self.index_map[index]
        traj = self.get_trajectory(idx)
        return {
            k: v[start_idx : start_idx + self._traj_length] for k, v in traj.items()
        }

    def eval_logs(
        self, model: Callable, tokenizer_manager: TokenizerManager
    ) -> Dict[str, Any]:
        raise NotImplementedError("The eval_log function is not implemented for SoftgymDataset!")


def main():

    home = Path.home()
    dataset_path = home / "softagent/data/curl/data.pkl"
    dataset = SoftgymDataset(
        dataset_path=dataset_path,
        traj_length=4,
        discount=0.99,
    )

    print(len(dataset))
    print(dataset[0])


if __name__ == "__main__":
    main()

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

from sklearn.cluster import KMeans


def load_episodes(fn):
    print(f"Loading episodes from {fn}")
    with fn.open("rb") as f:
        episodes = pickle.load(f)
        return episodes
    
def get_datasets(
    dataset_path: Union[str, Path],
    seq_steps: int = 4,
    discount: float = 0.99,
    filter_trajectories: bool = False,
):
    """
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
        filter_trajectories=filter_trajectories,
    )
    val_dataset = SoftgymDataset(
        dataset_path=Path.joinpath(dataset_path, 'val_dataset'),
        traj_length=seq_steps,
        discount=discount,
        filter_trajectories=filter_trajectories,    
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
        states_type: Union["keypoints", "rgb", "depth"]= "rgb",
        actions_type: Union["continuous", "discrete"] = "continuous",
        n_clusters: int = 32,
        filter_trajectories: bool = True, # whether filter out the segments with no rewards
    ):

        assert states_type in ["keypoints", "rgb", "depth"], "states_type should be one of keypoints, rgb, depth"
        self.states_type = states_type
        self.actions_type = actions_type

        self._traj_length = traj_length
        self.n_clusters = n_clusters

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
        self.raw_actions = np.array(merged_data_obj['actions']) # [n_paths x n_timesteps x action_dim]
        self.rgb = np.array(merged_data_obj['states'])
        self.rewards = np.array(merged_data_obj['rewards'])
        self.keypoints = np.array(merged_data_obj['keypoints'])
        self.depths = np.array(merged_data_obj['depths'])
        # TODO: more modality? such as depth?

        # process the grasp flag of the actions
        self.grasp_flags = self.raw_actions[..., -1]
        new_flags = np.ones(self.grasp_flags.shape)
        new_flags[self.grasp_flags < 0] = -1
        self.raw_actions[..., -1] = new_flags

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

        for idx, pl in enumerate(self._path_lengths):
            for i in range(pl - self._traj_length + 1):
                if filter_trajectories and self.rewards[idx, i : i + self._traj_length].sum() == 0:
                    continue
                index_map[count] = (idx, i)
                count += 1
        self.index_map = index_map

        self.num_trajectories = self.returns.shape[0]
        self.returns = self.returns[..., None]
        self.rewards = self.rewards[..., None]

        # for discretization of actions, use kmeans to cluster the actions
        self.cluster_actions()

    def filter_trajectories(self) -> None:
        "We can filter the trajectories, probably based on stationarity, or low returns, etc."
        pass

    # @property
    # def env(self) -> None:
    #     return None
    #     # to be compatible wit

    def cluster_actions(self, n_clusters: int = 32) -> None:
        """
        Cluster the actions into n_clusters clusters using KMeans.
        """
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(self.raw_actions.reshape(-1, self.raw_actions.shape[-1]))
        self.actions_centers = self.kmeans.cluster_centers_
        self.actions_labels = self.kmeans.labels_.reshape(self.raw_actions.shape[:-1])
        return self.kmeans


    def trajectory_statistics(self) -> Dict[str, DataStatistics]:

        ret = {
            "actions": DataStatistics(
                mean=self.raw_actions.mean(axis=None, keepdims=True),
                std=self.raw_actions.std(axis=None, keepdims=True),
                min=self.raw_actions.min(axis=None, keepdims=True),
                max=self.raw_actions.max(axis=None, keepdims=True),
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
            "keypoints": DataStatistics(
                mean=self.keypoints.mean(axis=None, keepdims=True),
                std=self.keypoints.std(axis=None, keepdims=True),
                min=self.keypoints.min(axis=None, keepdims=True),
                max=self.keypoints.max(axis=None, keepdims=True),
            ),
        }
        if self.states_type == "keypoints":
            ret["states"] = ret["keypoints"]
        return ret


    def __len__(self) -> int:
        # return self.num_trajectories
        return len(self.index_map)
    
    @property
    def states(self) -> np.ndarray:
        if self.states_type == "keypoints":
            return self.keypoints
        elif self.states_type == "rgb":
            return self.rgb
        elif self.states_type == "depth":
            return self.depths
        else:
            raise ValueError("states_type should be one of keypoints, rgb, depth")
        
    @property
    def actions(self) -> np.ndarray:
        if self.actions_type == "continuous":
            return self.raw_actions
        else:
            raise ValueError("actions_type should be continuous")
        # elif self.actions_type == "discrete":
        #     return self.actions_labels

    def get_trajectory(self, traj_index: int) -> Dict[str, np.ndarray]:
        return {
            "states": self.states[traj_index],
            "actions": self.actions[traj_index],
            "rewards": self.rewards[traj_index],
            "returns": self.returns[traj_index],
            "keypoints": self.keypoints[traj_index],
            "depths": self.depths[traj_index],
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

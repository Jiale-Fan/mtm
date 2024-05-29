# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.utils.data import Dataset

from research.mtm.tokenizers.base import Tokenizer
from sklearn.cluster import KMeans


class DiscreteActionsTokenizer(Tokenizer):
    """Dummy tokenizer for trajectories that are already discrete."""

    def __init__(self, kmeans_result: KMeans, embedding_dim: int = 512):
        super().__init__()
        self.kmeans_result = kmeans_result
        self.num_classes = kmeans_result.n_clusters
        torch.manual_seed(42)
        self.vocabulary = torch.nn.Embedding(
            self.num_classes, embedding_dim
        )


    @classmethod
    def create(
        cls, key: str, train_dataset: Dataset, embedding_dim: int = 512
    ) -> "DiscreteActionsTokenizer":
        # add some slack
        return cls(train_dataset.kmeans, embedding_dim=embedding_dim)

    @property
    def discrete(self) -> bool:
        return True
    
    def get_one_hot(self, trajectory: torch.Tensor) -> torch.Tensor:
        B, T, _ = trajectory.size()
        labels = self.kmeans_result.predict(trajectory.view(-1, trajectory.size(-1)).cpu().numpy()).reshape(B, T)
        one_hot = torch.nn.functional.one_hot(torch.Tensor(labels).to(torch.long), num_classes=self.num_classes)
        return one_hot.to(torch.float32)

    def encode(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode the actions to discrete embeddings.
        Actions: [batch_size, seq_len] of integers
        return [batch_size, seq_len, 1, embedding_dim]
        """
        B, T, _ = trajectory.size()
        labels = self.kmeans_result.predict(trajectory.view(-1, trajectory.size(-1)).cpu().numpy()).reshape(B, T)
        trajectory_embs = self.vocabulary(torch.Tensor(labels).to(torch.int).to('cuda'))
        assert trajectory_embs.dim() == 3
        return trajectory_embs.unsqueeze(2).to(torch.float32)

    def get_probs(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
            input: trajectory: [batch_size, seq_len, 1, embedding_dim]
            return: probs: [batch_size, seq_len, num_classes]
        """
        assert trajectory.dim() == 4
        assert trajectory.size(2) == 1
        # denormalize trajectory
        trajectory = trajectory.squeeze(2)

        scores = torch.cdist(
            trajectory, self.vocabulary.weight.unsqueeze(0), p=2
        ) # [batch_size, seq_len, num_classes]
        probs = torch.nn.functional.softmax(scores, dim=-1)
        return probs # [batch_size, seq_len, num_classes]
    
    def look_up_token(self, idx: torch.Tensor) -> torch.Tensor:
        idx = idx.to(torch.long).cpu().numpy()
        return self.vocabulary(torch.Tensor(idx).to(torch.int).to('cuda')).to('cpu')

    def decode(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """
            input: trajectory: [batch_size, seq_len, 1, embedding_dim]
            return: probs: [batch_size, seq_len, action_dim]
        """
        probs = self.get_probs(trajectory)
        idx = torch.argmax(probs, dim=-1)
        idx = idx.cpu().numpy()
        return torch.Tensor(self.kmeans_result.cluster_centers_[idx]).to("cuda")

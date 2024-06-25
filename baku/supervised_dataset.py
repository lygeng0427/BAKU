import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class SupervisedDataset(Dataset):
    def __init__(self, episodes, encoder, language_projector, mode='train', device='cuda'):
        self._episodes = episodes
        self.encoder = encoder
        self.language_projector = language_projector
        self.mode = mode
        self.device = device
        self.envs_till_idx = len(self._episodes)
        assert self.envs_till_idx == 4
        self._num_episodes = 0
        for i in range(self.envs_till_idx):
            self._num_episodes += len(self._episodes[i])
        self._train_episodes = len(self._episodes[0]) + len(self._episodes[1])
        self._val_episodes = len(self._episodes[2]) + len(self._episodes[3])
        self._get_max_episode_length()

    def __len__(self):
        if self.mode == 'train':
            return self._train_episodes
        elif self.mode == 'val':
            return self._val_episodes

    def _get_max_episode_length(self):
        self._max_episode_len = 0
        for idx in self._episodes:
            for episode in self._episodes[idx]:
                self._max_episode_len = max(
                    self._max_episode_len,
                    (
                        len(episode["observation"])
                        if not isinstance(episode["observation"], dict)
                        else len(episode["observation"]['pixels'])
                    ),
                )

    def __getitem__(self, idx):
        if self.mode == 'train':
            idx1, idx2 = 0, 1
        elif self.mode == 'val':
            idx1, idx2 = 2, 3
        
        # if idx < len(self._episodes[idx1]):
        #     episode = self._episodes[idx1][idx]
        #     label = 1

        # else:
        #     episode = self._episodes[idx2][idx - len(self._episodes[idx1])]
        #     label = 0

        # Randomly choose between idx1 and idx2
        if random.random() < 0.5:
            if idx < len(self._episodes[idx1]):
                episode = self._episodes[idx1][idx]
                label = 1
            else:
                # If idx is out of bounds for idx1, default to idx2
                episode = self._episodes[idx2][idx - len(self._episodes[idx1])]
                label = 0
        else:
            if idx < len(self._episodes[idx2]):
                episode = self._episodes[idx2][idx]
                label = 0
            else:
                # If idx is out of bounds for idx2, default to idx1
                episode = self._episodes[idx1][idx - len(self._episodes[idx2])]
                label = 1
        
        traj = episode["observation"]['pixels']
        traj = torch.tensor(traj).to(self.device).float()
        traj = traj.permute(0, 3, 1, 2)

        task_emb = episode["task_emb"]
        task_emb = torch.tensor(task_emb).to(self.device).float()

        lang_features = self.language_projector(task_emb[None]).repeat(traj.shape[0], 1)

        traj = self.encoder(traj, lang=lang_features)

        pad = torch.zeros(self._max_episode_len - traj.shape[0], traj.shape[1]).to(self.device)
        traj = torch.cat([pad, traj])

        return traj, label
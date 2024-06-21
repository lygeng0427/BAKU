import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class ContrDataset(Dataset):
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
        
        if idx < len(self._episodes[idx1]):
            episode_1 = self._episodes[idx1][idx]
            random_idx = random.randint(0, len(self._episodes[idx1]) - 1)
            while random_idx == idx:
                random_idx = random.randint(0, len(self._episodes[idx1]) - 1)
            episode_2 = self._episodes[idx1][random_idx]

        else:
            episode_1 = self._episodes[idx2][idx - len(self._episodes[idx1])]
            random_idx = random.randint(0, len(self._episodes[idx2]) - 1)
            while random_idx == idx - len(self._episodes[idx1]):
                random_idx = random.randint(0, len(self._episodes[idx2]) - 1)
            episode_2 = self._episodes[idx2][random_idx]
        
        traj1 = episode_1["observation"]['pixels']
        traj2 = episode_2["observation"]['pixels']
        traj1 = torch.tensor(traj1).to(self.device).float()
        traj2 = torch.tensor(traj2).to(self.device).float()
        traj1 = traj1.permute(0, 3, 1, 2)
        traj2 = traj2.permute(0, 3, 1, 2)

        task_emb1 = episode_1["task_emb"]
        task_emb1 = torch.tensor(task_emb1).to(self.device).float()
        task_emb2 = episode_2["task_emb"]
        task_emb2 = torch.tensor(task_emb2).to(self.device).float()

        lang_features1 = self.language_projector(task_emb1[None]).repeat(traj1.shape[0], 1)
        lang_features2 = self.language_projector(task_emb2[None]).repeat(traj2.shape[0], 1)

        traj1 = self.encoder(traj1, lang=lang_features1)
        traj2 = self.encoder(traj2, lang=lang_features2)

        pad1 = torch.zeros(self._max_episode_len - traj1.shape[0], traj1.shape[1]).to(self.device)
        traj1 = torch.cat([pad1, traj1])
        pad2 = torch.zeros(self._max_episode_len - traj2.shape[0], traj2.shape[1]).to(self.device)
        traj2 = torch.cat([pad2, traj2])

        return traj1, traj2


class InfoNCEDataset(Dataset):
    def __init__(self, episodes, encoder, language_projector, mode='train', device='cuda', negatives_num=5):
        self._episodes = episodes
        self.encoder = encoder
        self.language_projector = language_projector
        self.negatives_num = negatives_num
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
            if idx < len(self._episodes[0]):
                query = self._episodes[0][idx]
                positive_key = random.choice(self._episodes[0])
                negative_keys = random.sample(self._episodes[1], self.negatives_num)
            
            else:
                query = self._episodes[1][idx - len(self._episodes[0])]
                positive_key = random.choice(self._episodes[1])
                negative_keys = random.sample(self._episodes[0], self.negatives_num)

            query_image = query["observation"]['pixels']
            positive_image = positive_key["observation"]['pixels']
            negative_images = [negative_key["observation"]['pixels'] for negative_key in negative_keys]
            query_image = torch.tensor(query_image).to(self.device).float()
            positive_image = torch.tensor(positive_image).to(self.device).float()
            negative_images = [torch.tensor(image).to(self.device).float() for image in negative_images]

            query_image = query_image.permute(0, 3, 1, 2)
            positive_image = positive_image.permute(0, 3, 1, 2)
            negative_images = [image.permute(0, 3, 1, 2) for image in negative_images]
            with torch.no_grad():
                query_embeddings = self.encoder(query_image)
                positive_embeddings = self.encoder(positive_image)
                negative_embeddings = [self.encoder(negative) for negative in negative_images]

            pad1 = torch.zeros(self._max_episode_len - query_embeddings.shape[0], query_embeddings.shape[1]).to(self.device)
            query_embeddings = torch.cat([pad1, query_embeddings])
            pad2 = torch.zeros(self._max_episode_len - positive_embeddings.shape[0], positive_embeddings.shape[1]).to(self.device)
            positive_embeddings = torch.cat([pad2, positive_embeddings])
            combined_embeddings = torch.stack([query_embeddings, positive_embeddings], dim=0)

            negative_embeddings = [torch.cat([torch.zeros(self._max_episode_len - negative.shape[0], negative.shape[1]).to(self.device), negative]) for negative in negative_embeddings]
            negative_embeddings = torch.stack(negative_embeddings, dim=0)

            return {
                'traj_1': combined_embeddings, # [2, L, embed_dim]
                'traj_2': negative_embeddings
            }
        
        elif self.mode == 'val':
            label = random.choice([1, 0])
            if idx < len(self._episodes[2]):
                traj1 = self._episodes[2][idx]["observation"]['pixels']
                if label == 0:
                    traj2 = random.choice(self._episodes[2])["observation"]['pixels']
                else:
                    traj2 = random.choice(self._episodes[3])["observation"]['pixels']

            else:
                traj1 = self._episodes[3][idx - len(self._episodes[2])]["observation"]['pixels']
                if label == 0:
                    traj2 = random.choice(self._episodes[3])["observation"]['pixels']
                else:
                    traj2 = random.choice(self._episodes[2])["observation"]['pixels']

            # Turn into embeddings
            traj1 = torch.tensor(traj1).to(self.device).float()
            traj2 = torch.tensor(traj2).to(self.device).float()
            traj1 = traj1.permute(0, 3, 1, 2)
            traj2 = traj2.permute(0, 3, 1, 2)

            traj1 = self.encoder(traj1)
            traj2 = self.encoder(traj2)
            pad1 = torch.zeros(self._max_episode_len - traj1.shape[0], traj1.shape[1]).to(self.device)
            traj1 = torch.cat([pad1, traj1])
            pad2 = torch.zeros(self._max_episode_len - traj2.shape[0], traj2.shape[1]).to(self.device)
            traj2 = torch.cat([pad2, traj2]) # (L, embed_dim)

            return {'traj_1': traj1, 'traj_2': traj2, 'label': label}

import numpy as np
import random
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image
import imageio


class SupervisedDataset(Dataset):
    def __init__(self, cfg, episodes, encoder, language_projector, padding, mode='train', device='cuda'):
        self.cfg = cfg
        self._episodes = episodes
        self.encoder = encoder
        self.language_projector = language_projector
        self.padding = padding
        self.mode = mode
        self.device = device
        self.envs_till_idx = len(self._episodes)
        assert self.envs_till_idx == 4
        self._num_episodes = 0
        for i in range(self.envs_till_idx):
            self._num_episodes += len(self._episodes[i])
        self._train_episodes = len(self._episodes[0]) + len(self._episodes[1])
        self._val_episodes = len(self._episodes[2]) + len(self._episodes[3])
        print(f"Train episodes: {self._train_episodes}, Val episodes: {self._val_episodes}")
        self._get_max_episode_length()
        self.indices = self._create_indices()
        random.shuffle(self.indices)
        self.aug = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )
        # Turn off gradients for encoder and language projector
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.language_projector.parameters():
            param.requires_grad = False

    def __len__(self):
        return len(self.indices)

    def _get_max_episode_length(self):
        self._max_episode_len = 0
        for idx in self._episodes:
            for episode in self._episodes[idx]:
                self._max_episode_len = max(
                    self._max_episode_len,
                    (
                        len(episode["observation"])
                        if not isinstance(episode["observation"], dict)
                        else len(episode["observation"][self.cfg.obs_type])
                    ),
                )

    def _create_indices(self):
        if self.mode == 'train':
            idx1, idx2 = 0, 1
        elif self.mode == 'val':
            idx1, idx2 = 2, 3
        # Create a list of tuples (list_index, episode_index)
        indices = [(idx1, i) for i in range(len(self._episodes[idx1]))] + \
                  [(idx2, i) for i in range(len(self._episodes[idx2]))]
        return indices

    def __getitem__(self, idx):
        # if self.mode == 'train':
        #     idx1, idx2 = 0, 1
        # elif self.mode == 'val':
        #     idx1, idx2 = 2, 3

        # # Randomly choose between idx1 and idx2
        # if random.random() < 0.5:
        #     if idx < len(self._episodes[idx1]):
        #         episode = self._episodes[idx1][idx]
        #         label = 1
        #     else:
        #         # If idx is out of bounds for idx1, default to idx2
        #         episode = self._episodes[idx2][idx - len(self._episodes[idx1])]
        #         label = 0
        # else:
        #     if idx < len(self._episodes[idx2]):
        #         episode = self._episodes[idx2][idx]
        #         label = 0
        #     else:
        #         # If idx is out of bounds for idx2, default to idx1
        #         episode = self._episodes[idx1][idx - len(self._episodes[idx2])]
        #         label = 1

        list_idx, episode_idx = self.indices[idx]
        episode = self._episodes[list_idx][episode_idx]
        label = 1 if list_idx in [0, 2] else 0
        
        traj = episode["observation"]['pixels']
        traj_numpy = traj
        # traj = torch.tensor(traj).to(self.device).float()
        # traj = traj.permute(0, 3, 1, 2)
        # traj = traj / 255.0

        traj = torch.stack(
            [self.aug(traj[i]) for i in range(len(traj))]
        ).to(self.device).float()
        first_frame = traj[0].unsqueeze(0)
        last_frame = traj[-1].unsqueeze(0)

        # # save first frame of traj
        # first_frame = traj[0]
        # save_path = '/home/lgeng/BAKU/baku/imagess/success/' if label == 1 else '/home/lgeng/BAKU/baku/imagess/failure/'
        # # create the dir if not exits
        # os.makedirs(save_path, exist_ok=True)
        # # save the image
        # torchvision.utils.save_image(first_frame, os.path.join(save_path, f'{idx}.png'))

        # # save videos
        # if list_idx == 0:
        #     save_path = '/home/lgeng/BAKU/baku/imagesss/train_success/'
        # elif list_idx == 1:
        #     save_path = '/home/lgeng/BAKU/baku/imagesss/train_failure/'
        # elif list_idx == 2:
        #     save_path = '/home/lgeng/BAKU/baku/imagesss/val_success/'
        # else:
        #     save_path = '/home/lgeng/BAKU/baku/imagesss/val_failure/'
        
        # os.makedirs(save_path, exist_ok=True)
        # video = traj.cpu().detach().numpy() * 255
        # video = video.astype(np.uint8)
        # if video.shape[-1] != 3:
        #     video = video.transpose(0, 2, 3, 1)
        # imageio.mimsave(os.path.join(save_path, f'{idx}.mp4'), video, fps=30)

        task_emb = episode["task_emb"]
        task_emb = torch.tensor(task_emb).to(self.device).float()
        
        with torch.no_grad():
            lang_features = self.language_projector(task_emb[None]).repeat(traj.shape[0], 1)
            traj = self.encoder(traj, lang=lang_features)
            first_frame = self.encoder(first_frame, lang=lang_features[0]).squeeze(0)
            last_frame = self.encoder(last_frame, lang=lang_features[-1]).squeeze(0)

        if self.padding == "first_frame":
            # padding first frame before the trajectory
            first_frames_pad = first_frame.repeat(self._max_episode_len - traj.shape[0], 1)
            traj = torch.cat([first_frames_pad, traj])

        elif self.padding == "last_frame":
            # # padding last frame after the trajectory
            last_frames_pad = last_frame.repeat(self._max_episode_len - traj.shape[0], 1)
            traj = torch.cat([traj, last_frames_pad])
        
        elif self.padding == "zero_padding_before":
            # zero padding before the trajectory
            pad = torch.zeros(self._max_episode_len - traj.shape[0], traj.shape[1]).to(self.device)
            traj = torch.cat([pad, traj])
        
        elif self.padding == "zero_padding_after":
            # zero padding after the trajectory
            pad = torch.zeros(self._max_episode_len - traj.shape[0], traj.shape[1]).to(self.device)
            traj = torch.cat([traj, pad])

        return traj, label, traj_numpy.shape[0]
#!/usr/bin/env python3

import warnings
import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
import wandb
from datetime import date

import utils
from logger import Logger
from replay_buffer import make_expert_replay_loader
from video import VideoRecorder
from rewarder import optimal_transport_plan, cosine_distance, euclidean_distance
from contr_dataset import ContrDataset, InfoNCEDataset
from supervised_dataset import SupervisedDataset
from agent.networks.gpt import GPT, GPTConfig

from info_nce import InfoNCE, info_nce

from moviepy.editor import VideoClip, TextClip, CompositeVideoClip

warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    obs_shape = {}
    for key in cfg.suite.pixel_keys:
        obs_shape[key] = obs_spec[key].shape
    if cfg.use_proprio:
        obs_shape[cfg.suite.proprio_key] = obs_spec[cfg.suite.proprio_key].shape
    obs_shape[cfg.suite.feature_key] = obs_spec[cfg.suite.feature_key].shape
    cfg.agent.obs_shape = obs_shape
    cfg.agent.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg.agent)


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=256,
        dropout=0,
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class CLIPTraj(nn.Module):
    def __init__(
        self,
        embedding_dim=512,
        temperature=1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.actor_1 = GPT(
            GPTConfig(
                block_size=256,
                input_dim=embedding_dim,
                output_dim=embedding_dim,
                n_layer=8,
                n_head=4,
                n_embd=embedding_dim,
                dropout=0.1,
            )
        )
        self.actor_2 = GPT(
            GPTConfig(
                block_size=256,
                input_dim=embedding_dim,
                output_dim=embedding_dim,
                n_layer=8,
                n_head=4,
                n_embd=embedding_dim,
                dropout=0.1,
            )
        )
        self.projection_1 = ProjectionHead(512)
        self.projection_2 = ProjectionHead(512)

    def forward(self, batch):
        # Getting Image and Text Features
        first_features = self.actor_1(batch[0])
        second_features = self.actor_2(batch[1])

        # Take the last token of the sequence
        first_features = first_features[:, -1]
        second_features = second_features[:, -1]

        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.projection_1(first_features)
        text_embeddings = self.projection_2(second_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    
class infoNCETraj(nn.Module):
    def __init__(
        self,
        embedding_dim=512,
    ):
        super().__init__()
        self.encoder = GPT(
            GPTConfig(
                block_size=256,
                input_dim=embedding_dim,
                output_dim=embedding_dim,
                n_layer=8,
                n_head=4,
                n_embd=embedding_dim,
                dropout=0.1,
            )
        )
        self.loss = InfoNCE(negative_mode='paired')

    def forward(self, batch):
        """
        Input:
        traj_embed: (B, P/N, L, embed_dim)
        where B stands for the batch size, L stands for the max_length of the traj, embed_dim stands for the dimension of the model
        """
        # Getting Image and Text Features
        B, P, L, embed_dim = batch['traj_1'].shape
        B, N, L, embed_dim = batch['traj_2'].shape

        traj_1 = batch['traj_1'].view(B * P, L, embed_dim)
        traj_2 = batch['traj_2'].view(B * N, L, embed_dim)

        first_features = self.encoder(traj_1)[:, -1, :]
        second_features = self.encoder(traj_2)[:, -1, :]

        first_features = first_features.view(B, P, -1)
        second_features = second_features.view(B, N, -1)

        # first_features = self.projection_1(first_features)
        # second_features = self.projection_2(second_features)

        # Calculating the Loss
        query = first_features[:, 0, :]
        positive_key = first_features[:, 1, :]
        negative_keys = second_features

        loss = self.loss(query, positive_key, negative_keys)
        
        return loss

def create_tensor(N, L):
    # Ensure N and L are integers and N <= L
    N = int(N)
    L = int(L)
    if N > L:
        raise ValueError("N should not be greater than L")

    # Create a tensor of zeros with length L
    tensor = torch.zeros(L)
    
    # Fill the first N elements with ones
    tensor[:N] = 1
    
    return tensor

class supervisedTraj(nn.Module):
    def __init__(
        self,
        embedding_dim=512,
    ):
        super().__init__()
        self.encoder = GPT(
            GPTConfig(
                block_size=256,
                input_dim=embedding_dim,
                output_dim=embedding_dim,
                n_layer=2,
                n_head=4,
                n_embd=embedding_dim,
                dropout=0.1,
            )
        )
        self.projection = nn.Linear(embedding_dim, 1)
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, batch):
        traj = batch[0]
        labels = batch[1]
        length = batch[2]
        BS, L, _ = traj.shape
        masks = torch.stack([create_tensor(length[i].item(), L) for i in range(len(length))], dim=0).to(traj.device)

        features = self.encoder(traj)
        logits = self.projection(features).squeeze(-1)
        labels = labels.to(logits.device).float().unsqueeze(-1)
        labels = labels.repeat(1, L)

        element_wise_loss = self.criterion(logits, labels)
        masked_loss = element_wise_loss * masks
        loss = masked_loss.sum() / masks.sum()
        return loss
    
    def show_prob(self, batch):
        traj = batch[0]
        labels = batch[1]

        features = self.encoder(traj)
        logits = self.projection(features).squeeze(-1)
        probs = torch.sigmoid(logits)
        return probs, labels

# class supervisedTraj(nn.Module):
#     def __init__(
#         self,
#         embedding_dim=512,
#     ):
#         super().__init__()
#         self.encoder = GPT(
#             GPTConfig(
#                 block_size=256,
#                 input_dim=embedding_dim,
#                 output_dim=embedding_dim,
#                 n_layer=2,
#                 n_head=4,
#                 n_embd=embedding_dim,
#                 dropout=0.1,
#             )
#         )
#         self.projection = nn.Linear(embedding_dim, 1)
#         self.criterion = nn.BCEWithLogitsLoss()

#     def forward(self, batch):
#         traj = batch[0]
#         labels = batch[1]
#         B, L, embed_dim = traj.shape

#         features = self.encoder(traj)
#         logits = self.projection(features).squeeze(-1)
#         labels = labels.to(logits.device).float().unsqueeze(-1)
#         labels = labels.repeat(1, L)
#         loss = self.criterion(logits, labels) / L
#         return loss
    
#     def show_prob(self, batch):
#         traj = batch[0]
#         labels = batch[1]

#         features = self.encoder(traj)
#         logits = self.projection(features).squeeze(-1)
#         probs = torch.sigmoid(logits)
#         return probs, labels

class WorkspaceIL:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # load data
        dataset_iterable = hydra.utils.call(self.cfg.expert_dataset)
        self.expert_replay_loader = make_expert_replay_loader(
            dataset_iterable, self.cfg.batch_size
        )
        self.expert_replay_iter = iter(self.expert_replay_loader)
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        self.cfg.suite.task_make_fn.max_episode_len = (
            self.expert_replay_loader.dataset._max_episode_len
        )
        self.cfg.suite.task_make_fn.max_state_dim = (
            self.expert_replay_loader.dataset._max_state_dim
        )
        if self.cfg.suite.name == "dmc":
            self.cfg.suite.task_make_fn.max_action_dim = (
                self.expert_replay_loader.dataset._max_action_dim
            )
        self.env, self.task_descriptions = hydra.utils.call(self.cfg.suite.task_make_fn)

        # create agent
        self.agent = make_agent(
            self.env[0].observation_spec(), self.env[0].action_spec(), cfg
        )

        self.envs_till_idx = len(self.env)
        self.expert_replay_loader.dataset.envs_till_idx = self.envs_till_idx
        self.expert_replay_iter = iter(self.expert_replay_loader)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None
        )
        self.CLIPTraj = CLIPTraj().to(self.device)
        self.infoNCETraj = infoNCETraj().to(self.device)
        self.supervisedTraj = supervisedTraj().to(self.device)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.suite.action_repeat

    def compute_distance(self, traj1, traj2, task_emb1, task_emb2, metric):
        self.agent.train(False)

        traj1 = torch.tensor(traj1).to(self.device).float()
        traj2 = torch.tensor(traj2).to(self.device).float()
        traj1 = traj1.permute(0, 3, 1, 2)
        traj2 = traj2.permute(0, 3, 1, 2)
        
        # encode task emb
        task_emb1 = torch.tensor(task_emb1).to(self.device).float()
        lang_features1 = self.agent.language_projector(task_emb1[None]).repeat(traj1.shape[0], 1)
        task_emb2 = torch.tensor(task_emb2).to(self.device).float()
        lang_features2 = self.agent.language_projector(task_emb2[None]).repeat(traj2.shape[0], 1)
        
        # encoder traj
        traj1 = self.agent.encoder(traj1, lang=lang_features1)
        traj2 = self.agent.encoder(traj2, lang=lang_features2)

        # pad start of shorter traj with zeros if not OT based distance
        if not metric.startswith('sinkhorn'):
            if traj1.shape[0] < traj2.shape[0]:
                pad = torch.zeros(traj2.shape[0] - traj1.shape[0], traj1.shape[1]).to(self.device)
                traj1 = torch.cat([pad, traj1])
            elif traj2.shape[0] < traj1.shape[0]:
                pad = torch.zeros(traj1.shape[0] - traj2.shape[0], traj2.shape[1]).to(self.device)
                traj2 = torch.cat([pad, traj2])

        # Compute distance be tween traj1 and traj2
        if metric == 'cosine':
            dist = 1. - torch.nn.functional.cosine_similarity(traj1, traj2, dim=1)
        elif metric == 'euclidean':
            dist = torch.nn.functional.pairwise_distance(traj1, traj2, p=2)
        elif metric == 'sinkhorn_cosine':
            cost_matrix = cosine_distance(traj1, traj2)
            transport_plan = optimal_transport_plan(
                traj1, traj2, cost_matrix, method='sinkhorn',
                niter=100).float()  # Getting optimal coupling
            dist = torch.diag(torch.mm(transport_plan, cost_matrix.T)).detach().cpu().numpy()
        elif metric == 'sinkhorn_euclidean':
            cost_matrix = euclidean_distance(traj1, traj2)
            transport_plan = optimal_transport_plan(
                traj1, traj2, cost_matrix, method='sinkhorn',
                niter=100).float()  # Getting optimal coupling
            dist = torch.diag(torch.mm(transport_plan, cost_matrix.T)).detach().cpu().numpy()
        else:
            raise NotImplementedError()
        
        return dist
    
    def compare(self):
        # compare two trajectories
        # similar traj lead to higher similarity, which is lower distance
        self.agent.train(False)
        task_idx1, task_idx2 = 0, 1
        traj_idx1, traj_idx2 = 10, 4
        metric = 'cosine' # cosine, euclidean, sinkhorn_cosine, sinkhorn_euclidean

        traj1 = self.expert_replay_loader.dataset._episodes[task_idx1][traj_idx1]['observation'][self.cfg.obs_type]
        traj2 = self.expert_replay_loader.dataset._episodes[task_idx2][traj_idx2]['observation'][self.cfg.obs_type]
        task_emb1 = self.expert_replay_loader.dataset._episodes[task_idx1][traj_idx1]['task_emb']
        task_emb2 = self.expert_replay_loader.dataset._episodes[task_idx2][traj_idx2]['task_emb']

        dist = self.compute_distance(traj1, traj2, task_emb1, task_emb2, metric)
        print(dist.mean().item())

    def compare_all(self):
        # compare all trajectories
        # similar traj lead to higher similarity, which is lower distance
        # save all distances and gt distances in a csv file
        df = pd.DataFrame(columns=["Distance", "GT Distance"])

        self.agent.train(False)
        episodes = self.expert_replay_loader.dataset._episodes
        for i in range(len(episodes[0])):
            gt_distance = random.randint(0, 1)
            if gt_distance == 0:
                task_idx1, task_idx2 = 0, 0
                traj_idx1, traj_idx2 = i, random.randint(0, len(episodes[0]) - 1)
            else:
                task_idx1, task_idx2 = 0, 1
                traj_idx1, traj_idx2 = i, random.randint(0, len(episodes[1]) - 1)
            metric = 'cosine' # cosine, euclidean, sinkhorn_cosine, sinkhorn_euclidean
            traj1 = episodes[task_idx1][traj_idx1]['observation'][self.cfg.obs_type]
            traj2 = episodes[task_idx2][traj_idx2]['observation'][self.cfg.obs_type]
            task_emb1 = episodes[task_idx1][traj_idx1]['task_emb']
            task_emb2 = episodes[task_idx2][traj_idx2]['task_emb']

            dist = self.compute_distance(traj1, traj2, task_emb1, task_emb2, metric)
            df = df.append({"Distance": dist.mean().item(), "GT Distance": gt_distance}, ignore_index=True)
            print(f"Distance: {dist.mean().item()}")
            print(f"GT Distance: {gt_distance}")

        for i in range(len(episodes[1])):
            # repeat
            gt_distance = random.randint(0, 1)
            if gt_distance == 0:
                task_idx1, task_idx2 = 1, 1
                traj_idx1, traj_idx2 = i, random.randint(0, len(episodes[1]) - 1)
            else:
                task_idx1, task_idx2 = 1, 0
                traj_idx1, traj_idx2 = i, random.randint(0, len(episodes[0]) - 1)
            metric = 'cosine' # cosine, euclidean, sinkhorn_cosine, sinkhorn_euclidean
            traj1 = episodes[task_idx1][traj_idx1]['observation'][self.cfg.obs_type]
            traj2 = episodes[task_idx2][traj_idx2]['observation'][self.cfg.obs_type]
            task_emb1 = episodes[task_idx1][traj_idx1]['task_emb']
            task_emb2 = episodes[task_idx2][traj_idx2]['task_emb']

            dist = self.compute_distance(traj1, traj2, task_emb1, task_emb2, metric)
            df = df.append({"Distance": dist.mean().item(), "GT Distance": gt_distance}, ignore_index=True)
            print(f"Distance: {dist.mean().item()}")
            print(f"GT Distance: {gt_distance}")
        df.to_csv(self.work_dir / 'distances.csv', index=False)

    def contr(self, epochs=10):
        encoder = self.agent.encoder
        language_projector = self.agent.language_projector

        contr_train_dataset = ContrDataset(
            self.expert_replay_loader.dataset._episodes, encoder, language_projector, 'train', self.device
        )
        contr_val_dataset = ContrDataset(
            self.expert_replay_loader.dataset._episodes, encoder, language_projector, 'val', self.device
        )
        train_loader = torch.utils.data.DataLoader(
            contr_train_dataset,
            batch_size=2,
            num_workers=0,
        )
        val_loader = torch.utils.data.DataLoader(
            contr_val_dataset,
            batch_size=2,
            num_workers=0,
        )

        optimizer = torch.optim.Adam(self.CLIPTraj.parameters())
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode="min", patience=2, factor=0.5
        # )
        for i in range(epochs):
            print(f"Epoch {i + 1} / {epochs}")
            self.CLIPTraj.train()
            train_tqdm = tqdm(train_loader, total=len(train_loader), desc="Training")
            train_loss_meter = utils.AvgMeter()

            for batch in train_tqdm:
                optimizer.zero_grad()
                loss = self.CLIPTraj(batch)
                loss.backward()
                optimizer.step()

                count = batch[0].shape[0]
                train_loss_meter.update(loss.item(), count)
                train_tqdm.set_postfix(loss=train_loss_meter.avg)

            self.CLIPTraj.eval()
            val_loss_meter = utils.AvgMeter()
            val_tqdm = tqdm(val_loader, total=len(val_loader), desc="Validation")

            with torch.no_grad():
                for batch in val_tqdm:
                    val_loss = self.CLIPTraj(batch)
                    count = batch[0].shape[0]
                    val_loss_meter.update(val_loss.item(), count)
                    val_tqdm.set_postfix(loss=val_loss_meter.avg)

    def infoNCE(self, epochs=10):
        encoder = self.agent.encoder
        language_projector = self.agent.language_projector

        contr_train_dataset = InfoNCEDataset(
            self.expert_replay_loader.dataset._episodes, encoder, language_projector, 'train', self.device
        )
        contr_val_dataset = InfoNCEDataset(
            self.expert_replay_loader.dataset._episodes, encoder, language_projector, 'val', self.device
        )
        train_loader = torch.utils.data.DataLoader(
            contr_train_dataset,
            batch_size=2,
            num_workers=0,
        )
        val_loader = torch.utils.data.DataLoader(
            contr_val_dataset,
            batch_size=2,
            num_workers=0,
        )

        optimizer = torch.optim.Adam(self.CLIPTraj.parameters())
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2, factor=0.5
        )
        for i in range(epochs):
            print(f"Epoch {i + 1} / {epochs}")
            train_tqdm = tqdm(train_loader, total=len(train_loader), desc="Training")
            train_loss_meter = utils.AvgMeter()

            for batch in train_tqdm:
                optimizer.zero_grad()
                loss = self.infoNCETraj(batch)
                loss.backward()
                optimizer.step()

                lr_scheduler.step(loss)

                count = batch["traj_1"].shape[0]
                train_loss_meter.update(loss.item(), count)
                train_tqdm.set_postfix(loss=train_loss_meter.avg)

            self.infoNCETraj.eval()
            val_loss_meter = utils.AvgMeter()
            val_tqdm = tqdm(val_loader, total=len(val_loader), desc="Validation")
            
            df = pd.DataFrame(columns=["Distance", "GT Distance"])
            with torch.no_grad():
                for batch in val_tqdm:
                    feature_1 = self.infoNCETraj.encoder(batch['traj_1'])[:, -1]
                    feature_2 = self.infoNCETraj.encoder(batch['traj_2'])[:, -1]

                    dist = 1. - torch.nn.functional.cosine_similarity(feature_1, feature_2, dim=1).cpu()
                    gt_dist = batch['label']
                    # print(dist, gt_dist)
                    for j in range(len(dist)):
                        df = df.append({"Distance": dist[j].item(), "GT Distance": gt_dist[j].item()}, ignore_index=True) 
                df.to_csv(self.work_dir / 'features_distances.csv', index=False)

    def supervised_train(self):
        wandb.init(project=f"baku_libero_traj{date.today()}", entity="lg3490", name=f'nheads_4_layers_2_bs_16_lr_0.01_{self.cfg.padding}_newLoss_oldData')
        config = wandb.config
        config.lr = 0.01
        config.epochs = 100
        config.bs = 16
        config.step_size = 6

        # Rest of the code goes here
        encoder = self.agent.encoder
        language_projector = self.agent.language_projector

        contr_train_dataset = SupervisedDataset(
            self.cfg, self.expert_replay_loader.dataset._episodes, encoder, language_projector, self.cfg.padding, 'train', self.device
        )
        contr_val_dataset = SupervisedDataset(
            self.cfg, self.expert_replay_loader.dataset._episodes, encoder, language_projector, self.cfg.padding, 'val', self.device
        )
        train_loader = torch.utils.data.DataLoader(
            contr_train_dataset,
            batch_size=config.bs,
            num_workers=0,
        )
        val_loader = torch.utils.data.DataLoader(
            contr_val_dataset,
            batch_size=config.bs,
            num_workers=0,
        )

        optimizer = torch.optim.Adam(self.supervisedTraj.parameters(), lr=config.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=0.1)
        min_val_loss = float('inf')

        for i in range(config.epochs):
            print(f"Epoch {i + 1} / {config.epochs}")
            self.supervisedTraj.train()
            train_tqdm = tqdm(train_loader, total=len(train_loader), desc="Training")
            train_loss_meter = utils.AvgMeter()

            for batch in train_tqdm:
                optimizer.zero_grad()
                loss = self.supervisedTraj(batch)
                loss.backward()
                optimizer.step()

                count = batch[0].shape[0]
                train_loss_meter.update(loss.item(), count)
                train_tqdm.set_postfix(loss=train_loss_meter.avg)

            wandb.log({"train_loss": train_loss_meter.avg}, step=i)
            lr_scheduler.step()

            self.supervisedTraj.eval()
            val_loss_meter = utils.AvgMeter()
            val_tqdm = tqdm(val_loader, total=len(val_loader), desc="Validation")

            with torch.no_grad():
                for batch in val_tqdm:
                    val_loss = self.supervisedTraj(batch)
                    count = batch[0].shape[0]
                    val_loss_meter.update(val_loss.item(), count)
                    val_tqdm.set_postfix(loss=val_loss_meter.avg)

                wandb.log({"val_loss": val_loss_meter.avg}, step=i)

            # if val_loss_meter.avg < min_val_loss:
            #     min_val_loss = val_loss_meter.avg
            #     torch.save(self.supervisedTraj.state_dict(), self.work_dir / 'supervised_model.pth')
            torch.save(self.supervisedTraj.state_dict(), self.work_dir / f'supervised_model_{i+1}.pth')

        wandb.finish()
    
    def supervised_eval(self, model_path):
        encoder = self.agent.encoder
        language_projector = self.agent.language_projector

        contr_train_dataset = SupervisedDataset(
            self.cfg, self.expert_replay_loader.dataset._episodes, encoder, language_projector, self.cfg.padding, 'train', self.device
        )
        contr_val_dataset = SupervisedDataset(
            self.cfg, self.expert_replay_loader.dataset._episodes, encoder, language_projector, self.cfg.padding, 'val', self.device
        )
        train_loader = torch.utils.data.DataLoader(
            contr_train_dataset,
            batch_size=8,
            num_workers=0
        )
        val_loader = torch.utils.data.DataLoader(
            contr_val_dataset,
            batch_size=8,
            num_workers=0,
        )

        self.supervisedTraj.load_state_dict(torch.load(model_path))
        self.supervisedTraj.eval()
        # val_tqdm = tqdm(val_loader, total=len(val_loader), desc="Validation")
        df = pd.DataFrame(columns=["Probs", "Labels"])
        train_result_df = pd.DataFrame(columns=["Probs", "Labels"])
        
        with torch.no_grad():
            for batch in train_loader:
                probs, labels = self.supervisedTraj.show_prob(batch)
                for i in range(len(probs)):
                    # print(f"Prob: {probs_formatted}")
                    # print(f"GT: {labels[i].item()}")
                    traj_length = batch[2][i]
                    prob = probs[i].tolist()
                    if self.cfg.padding == "first_frame" or self.cfg.padding == "zero_padding_before":
                        prob = prob[-traj_length:]
                    elif self.cfg.padding == "last_frame" or self.cfg.padding == "zero_padding_after":
                        prob = prob[:traj_length]

                    train_result_df = train_result_df.append({"Probs": prob, "Labels": labels[i].item()}, ignore_index=True)
            train_result_df.to_csv(self.work_dir / 'train_probs.csv', index=False)

            for batch in val_loader:
                probs, labels = self.supervisedTraj.show_prob(batch)
                for i in range(len(probs)):
                    # print(f"Prob: {probs_formatted}")
                    # print(f"GT: {labels[i].item()}")
                    traj_length = batch[2][i]
                    prob = probs[i].tolist()
                    if self.cfg.padding == "first_frame" or self.cfg.padding == "zero_padding_before":
                        prob = prob[-traj_length:]
                    elif self.cfg.padding == "last_frame" or self.cfg.padding == "zero_padding_after":
                        prob = prob[:traj_length]
                    
                    df = df.append({"Probs": prob, "Labels": labels[i].item()}, ignore_index=True)

            df.to_csv(self.work_dir / 'probs.csv', index=False)

    def eval(self):
        self.agent.train(False)
        episode_rewards = []
        successes = []
        for env_idx in range(self.envs_till_idx):
            print(f"evaluating env {env_idx}")
            episode, total_reward = 0, 0
            eval_until_episode = utils.Until(self.cfg.suite.num_eval_episodes)
            success = []

            while eval_until_episode(episode):
                time_step = self.env[env_idx].reset()
                self.agent.buffer_reset()
                step = 0

                # prompt
                if self.cfg.prompt != None and self.cfg.prompt != "intermediate_goal":
                    prompt = self.expert_replay_loader.dataset.sample_test(env_idx)
                else:
                    prompt = None

                if episode == 0:
                    self.video_recorder.init(self.env[env_idx], enabled=True)

                # plot obs with cv2
                while not time_step.last():
                    if self.cfg.prompt == "intermediate_goal":
                        prompt = self.expert_replay_loader.dataset.sample_test(
                            env_idx, step
                        )
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        action = self.agent.act(
                            time_step.observation,
                            prompt,
                            self.expert_replay_loader.dataset.stats,
                            step,
                            self.global_step,
                            eval_mode=True,
                        )
                    time_step = self.env[env_idx].step(action)
                    self.video_recorder.record(self.env[env_idx])
                    total_reward += time_step.reward
                    step += 1

                    if self.cfg.suite.name == "calvin" and time_step.reward == 1:
                        self.agent.buffer_reset()

                episode += 1
                success.append(time_step.observation["goal_achieved"])
            self.video_recorder.save(f"{self.global_frame}_env{env_idx}.mp4")
            episode_rewards.append(total_reward / episode)
            successes.append(np.mean(success))

        for _ in range(len(self.env) - self.envs_till_idx):
            episode_rewards.append(0)
            successes.append(0)

        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            for env_idx, reward in enumerate(episode_rewards):
                log(f"episode_reward_env{env_idx}", reward)
                log(f"success_env{env_idx}", successes[env_idx])
            log("episode_reward", np.mean(episode_rewards[: self.envs_till_idx]))
            log("success", np.mean(successes))
            log("episode_length", step * self.cfg.suite.action_repeat / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)

        self.agent.train(True)

    def save_snapshot(self):
        snapshot = self.work_dir / "snapshot.pt"
        self.agent.clear_buffers()
        keys_to_save = ["timer", "_global_step", "_global_episode"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        payload.update(self.agent.save_snapshot())
        with snapshot.open("wb") as f:
            torch.save(payload, f)

        self.agent.buffer_reset()

    def load_snapshot(self, snapshots):
        # bc
        with snapshots["bc"].open("rb") as f:
            payload = torch.load(f)
        agent_payload = {}
        for k, v in payload.items():
            if k not in self.__dict__:
                agent_payload[k] = v
        if "vqvae" in snapshots:
            with snapshots["vqvae"].open("rb") as f:
                payload = torch.load(f)
            agent_payload["vqvae"] = payload
        self.agent.load_snapshot(agent_payload, eval=True)


@hydra.main(config_path="cfgs", config_name="config_traj")
def main(cfg):
    from compare_traj import WorkspaceIL as W

    root_dir = Path.cwd()
    workspace = W(cfg)

    # Load weights
    snapshots = {}
    # bc
    bc_snapshot = Path(cfg.bc_weight)
    if not bc_snapshot.exists():
        raise FileNotFoundError(f"bc weight not found: {bc_snapshot}")
    print(f"loading bc weight: {bc_snapshot}")
    snapshots["bc"] = bc_snapshot
    workspace.load_snapshot(snapshots)

    # workspace.eval()
    # workspace.compare_all()
    # workspace.contr()
    # workspace.infoNCE()
    if cfg.supervised_train:
        workspace.supervised_train()
    else:
        supervised_model_path = "/home/lgeng/BAKU/baku/exp_local/eval/2024.08.07_supervised_train/deterministic/121752_hidden_dim_256/supervised_model_3.pth"
        # works well: /home/lgeng/BAKU/baku/exp_local/eval/2024.07.16_supervised_train/deterministic/135045_hidden_dim_256/supervised_model.pth
        workspace.supervised_eval(supervised_model_path)


if __name__ == "__main__":
    main()
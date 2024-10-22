import importlib
import yaml
#import optuna
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Flatten, Input, Dense, concatenate
from sklearn.metrics import accuracy_score, precision_score, recall_score, r2_score, mean_squared_error
from tensorflow.keras.models import Model, Sequential, load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import preprocessing
import dataloader
from preprocessing import preprocessor
from dataloader import dataLoader
import sys
import os
import os.path
import csv
#import ensemRegressor
#import optunatransformator1
import util
#from be_great import GReaT



# current implementation: only support numerical values
import numpy as np
import torch, os
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import math
import argparse

class MaskEmbed(nn.Module):
    """ record to mask embedding
    """
    def __init__(self, rec_len=25, embed_dim=64, norm_layer=None):

        super().__init__()
        self.rec_len = rec_len
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=1, stride=1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, _, L = x.shape
        # assert(L == self.rec_len, f"Input data width ({L}) doesn't match model ({self.rec_len}).")
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class ActiveEmbed(nn.Module):
    """ record to mask embedding
    """
    def __init__(self, rec_len=25, embed_dim=64, norm_layer=None):

        super().__init__()
        self.rec_len = rec_len
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=1, stride=1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, _, L = x.shape
        # assert(L == self.rec_len, f"Input data width ({L}) doesn't match model ({self.rec_len}).")
        x = self.proj(x)
        x = torch.sin(x)
        x = x.transpose(1, 2)
        #   x = torch.cat((torch.sin(x), torch.cos(x + math.pi/2)), -1)
        x = self.norm(x)
        return x



def get_1d_sincos_pos_embed(embed_dim, pos, cls_token=False):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """

    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = np.arange(pos)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    pos_embed = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return pos_embed


def adjust_learning_rate(optimizer, epoch, lr, min_lr, max_epochs, warmup_epochs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        tmp_lr = lr * epoch / warmup_epochs
    else:
        tmp_lr = min_lr + (lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = tmp_lr * param_group["lr_scale"]
        else:
            param_group["lr"] = tmp_lr
    return tmp_lr


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == np.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScaler:

    state_dict_key = "amp_scaler"
    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)



class MAEDataset(Dataset):

    def __init__(self, X, M):
         self.X = X
         self.M = M

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.M[idx]



def get_dataset(dataset : str, path : str):

    if dataset in ['climate', 'compression', 'wine', 'yacht', 'spam', 'letter', 'credit', 'raisin', 'bike', 'obesity', 'airfoil', 'blood', 'yeast', 'health', 'review', 'travel']:
        df = pd.read_csv(os.path.join(path, 'data', dataset + '.csv'))
        last_col = df.columns[-1]
        y = df[last_col]
        X = df.drop(columns=[last_col])
    elif dataset == 'california':
        from sklearn.datasets import fetch_california_housing
        X, y = fetch_california_housing(as_frame=True, return_X_y=True)
    elif dataset == 'diabetes':
        from sklearn.datasets import load_diabetes
        X, y = load_diabetes(as_frame=True, return_X_y=True)
    elif dataset == 'iris':
        # only for testing
        from sklearn.datasets import load_iris
        X, y = load_iris(as_frame=True, return_X_y=True)


    return X, y


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--dataset', default='california', type=str)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--max_epochs', default=600, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--mask_ratio', default=0.5, type=float, help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--embed_dim', default=32, type=int, help='embedding dimensions')
    parser.add_argument('--depth', default=6, type=int, help='encoder depth')
    parser.add_argument('--decoder_depth', default=4, type=int, help='decoder depth')
    parser.add_argument('--num_heads', default=4, type=int, help='number of heads')
    parser.add_argument('--mlp_ratio', default=4., type=float, help='mlp ratio')
    parser.add_argument('--encode_func', default='linear', type=str, help='encoding function')

    parser.add_argument('--norm_field_loss', default=False,
                        help='Use (per-patch) normalized field as targets for computing loss')
    parser.set_defaults(norm_field_loss=False)
    print("Model params initilized")
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')
    print("Optimizer params initilized")
    # Augmentation parameters
    ###### change this path
    parser.add_argument('--path', default='/sample_data/', type=str, help='dataset path')
    parser.add_argument('--exp_name', default='test', type=str, help='experiment name')
    print("Augmentation params initilized")
    # Dataset parameters
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=666, type=int)
    print("Dataset params initilized")
    #
    parser.add_argument('--overwrite', default=True, help='whether to overwrite default config')
    parser.add_argument('--pin_mem', action='store_false')
    print("other params intialized")
    # distributed training parameters
    return parser








######model mae class
# current implementation: only support numerical values

from functools import partial
from tkinter import E

import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import timm
from timm.models.vision_transformer import Block
#from utils import MaskEmbed, get_1d_sincos_pos_embed, ActiveEmbed
eps = 1e-6

class MaskedAutoencoder(nn.Module):

    """ Masked Autoencoder with Transformer backbone
    """

    def __init__(self, rec_len=25, embed_dim=64, depth=4, num_heads=4,
        decoder_embed_dim=64, decoder_depth=2, decoder_num_heads=4,
        mlp_ratio=4., norm_layer=nn.LayerNorm, norm_field_loss=False, encode_func='linear'):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics

        if encode_func == 'active':
            self.mask_embed = ActiveEmbed(rec_len, embed_dim)
        else:
            self.mask_embed = MaskEmbed(rec_len, embed_dim)

        self.rec_len = rec_len
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, rec_len + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)


        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, rec_len + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, 1, bias=True)  # decoder to patch

        # --------------------------------------------------------------------------

        self.norm_field_loss = norm_field_loss
        self.initialize_weights()


    def initialize_weights(self):

        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.mask_embed.rec_len, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.mask_embed.rec_len, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.mask_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def random_masking(self, x, m, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        if self.training:
            len_keep = int(L * (1 - mask_ratio))
        else:
            len_keep = int(torch.min(torch.sum(m, dim=1)))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        noise[m < eps] = 1

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        nask = torch.ones([N, L], device=x.device) - mask

        if self.training:
            mask[m < eps] = 0

        return x_masked, mask, nask, ids_restore


    def forward_encoder(self, x, m, mask_ratio=0.5):

        # embed patches
        x = self.mask_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, nask, ids_restore = self.random_masking(x, m, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x, mask, nask, ids_restore


    def forward_decoder(self, x, ids_restore):

        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)

        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        # x = self.decoder_pred(x)
        x = torch.tanh(self.decoder_pred(x))/2 + 0.5

        # remove cls token
        x = x[:, 1:, :]

        return x


    def forward_loss(self, data, pred, mask, nask):
        """
        data: [N, 1, L]
        pred: [N, L]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        # target = self.patchify(data)
        target = data.squeeze(dim=1)
        if self.norm_field_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + eps)**.5


        loss = (pred.squeeze(dim=2) - target) ** 2
        loss = (loss * mask).sum() / mask.sum()  + (loss * nask).sum() / nask.sum()
        # mean loss on removed patches

        return loss


    def forward(self, data, miss_idx, mask_ratio=0.5):

        latent, mask, nask, ids_restore = self.forward_encoder(data, miss_idx, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(data, pred, mask, nask)
        return loss, pred, mask, nask


def mae_base(**kwargs):
    model = MaskedAutoencoder(
        embed_dim=64, depth=8, num_heads=4,
        decoder_embed_dim=64, decoder_depth=4, decoder_num_heads=4,
        mlp_ratio=2., norm_layer=partial(nn.LayerNorm, eps=eps), **kwargs)
    return model


def mae_medium(**kwargs):
    model = MaskedAutoencoder(
        embed_dim=32, depth=4, num_heads=4,
        decoder_embed_dim=32, decoder_depth=4, decoder_num_heads=4,
        mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=eps), **kwargs)
    return model


def mae_large(**kwargs):
    model = MaskedAutoencoder(
        embed_dim=64, depth=8, num_heads=4,
        decoder_embed_dim=64, decoder_depth=4, decoder_num_heads=4,
        mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=eps), **kwargs)
    return model



###remasker class

# stdlib
from typing import Any, List, Tuple, Union

# third party
import numpy as np
import math, sys, argparse
import pandas as pd
import torch
from torch import nn
from functools import partial
import time, os, json
#from utils import NativeScaler, MAEDataset, adjust_learning_rate, get_dataset
#import model_mae
from torch.utils.data import DataLoader, RandomSampler
import sys
#import timm.optim.optim_factory as optim_factory
#from utils import get_args_parser

# hyperimpute absolute
#from hyperimpute.plugins.imputers import ImputerPlugin
#from sklearn.datasets import load_iris
#from hyperimpute.utils.benchmarks import compare_models
#from hyperimpute.plugins.imputers import Imputers

eps = 1e-8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReMasker:

    def __init__(self):
        """
        print("Entered the Remasker")
        p = get_args_parser()
        print("Parser Called")
        args = p.parse_args()
        print("ARGS")
        print(args)
        print("args func called")
        """
        self.batch_size = 64 #args.batch_size
        self.accum_iter = 1#args.accum_iter
        self.min_lr = 1e-5#args.min_lr
        self.norm_field_loss = False#args.norm_field_loss
        self.weight_decay = 0.05#args.weight_decay
        self.lr = 0.05#args.lr
        self.blr = 1e-3#args.blr
        self.warmup_epochs = 40#args.warmup_epochs
        self.model = None
        self.norm_parameters = None
        print("first phase initialized")
        self.embed_dim = 32#args.embed_dim
        self.depth = 6#args.depth
        self.decoder_depth = 4#args.decoder_depth
        self.num_heads = 4#args.num_heads
        self.mlp_ratio = 4#args.mlp_ratio
        self.max_epochs = 600#args.max_epochs
        self.mask_ratio = 0.5#args.mask_ratio
        self.encode_func = "linear"#args.encode_func

    def fit(self, X_raw: pd.DataFrame):
        X = X_raw.copy(deep=True) #clone()

        # Parameters
        no = len(X)
        dim = len(X.loc[0]) #, :])
        print(f"dim is {dim}")
        X_names = X.columns 
        #X = X.cpu()

        min_val = np.zeros(dim)
        max_val = np.zeros(dim)
        print("Passed the first phase")
        i = 0
        print(X_names)
        for name in X_names:  #for i in range(dim):
            print(f"X[i] = {X[name]}")
            min_val[i] = np.nanmin(X[name]) #(X[:, i])
            max_val[i] = np.nanmax(X[name]) #(X[:, i])
            #X[:, i] = (X[:, i] - min_val[i]) / (max_val[i] - min_val[i] + eps)
            X[name] = (X[name] - min_val[i]) / (max_val[i] - min_val[i] + eps)
            i = i + 1
        self.norm_parameters = {"min": min_val, "max": max_val}
        

        X_ = torch.tensor(X.to_numpy(dtype='f'))
        # Set missing
        M = 1 - (1 * (np.isnan(X_)))
        M = M.float().to(device)

        X_ = torch.nan_to_num(X_)
        X_ = X_.to(device)

        self.model = MaskedAutoencoder(
            rec_len=dim,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            decoder_embed_dim=self.embed_dim,
            decoder_depth=self.decoder_depth,
            decoder_num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            norm_layer=partial(nn.LayerNorm, eps=eps),
            norm_field_loss=self.norm_field_loss,
            encode_func=self.encode_func
        )

        # if self.improve and os.path.exists(self.path):
        #     self.model.load_state_dict(torch.load(self.path))
        #     self.model.to(device)
        #     return self

        self.model.to(device)

        # set optimizers
        # param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
        eff_batch_size = self.batch_size * self.accum_iter
        if self.lr is None:  # only base_lr is specified
            self.lr = self.blr * eff_batch_size / 64
        # param_groups = optim_factory.add_weight_decay(self.model, self.weight_decay)
        # optimizer = torch.optim.AdamW(param_groups, lr=self.lr, betas=(0.9, 0.95))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.95))
        loss_scaler = NativeScaler()

        dataset = MAEDataset(X_, M)
        dataloader = DataLoader(
            dataset, sampler=RandomSampler(dataset),
            batch_size=self.batch_size,
        )

        # if self.resume and os.path.exists(self.path):
        #     self.model.load_state_dict(torch.load(self.path))
        #     self.lr *= 0.5

        self.model.train()

        for epoch in range(self.max_epochs):

            optimizer.zero_grad()
            total_loss = 0

            iter = 0
            for iter, (samples, masks) in enumerate(dataloader):

                # we use a per iteration (instead of per epoch) lr scheduler
                if iter % self.accum_iter == 0:
                    adjust_learning_rate(optimizer, iter / len(dataloader) + epoch, self.lr, self.min_lr,
                                         self.max_epochs, self.warmup_epochs)

                samples = samples.unsqueeze(dim=1)
                samples = samples.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                # print(samples, masks)

                with torch.cuda.amp.autocast():
                    loss, _, _, _ = self.model(samples, masks, mask_ratio=self.mask_ratio)
                    loss_value = loss.item()
                    total_loss += loss_value

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)

                loss /= self.accum_iter
                loss_scaler(loss, optimizer, parameters=self.model.parameters(),
                            update_grad=(iter + 1) % self.accum_iter == 0)

                if (iter + 1) % self.accum_iter == 0:
                    optimizer.zero_grad()

            total_loss = (total_loss / (iter + 1)) ** 0.5
            # if total_loss < best_loss:
            #     best_loss = total_loss
            #     torch.save(self.model.state_dict(), self.path)
            # if (epoch + 1) % 10 == 0 or epoch == 0:
            # print((epoch+1),',', total_loss)

        # torch.save(self.model.state_dict(), self.path)
        return self

    def transform(self, X_raw: torch.Tensor):

        X = X_raw.copy(deep=True) #clone()
        X1 = torch.tensor(X_raw.to_numpy(dtype='f'))

        min_val = self.norm_parameters["min"]
        max_val = self.norm_parameters["max"]

        no, dim = X.shape
        #X = X.cpu()

        # MinMaxScaler normalization
        X_names = X.columns
        for i in range(dim):
            #X[:, i] = (X[:, i] - min_val[i]) / (max_val[i] - min_val[i] + eps)
            X[X_names[i]] = (X[X_names[i]] - min_val[i]) / (max_val[i] - min_val[i] + eps)
        # Set missing
        X_ = torch.tensor(X.to_numpy(dtype='f'))
        M = 1 - (1 * (np.isnan(X_)))
        X = torch.nan_to_num(X_)

        X_ = X_.to(device) #torch.from_numpy(X_).to(device)
        M = M.to(device)

        self.model.eval()

        # Imputed data
        with torch.no_grad():
            for i in range(no):
                sample = torch.reshape(X_[i], (1, 1, -1))
                mask = torch.reshape(M[i], (1, -1))
                _, pred, _, _ = self.model(sample, mask)
                pred = pred.squeeze(dim=2)
                if i == 0:
                    imputed_data = pred
                else:
                    imputed_data = torch.cat((imputed_data, pred), 0)

                    # Renormalize
        for i in range(dim):
            imputed_data[:, i] = imputed_data[:, i] * (max_val[i] - min_val[i] + eps) + min_val[i]

        if np.all(np.isnan(imputed_data.detach().cpu().numpy())):
            err = "The imputed result contains nan. This is a bug. Please report it on the issue tracker."
            raise RuntimeError(err)

        M = M.cpu()
        imputed_data = imputed_data.detach().cpu()
        # print('imputed', imputed_data, M)
        # print('imputed', M * np.nan_to_num(X_raw.cpu()) + (1 - M) * imputed_data)
        return M * np.nan_to_num(X1.cpu()) + (1 - M) * imputed_data

    def get_device(self):
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu') # don't have GPU
        return device

    def df_to_tensor(self, df):
        device = self.get_device()
        return torch.from_numpy(df.values).float().to(device)

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Imputes the provided dataset using the GAIN strategy.
        Args:
            X: np.ndarray
                A dataset with missing values.
        Returns:
            Xhat: The imputed dataset.
        """
        print(X)
        #X = self.df_to_tensor(X)
        X = torch.tensor(X.values, dtype=torch.float32)
        #X1 = X.ravel()
        return self.fit(X).transform(X).detach().cpu().numpy()




class remask_caller():
  def __init__(self, num_of_estimators):
    self.num_of_estimators = int(num_of_estimators)
  def __call__(self, config_yaml, result_path, test_samples, target_app, num_of_frozen_layers, loader, processor,  source_features, source_labels, rank):
    with open(config_yaml, "r") as f:
      global_config= yaml.load(f, Loader=yaml.FullLoader)

    src_tx = loader.src_tx.copy(deep=True)
    tar_tx = loader.tar_tx.copy(deep=True)
    
    csv_path = os.getcwd()+global_config["csv_path"]
    indices_path = os.getcwd()+global_config["indices_path"]
    #callback2 = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=40)
    folder_name = "re_caller"
    list_of_classes = global_config["class_caller"]["list_of_classes"]
    list_of_class_args = global_config["class_caller"]["list_of_class_args"]
    #llm = GReaT(llm='distilgpt2', batch_size=32, epochs=100)
    llm = ReMasker()
    os.environ["WANDB_DISABLED"] = "true"

    """
    tar_x_train, tar_x_test, tar_y_train, tar_y_test = processor.get_train_test_target(test_size = 0.9, rand_state=rank*50)
    tar_to_impute = processor.tar_x_scaled
    cols1 =[]
    cols2 = []
    for i in range(len(source_features[0])):
      cols1.append(f"C{i}")
    """
    src_df = src_tx #pd.DataFrame(source_features, columns=cols1)
    tar_df = tar_tx #pd.DataFrame(tar_to_impute, columns= cols1[0:len(cols1)-1])
    tar_df["GPC"]= float("NAN") #np.nan
    llm.fit(src_df)
    print("Fitting done")
    new_tar_x_test = llm.transform(tar_df)
    print("Imputation done")
    print(new_tar_x_test)
    print(new_tar_x_test.shape)
    print(tar_df)
    tar_df = pd.DataFrame(new_tar_x_test, columns= tar_df.columns)
    print(tar_df)
    
    p = preprocessor(src_df.values, loader.src_y, tar_df.values, loader.tar_y, 0)
    p.setUseCase("remask_caller")
    p.preprocess()
    #tar_x_scaled, tar_y_scaled = p.getTargetScaled()
    X_train, y_train, src_train, src_y_train, src_val, src_y_val, X_test, y_test = p.train_test_val( global_config["test_split"], global_config["val_split"], global_config["rand_state"], global_config["rand_state2"])
    #tar_x_train, tar_x_test, tar_y_train, tar_y_test = processor.get_train_test_target(test_size = 0.9, rand_state=rank*50)

    
    for i in range(rank, rank+1):
      os.makedirs(os.path.dirname(f"{csv_path}{folder_name}/Source-model-on-target-{target_app}-{folder_name}-results-{rank}-MSE.csv"), exist_ok=True)
      os.makedirs(os.path.dirname(f"{csv_path}{folder_name}/Source-model-on-target-{target_app}-{folder_name}-results-{rank}-MAPE.csv"), exist_ok=True)
      fileI = open(f"{csv_path}{folder_name}/Source-model-on-target-{target_app}-{folder_name}-results-{rank}-MSE.csv", "w")
      fileJ = open(f"{csv_path}{folder_name}/Source-model-on-target-{target_app}-{folder_name}-results-{rank}-MAPE.csv", "w")
      writerI = csv.writer(fileI)
      writerJ = csv.writer(fileJ)
      class_number = 0
      tar_x_train, tar_x_test, tar_y_train, tar_y_test = p.get_train_test_target(test_size = 0.9, rand_state=i*50)
      j = 0.1
      if isinstance(j, int):
          fname = f"{j}-samples"
          fidx = j/5
      else:
          fname = f"{j}-percent"
          fidx = int(j*10.0)
      n = j/10
      tar_x_scaled, tar_y_scaled = p.get_tar_train()
      x2, lb2, tar_x_scaled, tar_y_scaled = util.sampleLoader(tar_x_scaled, tar_y_scaled,f"{indices_path}/{target_app}-indices-{rank}-{fname}.csv" ,j)
      print(f"just loaded x2 {x2} for {fname}")
      print(f"lb2 {lb2}")
      print(f"source labels {source_labels}")

      for module_name in list_of_classes:
        module = importlib.import_module(module_name)
        func = getattr(module, module_name)
        obj = func(list_of_class_args[class_number])
        mse0, mape0 = obj(X_train, y_train, x2, lb2, tar_x_test, tar_y_test, rank)  #new_tar_x_test, tar_y_test, rank)
        rowMSE =[]
        rowMAPE = []
        rowMSE.append(module_name)
        rowMSE.append(mse0)
        writerI.writerow(rowMSE)
        rowMAPE.append(module_name)
        rowMAPE.append(mape0)
        writerJ.writerow(rowMAPE)
        class_number = class_number + 1
      fileI.close()
      fileJ.close()

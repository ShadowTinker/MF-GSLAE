# -*- coding: utf-8 -*-
# @Time   : 2022/3/23
# @Author : Gaowei Zhang
# @Email  : 1462034631@qq.com
import dgl
from dgl.nn.pytorch.conv import GATConv, GraphConv

import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender, AutoEncoderMixin
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import EmbLoss, BPRLoss
from recbole.model.layers import MLPLayers, AttLayer
from recbole.utils import InputType
from torch.distributions.binomial import Binomial
from recbole_cdr.model.layers import Encoder, Binarize, TransformerEncoder, MyTrmEncoderLayer

import torch
from torch import nn

import torch
from torch import nn

def InfoNCE(view1, view2, temperature=0.2):
    view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    cl_loss = -torch.log(pos_score / ttl_score)
    return torch.mean(cl_loss)


class MacridDAE32(CrossDomainRecommender, AutoEncoderMixin):
    r"""(27) + domain guided mask

    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(MacridDAE32, self).__init__(config, dataset)

        # load dataset info
        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field
        self.source_interaction_matrix = dataset.inter_matrix(form='coo', value_field=None, domain='source').astype(np.float32)
        self.target_interaction_matrix = dataset.inter_matrix(form='coo', value_field=None, domain='target').astype(np.float32)
        self.convert_sparse_matrix_to_rating_matrix(self.source_interaction_matrix + self.target_interaction_matrix)
        
        # load parameters info
        self.device = config['device']

        # load parameters info
        self.layers = config["mlp_hidden_size"]
        self.lat_dim = config["latent_dimension"]
        self.drop_out = config["dropout_prob"]
        self.cov_reg = config["cov_reg"]


        self.kfac = config["kfac"]
        self.ufac = config["ufac"]
        self.tau = config["tau"]
        self.nogb = config["nogb"]
        self.regs = config["reg_weights"]

        self.epsilon = config["epsilon"]
        self.lmd_disen = config["lmd_disen"]

        # define layers and loss

        self.encode_layer_dims = [self.total_num_items] + self.layers + [self.lat_dim]
        
        self.encoder = nn.Linear(self.total_num_items, self.lat_dim)
        self.encoder_source = nn.Linear(self.total_num_items, self.lat_dim)
        self.encoder_target = nn.Linear(self.total_num_items, self.lat_dim)
        self.item_embedding = nn.Embedding(self.total_num_items, self.lat_dim)
        self.k_embedding = nn.Embedding(self.kfac, self.lat_dim)
        self.k_embedding_source = nn.Embedding(self.kfac, self.lat_dim)
        self.k_embedding_target = nn.Embedding(self.kfac, self.lat_dim)
        self.l2_loss = EmbLoss()

        self.hidden_fourier_filter = Encoder(self.total_num_items, self.lat_dim)
        self.fourier_filter = nn.Parameter(torch.randn(self.total_num_items, 1))
        self.proj_source = nn.Linear(self.lat_dim, self.lat_dim)
        self.proj_target = nn.Linear(self.lat_dim, self.lat_dim)
        self.binary = Binarize.apply
        self.cores_mu = nn.Embedding(self.lat_dim, self.kfac)
        self.cores_logvar = nn.Embedding(self.kfac, self.kfac)

        self.transformer_layer = MyTrmEncoderLayer(
            d_model=self.lat_dim,
            nhead=1,
            dim_feedforward=self.lat_dim * 4,
            dropout=0.5,
            activation=nn.ReLU(),
            layer_norm_eps=1e-12,
            batch_first=True,
            norm_first=False
        )
        # self.mask_source = nn.Sequential(
        #     nn.Linear(self.source_num_items, self.lat_dim),
        #     nn.Dropout(),
        #     nn.Linear(self.lat_dim, self.kfac),
        # )
        # self.mask_target = nn.Sequential(
        #     nn.Linear(self.target_num_items, self.lat_dim),
        #     nn.Dropout(),
        #     nn.Linear(self.lat_dim, self.kfac),
        # )
        self.mask_source = nn.Sequential(
            nn.Linear(self.source_num_items, self.kfac),
        )
        self.mask_target = nn.Sequential(
            nn.Linear(self.target_num_items, self.kfac),
        )
        self.source_mask_budget = 5
        self.target_mask_budget = 5
        # parameters initialization
        # nn.init.xavier_normal_(self.cores_logvar)
        self.apply(xavier_normal_initialization)
        self.get_sparse_norm_rating_matrix()

    def get_sparse_norm_rating_matrix(self):
        self.rating_matrix_source = torch.cat([
            self.rating_matrix[:, :self.overlapped_num_items],
            self.rating_matrix[:, self.target_num_items:]
        ], dim=-1)
        self.rating_matrix_target = self.rating_matrix[:, :self.target_num_items]

    def get_cates(self, cates_logits):
        if self.nogb:
            cates = torch.softmax(cates_logits, dim=-1)
        else:
            cates_sample = F.gumbel_softmax(cates_logits, tau=1, hard=False, dim=-1)
            cates_mode = torch.softmax(cates_logits, dim=-1)
            cates = self.training * cates_sample + (1 - self.training) * cates_mode
        return cates

    def factor_selection(self, h):
        # h_source = self.proj_source(h).reshape(h.shape[0], self.ufac, -1)
        # h_target = self.proj_target(h).reshape(h.shape[0], self.ufac, -1)
        h_source = h.reshape(h.shape[0], self.ufac, -1)
        h_target = h.reshape(h.shape[0], self.ufac, -1)
        # mask_source = self.mask_source.unsqueeze(-1)
        # mask_target = self.mask_target.unsqueeze(-1)
        mask_source = self.binary(self.mask_source).unsqueeze(-1)
        mask_target = self.binary(self.mask_target).unsqueeze(-1)
        # h_source_aug = (h_source * mask_source).flatten(-2, -1)
        # h_target_aug = (h_target * mask_target).flatten(-2, -1)
        h_source_aug = F.normalize((h_source * mask_source).flatten(-2, -1))
        h_target_aug = F.normalize((h_target * mask_target).flatten(-2, -1))
        # h_source_aug = self.source_aug_norm((h_source * mask_source).flatten(-2, -1))
        # h_target_aug = self.target_aug_norm((h_target * mask_target).flatten(-2, -1))
        return h_source_aug, h_target_aug
    
    def get_cores(self):
        mu, log_var = self.cores_mu.weight, torch.tril(self.cores_logvar.weight)
        # mu, log_var = self.cores_mu.weight, self.cores_logvar.weight
        log_var = F.normalize(log_var)
        if self.training:
            std = torch.exp(log_var/2)
            eps = torch.randn_like(mu)
            z = F.normalize((mu + eps @ std)).T
        else:
            z = F.normalize(mu).T
        return z

    def forward(self, source_user=None, target_user=None):
        cores = self.get_cores()
        # cores = F.normalize(self.k_embedding.weight)
        items = F.normalize(self.item_embedding.weight, dim=1)
        items_source = items[self.target_num_items - 1:]
        items_target = items[:self.target_num_items]

        cates_logits = torch.matmul(items, cores.transpose(0, 1)) / self.tau

        cates = self.get_cates(cates_logits)

        # Source channel
        self.h_source_list = []
        self.graph_list_source = []
        self.debug_list = []
        if source_user != None:
            rating_matrix_source = self.rating_matrix[source_user].to(self.device)
            rating_matrix_source = F.normalize(rating_matrix_source)
            rating_matrix_source = F.dropout(rating_matrix_source, self.drop_out, training=self.training)
            probs_source = []

            guidance_source = self.rating_matrix_source[source_user].to(self.device)
            guidance_source = F.normalize(guidance_source)
            guidance_source = F.dropout(guidance_source, self.drop_out, training=self.training)
            mask_source = self.binary(self.mask_source(guidance_source))
            self.mask_reg = (F.relu(mask_source.sum(1) - self.source_mask_budget)).mean()

            for k in range(self.kfac):
                cates_k = cates[:, k].reshape(1, -1)
                # x_k_source = self.fft_enhance(rating_matrix_source, cates_k)
                x_k_source = rating_matrix_source * cates_k
                h_source = self.encoder(x_k_source)
                self.debug_list.append(h_source)
                h_source, graph = self.transformer_layer(F.normalize(h_source))
                # h_source, graph = h_source[-1].squeeze(), graph[-1]
                self.graph_list_source.append(graph)
                z_k_source = F.normalize(h_source)
                self.h_source_list.append(z_k_source)

                # decoder
                logits_k = torch.matmul(z_k_source, items_source.transpose(0, 1)) / self.tau
                probs_k = torch.exp(logits_k)
                probs_source.append(probs_k)

            probs_source = (torch.stack(probs_source, dim=1) * mask_source.unsqueeze(-1)).sum(1)
            logits_source = torch.log(probs_source + 1e-12)
        else:
            logits_source = None

        # Target channel
        self.h_target_list = []
        self.graph_list_target = []
        if target_user != None:
            rating_matrix_target = self.rating_matrix[target_user].to(self.device)
            rating_matrix_target = F.normalize(rating_matrix_target)
            rating_matrix_target = F.dropout(rating_matrix_target, self.drop_out, training=self.training)
            probs_target = []

            guidance_target = self.rating_matrix_target[target_user].to(self.device)
            guidance_target = F.normalize(guidance_target)
            guidance_target = F.dropout(guidance_target, self.drop_out, training=self.training)
            mask_target = self.binary(self.mask_target(guidance_target))
            self.mask_reg += (F.relu(mask_target.sum(1) - self.target_mask_budget)).mean()

            for k in range(self.kfac):
                cates_k = cates[:, k].reshape(1, -1)
                # x_k_target = self.fft_enhance(rating_matrix_target, cates_k)
                x_k_target = rating_matrix_target * cates_k
                h_target = self.encoder(x_k_target)
                h_target, graph = self.transformer_layer(F.normalize(h_target))
                # h_target, graph = h_target[-1].squeeze(), graph[-1]
                self.graph_list_target.append(graph)
                z_k_target = F.normalize(h_target)
                self.h_target_list.append(z_k_target)

                # decoder
                logits_k = torch.matmul(z_k_target, items_target.transpose(0, 1)) / self.tau
                probs_k = torch.exp(logits_k)
                probs_target.append(probs_k)

            probs_target = (torch.stack(probs_target, dim=1) * mask_target.unsqueeze(-1)).sum(1)
            logits_target = torch.log(probs_target + 1e-12)
        else:
            logits_target = None

        return logits_source, logits_target

    def process_source_user_id(self, id):
        id[id >= self.overlapped_num_users] += self.target_num_users - self.overlapped_num_users
        return id

    def covariance_reg(self):
        L = torch.tril(self.cores_logvar.weight)
        L = F.normalize(L)
        covariance = L @ L.T
        return covariance.abs().mean()

    def calculate_loss(self, interaction):
        source_user_id = interaction[self.SOURCE_USER_ID]
        target_user_id = interaction[self.TARGET_USER_ID].cpu()
        source_user_id = self.process_source_user_id(source_user_id).cpu()

        rating_matrix_source_label = self.rating_matrix_source[source_user_id].to(self.device)
        rating_matrix_target_label = self.rating_matrix_target[target_user_id].to(self.device)

        z_source, z_target = self.forward(source_user_id, target_user_id)

        # CE loss
        ce_loss_source = -(F.log_softmax(z_source, 1) * rating_matrix_source_label).sum(1).mean()
        ce_loss_target = -(F.log_softmax(z_target, 1) * rating_matrix_target_label).sum(1).mean()

        # graph reg
        reg_loss = torch.tensor(0., device=self.device)

        # compactness reg
        compactness_loss = torch.tensor(0., device=self.device)

        return (ce_loss_source, ce_loss_target, self.mask_reg)

    def full_sort_predict_source(self, interaction):
        user = interaction[self.SOURCE_USER_ID]
        scores, _ = self.forward(source_user=user.cpu())

        return scores.reshape(-1)

    def full_sort_predict_target(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        _, scores  = self.forward(target_user=user.cpu())

        return scores.reshape(-1)

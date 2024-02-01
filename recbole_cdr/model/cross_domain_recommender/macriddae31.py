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

class FilterLayer(nn.Module):
    def __init__(self, item_size, lat_dim):
        super(FilterLayer, self).__init__()
        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.linear = nn.Parameter(torch.randn(1, lat_dim, dtype=torch.float32))
        nn.init.xavier_normal_(self.linear.data)
        # self.LayerNorm = nn.LayerNorm(item_size, eps=1e-12)

    def forward(self, input_tensor):
        # input_tensor: [batch, hidden]
        batch, hidden = input_tensor.shape
        x = torch.fft.fft2(input_tensor)
        x = x * self.linear
        hidden_states = torch.fft.ifft2(x).real

        return hidden_states

class MacridDAE31(CrossDomainRecommender, AutoEncoderMixin):
    r"""(30) + Contrastive reg

    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(MacridDAE31, self).__init__(config, dataset)

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
        self.if_mask = config["if_mask"]

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
        # self.cores_logvar = nn.Parameter(torch.eye(self.kfac))
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
        self.mask_generator = nn.Sequential(
            nn.Linear(self.lat_dim, 1),
            nn.ReLU(),
        )
        # self.mask_source = MLPLayers([self.source_num_items, self.kfac], activation='tanh')
        # self.mask_target = MLPLayers([self.target_num_items, self.kfac], activation='tanh')
        # self.projector = MLPLayers([self.lat_dim, self.lat_dim, self.lat_dim], activation='tanh')
        # parameters initialization
        # nn.init.xavier_normal_(self.cores_logvar)
        self.mask_source = nn.Embedding(1, self.kfac)
        self.mask_target = nn.Embedding(1, self.kfac)
        self.apply(xavier_normal_initialization)
        # nn.init.uniform_(self.mask_source.weight, 0, 0.2)
        # nn.init.uniform_(self.mask_target.weight, 0, 0.2)
        init_mask_source = torch.concat([torch.ones(7, device=self.device), -torch.ones(3, device=self.device)])
        init_mask_target = torch.concat([-torch.ones(3, device=self.device), torch.ones(7, device=self.device)])
        self.mask_source.weight.data.copy_(0.2 * init_mask_source)
        self.mask_target.weight.data.copy_(0.2 * init_mask_target)
        # self.mask_source.weight.requires_grad_(False)
        # self.mask_target.weight.requires_grad_(False)
        self.source_mask_budget = 7
        self.target_mask_budget = 7
        self.get_sparse_norm_rating_matrix()


    def get_sparse_norm_rating_matrix(self):
        rating_matrix = F.normalize(self.rating_matrix).to_sparse()
        self.sparse_norm_rating_matrix = rating_matrix.to(self.device)
        rating_matrix = torch.tensor(self.source_interaction_matrix.toarray())
        self.sparse_norm_rating_source = F.normalize(rating_matrix).to_sparse().to(self.device)
        rating_matrix = torch.tensor(self.target_interaction_matrix.toarray())
        self.sparse_norm_rating_target = F.normalize(rating_matrix).to_sparse().to(self.device)

        self.rating_matrix_source = torch.cat([
            self.rating_matrix[:, :self.overlapped_num_items],
            self.rating_matrix[:, self.target_num_items:]
        ], dim=-1)
        self.rating_matrix_target = self.rating_matrix[:self.target_num_users][:, :self.target_num_items]

    def get_cates(self, cates_logits):
        if self.nogb:
            cates = torch.softmax(cates_logits, dim=-1)
        else:
            cates_sample = F.gumbel_softmax(cates_logits, tau=1, hard=False, dim=-1)
            cates_mode = torch.softmax(cates_logits, dim=-1)
            cates = self.training * cates_sample + (1 - self.training) * cates_mode
        return cates
    
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

            for k in range(self.kfac):
                cates_k = cates[:, k].reshape(1, -1)
                # x_k_source = self.fft_enhance(rating_matrix_source, cates_k)
                x_k_source = rating_matrix_source * cates_k
                h_source = self.encoder(x_k_source)
                self.debug_list.append(h_source)
                h_source, graph = self.transformer_layer(F.normalize(h_source))
                # h_source, graph = h_source[-1].squeeze(), graph[-1]
                self.h_source_list.append(h_source)
                self.graph_list_source.append(graph)
                z_k_source = F.normalize(h_source)

                # decoder
                logits_k = torch.matmul(z_k_source, items_source.transpose(0, 1)) / self.tau
                probs_k = torch.exp(logits_k)
                probs_source.append(probs_k)

            if self.if_mask:
                mask_source = self.binary(self.mask_source.weight.T).unsqueeze(-1)
                probs_source = (torch.stack(probs_source, dim=0) * mask_source).sum(0)
            else:
                probs_source = torch.stack(probs_source, dim=0).sum(0)
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
            for k in range(self.kfac):
                cates_k = cates[:, k].reshape(1, -1)
                # x_k_target = self.fft_enhance(rating_matrix_target, cates_k)
                x_k_target = rating_matrix_target * cates_k
                h_target = self.encoder(x_k_target)
                h_target, graph = self.transformer_layer(F.normalize(h_target))
                # h_target, graph = h_target[-1].squeeze(), graph[-1]
                self.h_target_list.append(h_target)
                self.graph_list_target.append(graph)
                z_k_target = F.normalize(h_target)

                # decoder
                logits_k = torch.matmul(z_k_target, items_target.transpose(0, 1)) / self.tau
                probs_k = torch.exp(logits_k)
                probs_target.append(probs_k)

            if self.if_mask:
                mask_target = self.binary(self.mask_target.weight.T).unsqueeze(-1)
                probs_target = (torch.stack(probs_target, dim=0) * mask_target).sum(0)
            else:
                probs_target = torch.stack(probs_target, dim=0).sum(0)
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

    def gsl_covariance_reg(self):
        L = torch.tril(self.cores_logvar.weight).detach()
        L = F.normalize(L)
        covariance = (L @ L.T)

        source_graph = F.normalize(torch.stack(self.h_source_list, dim=0).mean(1))
        source_graph = source_graph @ source_graph.T

        target_graph = F.normalize(torch.stack(self.h_target_list, dim=0).mean(1))
        target_graph = target_graph @ target_graph.T

        source_reg = F.mse_loss(source_graph, covariance)
        target_reg = F.mse_loss(target_graph, covariance)
        return source_reg + target_reg

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

        mask_reg = (
            F.relu(self.binary(self.mask_source.weight).sum() - self.source_mask_budget) + 
            F.relu(self.binary(self.mask_target.weight).sum() - self.target_mask_budget)
        )

        return (ce_loss_source, ce_loss_target, 0. * mask_reg, self.cov_reg * self.covariance_reg(), 1 * self.gsl_covariance_reg())

    def full_sort_predict_source(self, interaction):
        user = interaction[self.SOURCE_USER_ID]
        scores, _ = self.forward(source_user=user.cpu())

        return scores.reshape(-1)

    def full_sort_predict_target(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        _, scores  = self.forward(target_user=user.cpu())

        return scores.reshape(-1)

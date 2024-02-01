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
from recbole.model.layers import MLPLayers
from recbole.utils import InputType
from torch.distributions.binomial import Binomial
from recbole_cdr.model.layers import Encoder, Binarize

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

class MacridDAE17(CrossDomainRecommender, AutoEncoderMixin):
    r"""FFT + loss calculation + user gate

    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(MacridDAE17, self).__init__(config, dataset)

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
        # self.user_gate = nn.Sequential(
        #     nn.Linear(self.lat_dim, self.kfac),
        #     nn.Tanh(),
        # )
        self.binary = Binarize.apply
        # self.mask_source = MLPLayers([self.source_num_items, self.kfac], activation='tanh')
        # self.mask_target = MLPLayers([self.target_num_items, self.kfac], activation='tanh')
        # self.projector = MLPLayers([self.lat_dim, self.lat_dim, self.lat_dim], activation='tanh')
        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.get_sparse_norm_rating_matrix()
        # self.mask_source = nn.Parameter(torch.ones(1, self.kfac))
        # self.mask_target = nn.Parameter(torch.ones(1, self.kfac))
        self.mask_source = nn.Parameter(torch.zeros(self.kfac, 1))
        self.mask_target = nn.Parameter(torch.zeros(self.kfac, 1))
        nn.init.normal_(self.mask_source, 0, 0.5)
        nn.init.normal_(self.mask_target, 0, 0.5)
        # nn.init.uniform_(self.mask_source, -0.5, 0.5)
        # nn.init.uniform_(self.mask_target, -0.5, 0.5)


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

    def sprase_drop(self, A : torch.Tensor):
        if self.training:
            dist = Binomial(probs=1 - self.drop_out)
            mask = dist.sample(A.values().size()).bool()
            A_drop = torch.sparse_coo_tensor(A.indices()[:, mask], A.values()[mask] * 1.0 / (1 - self.drop_out), size=A.size())
            return A_drop.to(A.device)
        else:
            return A

    def fft_enhance(self, A, cates_k):
        A_fft = torch.fft.fft(A.T)
        A_abs, A_pha = torch.abs(A_fft), torch.angle(A_fft)
        A_abs = A_abs * cates_k.T
        A_hat = A_abs * (torch.e ** (1j * A_pha))
        A_hat = torch.fft.ifft(A_hat).real.T
        return A_hat

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

    def forward(self, source_user=None, target_user=None):
        cores = F.normalize(self.k_embedding.weight, dim=1)
        # cores_source = F.normalize(self.k_embedding_source.weight, dim=1)
        # cores_target = F.normalize(self.k_embedding_target.weight, dim=1)
        items = F.normalize(self.item_embedding.weight, dim=1)
        items_source = items[self.target_num_items - 1:]
        items_target = items[:self.target_num_items]

        # if self.training:
        #     gumbels_source = (
        #         -torch.empty_like(self.mask_source, memory_format=torch.legacy_contiguous_format).exponential_().log()
        #     ).to(self.device)
        #     gumbels_target = (
        #         -torch.empty_like(self.mask_target, memory_format=torch.legacy_contiguous_format).exponential_().log()
        #     ).to(self.device)
        # else:
        #     gumbels_source, gumbels_target = 0, 0
        # mask_source = self.binary(self.mask_source / 0.1 + gumbels_source)
        # mask_target = self.binary(self.mask_target / 0.1 + gumbels_target)
        # cores_source = cores * mask_source
        # cores_target = cores * mask_target

        # cates_logits_source = items_source[1:] @ cores_source.T / self.tau
        # cates_logits_target = items_target @ cores_target.T / self.tau
        # cates_logits = torch.cat([cates_logits_source, cates_logits_target])

        cates_logits = torch.matmul(items, cores.transpose(0, 1)) / self.tau

        cates = self.get_cates(cates_logits)

        # Source channel
        if source_user != None:
            rating_matrix_source = self.rating_matrix[source_user].to(self.device)
            rating_matrix_source = F.normalize(rating_matrix_source)
            rating_matrix_source = F.dropout(rating_matrix_source, self.drop_out, training=self.training)
            probs_source = None

            # user gate
            # user_gate = self.user_gate(self.encoder(rating_matrix_source))
            # cates_source = user_gate.unsqueeze(1) * cates.unsqueeze(0)

            for k in range(self.kfac):
                cates_k = cates[:, k].reshape(1, -1)
                # cates_k = cates_source[:, :, k]
                x_k_source = rating_matrix_source * cates_k
                h_source = F.normalize(self.encoder(x_k_source))
                z_k_source = F.normalize(h_source)

                # decoder
                logits_k = torch.matmul(z_k_source, items_source.transpose(0, 1)) / self.tau
                probs_k = torch.exp(logits_k)
                probs_k = probs_k
                probs_source = probs_k if (probs_source is None) else (probs_source + probs_k)

            logits_source = torch.log(probs_source)
        else:
            logits_source = None

        # Target channel
        if target_user != None:
            rating_matrix_target = self.rating_matrix[target_user].to(self.device)
            rating_matrix_target = F.normalize(rating_matrix_target)
            rating_matrix_target = F.dropout(rating_matrix_target, self.drop_out, training=self.training)
            probs_target = None
            # user gate
            # user_gate = self.user_gate(self.encoder(rating_matrix_target))
            # cates_target = user_gate.unsqueeze(1) * cates.unsqueeze(0)

            for k in range(self.kfac):
                cates_k = cates[:, k].reshape(1, -1)
                # cates_k = cates_target[:, :, k]
                x_k_target = rating_matrix_target * cates_k
                h_target = F.normalize(self.encoder(x_k_target))
                z_k_target = F.normalize(h_target)

                # decoder
                logits_k = torch.matmul(z_k_target, items_target.transpose(0, 1)) / self.tau
                probs_k = torch.exp(logits_k)
                probs_k = probs_k
                probs_target = probs_k if (probs_target is None) else (probs_target + probs_k)

            logits_target = torch.log(probs_target)
        else:
            logits_target = None

        return logits_source, logits_target

    def process_source_user_id(self, id):
        id[id >= self.overlapped_num_users] += self.target_num_users - self.overlapped_num_users
        return id

    def calculate_loss(self, interaction):
        source_user_id = interaction[self.SOURCE_USER_ID]
        target_user_id = interaction[self.TARGET_USER_ID].cpu()
        source_user_id = self.process_source_user_id(source_user_id).cpu()

        rating_matrix_source_label = self.rating_matrix_source[source_user_id].to(self.device)
        rating_matrix_target_label = self.rating_matrix_target[target_user_id].to(self.device)

        z_source, z_target = self.forward(source_user_id, target_user_id)

        # CE loss
        # ce_loss_source = -(F.log_softmax(z_source, 1) * rating_matrix_source).sum(1).mean()
        # ce_loss_target = -(F.log_softmax(z_target, 1) * rating_matrix_target).sum(1).mean()
        ce_loss_source = -(F.log_softmax(z_source, 1) * rating_matrix_source_label).sum(1).mean()
        ce_loss_target = -(F.log_softmax(z_target, 1) * rating_matrix_target_label).sum(1).mean()

        return (ce_loss_source, ce_loss_target)
        return (ce_loss)

    def full_sort_predict_source(self, interaction):
        user = interaction[self.SOURCE_USER_ID]
        scores, _ = self.forward(source_user=user.cpu())

        return scores.reshape(-1)

    def full_sort_predict_target(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        # _, scores  = self.forward(self.rating_matrix_source[[0, 1]].to(self.device), rating_matrix)
        _, scores  = self.forward(target_user=user.cpu())

        return scores.reshape(-1)

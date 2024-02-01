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


class MacridDAE11(CrossDomainRecommender, AutoEncoderMixin):
    r"""user intent

    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(MacridDAE11, self).__init__(config, dataset)

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
        self.tau = config["tau"]
        self.nogb = config["nogb"]
        self.regs = config["reg_weights"]

        self.epsilon = config["epsilon"]
        self.lmd_disen = config["lmd_disen"]

        # define layers and loss
        self.update = 0

        self.encode_layer_dims = [self.total_num_items] + self.layers + [self.lat_dim]
        
        self.encoder = nn.Linear(self.total_num_items, self.lat_dim)
        self.linear_proj = nn.ModuleList([nn.Linear(self.lat_dim, self.lat_dim) for _ in range(self.kfac)])
        self.item_embedding = nn.Embedding(self.total_num_items, self.lat_dim)
        self.k_embedding = nn.Embedding(self.kfac, self.lat_dim)
        self.l2_loss = EmbLoss()

        self.user_intent_extractor = nn.Linear(self.total_num_items, self.lat_dim)

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.get_sparse_norm_rating_matrix()
        
        self.graph_conv = GraphConv(1, 2, weight=False, bias=False)

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

    def get_cates(self, cates_logits):
        if self.nogb:
            cates = torch.softmax(cates_logits, dim=-1)
        else:
            cates_sample = F.gumbel_softmax(cates_logits, tau=1, hard=False, dim=-1)
            cates_mode = torch.softmax(cates_logits, dim=-1)
            cates = self.training * cates_sample + (1 - self.training) * cates_mode
        return cates

    def forward(self, rating_matrix, user=None):
        self.h_list = []
        cores = F.normalize(self.k_embedding.weight, dim=1)
        items = F.normalize(self.item_embedding.weight, dim=1)


        rating_matrix = F.normalize(rating_matrix)
        cates_logits = rating_matrix @ items @ cores.T / self.tau


        rating_matrix = F.dropout(rating_matrix, self.drop_out, training=self.training)

        cates = self.get_cates(cates_logits)

        probs = None
        for k in range(self.kfac):
            cates_k = cates[:, k].reshape(-1, 1)
            # encoder
            x_k = rating_matrix * cates_k
            # x_k = rating_matrix
            h = self.encoder(x_k)

            # decoder
            # self.h_list.append(h)
            z_k = F.normalize(h, dim=1)
            self.h_list.append(z_k)
            logits_k = torch.matmul(z_k, items.transpose(0, 1)) / self.tau
            probs_k = torch.exp(logits_k)
            probs_k = probs_k
            # probs_k = probs_k
            probs = probs_k if (probs is None) else (probs + probs_k)

        logits = torch.log(probs)

        return logits

    def process_source_user_id(self, id):
        id[id >= self.overlapped_num_users] += self.target_num_users - self.overlapped_num_users
        return id

    def calculate_loss(self, interaction):
        source_user_id = interaction[self.SOURCE_USER_ID]
        target_user_id = interaction[self.TARGET_USER_ID]
        source_user_id = self.process_source_user_id(source_user_id)
        all_user_id = torch.cat([source_user_id, target_user_id])
        rating_matrix = self.rating_matrix[all_user_id.cpu()].to(source_user_id.device)

        self.saved_user_id = all_user_id

        z = self.forward(rating_matrix, all_user_id)

        # CE loss
        ce_loss = -(F.log_softmax(z, 1) * rating_matrix).sum(1).mean()

        if self.regs[0] != 0 or self.regs[1] != 0:
            return ce_loss + self.reg_loss()
        return (ce_loss)
    
    def reg_loss(self):
        r"""Calculate the L2 normalization loss of model parameters.
        Including embedding matrices and weight matrices of model.

        Returns:
            loss(torch.FloatTensor): The L2 Loss tensor. shape of [1,]
        """
        reg_1, reg_2 = self.regs[:2]
        loss_1 = reg_1 * self.item_embedding.weight.norm(2)
        loss_2 = reg_1 * self.k_embedding.weight.norm(2)
        loss_3 = 0
        for name, parm in self.encoder.named_parameters():
            if name.endswith("weight"):
                loss_3 = loss_3 + reg_2 * parm.norm(2)
        return loss_1 + loss_2 + loss_3

    def full_sort_predict_source(self, interaction):
        user = interaction[self.SOURCE_USER_ID]

        rating_matrix = self.rating_matrix[user.cpu()].to(user.device)

        scores = self.forward(rating_matrix, user)
        scores = scores[:, self.target_num_items - self.overlapped_num_items:]
        return scores.reshape(-1)

    def full_sort_predict_target(self, interaction):
        user = interaction[self.TARGET_USER_ID]

        rating_matrix= self.rating_matrix[user.cpu()].to(user.device)
        
        scores = self.forward(rating_matrix, user)
        scores = scores[:, :self.target_num_items]
        return scores.reshape(-1)

    

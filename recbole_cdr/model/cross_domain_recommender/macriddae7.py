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


class MacridDAE7(CrossDomainRecommender, AutoEncoderMixin):
    r"""add two specific channel in MacirdDAE5

    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(MacridDAE7, self).__init__(config, dataset)

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

        transformer_encoder = torch.nn.TransformerEncoderLayer(
            d_model=self.lat_dim,
            nhead=2,
            dim_feedforward=self.lat_dim * 4,
            dropout=0.5,
            activation=nn.GELU(),
            layer_norm_eps=1e-12,
            batch_first=True,
            norm_first=False
        )
        self.transformer_layer = torch.nn.TransformerEncoder(
            encoder_layer=transformer_encoder,
            num_layers=1,
        )
        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.get_sparse_norm_rating_matrix()
        self.graph_construction(self.source_interaction_matrix, self.target_interaction_matrix)
        # self.graph_conv = GATConv(self.lat_dim, self.lat_dim, 2)
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

    def graph_construction(self, source_graph, target_graph):
        all_graph = source_graph + target_graph
        co_occurence = all_graph @ all_graph.T + sp.eye(self.total_num_users)
        # co_occurence = co_occurence.multiply(co_occurence > self.graph_threshold)
        degree = np.array((co_occurence > 0).sum(1)).flatten()
        degree = np.power(degree, -0.5)
        degree = sp.diags(degree)
        norm_adj = (degree @ co_occurence @ degree).tocoo()
        norm_adj = torch.sparse_coo_tensor(
            np.row_stack([norm_adj.row, norm_adj.col]),
            norm_adj.data,
            (self.total_num_users, self.total_num_users),
            dtype=torch.float32
        )
        self.norm_adj = norm_adj.to(self.device)
        self.norm_adj_dense = self.norm_adj.to_dense()
        self.g = dgl.from_scipy(co_occurence).to(self.device)

    def get_gnn_embeddings(self, emb, noise=False):
        emb_list = [emb]
        for idx in range(1):
            if noise:
                # emb = self.graph_conv(self.g, emb)
                emb = torch.sparse.mm(self.norm_adj, emb)
                random_noise = torch.rand_like(emb, device=self.device)
                emb += torch.sign(emb) * F.normalize(random_noise, dim=-1) * 0.05
            else:
                # emb = self.graph_conv(self.g, emb)
                emb = torch.sparse.mm(self.norm_adj, emb)
            emb_list.append(emb)
        emb = torch.stack(emb_list, dim=1).mean(1)
        return emb


    def mlp_layers(self, layer_dims):
        mlp_modules = []
        for i, (d_in, d_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            mlp_modules.append(nn.Linear(d_in, d_out))
            if i != len(layer_dims[:-1]) - 1:
                mlp_modules.append(nn.Tanh())
        return nn.Sequential(*mlp_modules)

    def sprase_drop(self, A : torch.Tensor):
        if self.training:
            dist = Binomial(probs=1 - self.drop_out)
            mask = dist.sample(A.values().size()).bool()
            A_drop = torch.sparse_coo_tensor(A.indices()[:, mask], A.values()[mask] * 1.0 / (1 - self.drop_out), size=A.size())
            return A_drop.to(A.device)
        else:
            return A

    def subgraph_convolution(self, emb, user):
        sg = dgl.node_subgraph(self.g, user)
        sg = dgl.add_self_loop(sg)
        emb_list = [emb]
        for idx in range(1):
            emb = self.graph_conv(sg, emb)
            emb_list.append(emb)
        # emb = torch.stack(emb_list, dim=1).mean(1)
        return emb

    def get_cates(self, cates_logits):
        if self.nogb:
            cates = torch.softmax(cates_logits, dim=-1)
        elif self.nobinarize:
            cates_sample = F.gumbel_softmax(cates_logits, tau=1, hard=False, dim=-1)
            cates_mode = torch.softmax(cates_logits, dim=-1)
            cates = self.training * cates_sample + (1 - self.training) * cates_mode
        return cates

    def forward(self, rating_matrix, rating_matrix_source, rating_matrix_target, user=None):
        self.h_list = []
        cores = F.normalize(self.k_embedding.weight, dim=1)
        items = F.normalize(self.item_embedding.weight, dim=1)

        rating_matrix = F.normalize(rating_matrix)
        rating_matrix = F.dropout(rating_matrix, self.drop_out, training=self.training)

        rating_matrix_source = F.normalize(rating_matrix_source)
        rating_matrix_source = F.dropout(rating_matrix_source, self.drop_out, training=self.training)

        rating_matrix_target = F.normalize(rating_matrix_target)
        rating_matrix_target = F.dropout(rating_matrix_target, self.drop_out, training=self.training)

        cates_logits = torch.matmul(items, cores.transpose(0, 1)) / self.tau

        if self.nogb:
            cates = torch.softmax(cates_logits, dim=-1)
        else:
            cates_sample = F.gumbel_softmax(cates_logits, tau=1, hard=False, dim=-1)
            cates_mode = torch.softmax(cates_logits, dim=-1)
            cates = self.training * cates_sample + (1 - self.training) * cates_mode


        # probs = None
        probs_source = None
        probs_target = None
        # att_bias = self.norm_adj_dense[user][:, user]
        att_bias = None
        for k in range(self.kfac):
            cates_k = cates[:, k].reshape(1, -1)
            # encoder
            x_k = rating_matrix * cates_k
            h = self.encoder(x_k)

            x_k_source = rating_matrix_source * cates_k
            h_source = self.encoder(x_k_source)

            x_k_target = rating_matrix_target * cates_k
            h_target = self.encoder(x_k_target)

            # h = self.transformer_layer(h, mask=att_bias)
            # h = h + self.subgraph_convolution(h, user)

            # decoder
            # self.h_list.append(h)
            z_k = F.normalize(h + h_source, dim=1)
            logits_k = torch.matmul(z_k, items.transpose(0, 1)) / self.tau
            probs_k = torch.exp(logits_k)
            # probs_k = probs_k * cates_k
            probs_k = probs_k * cates_k
            probs_source = probs_k if (probs_source is None) else (probs_source + probs_k)

            z_k = F.normalize(h + h_target, dim=1)
            logits_k = torch.matmul(z_k, items.transpose(0, 1)) / self.tau
            probs_k = torch.exp(logits_k)
            # probs_k = probs_k * cates_k
            probs_k = probs_k * cates_k
            probs_target = probs_k if (probs_target is None) else (probs_target + probs_k)



        logits_source = torch.log(probs_source)
        logits_target = torch.log(probs_target)

        return logits_source, logits_target

    def process_source_user_id(self, id):
        id[id >= self.overlapped_num_users] += self.target_num_users - self.overlapped_num_users
        return id

    def calculate_loss(self, interaction):
        source_user_id = interaction[self.SOURCE_USER_ID]
        target_user_id = interaction[self.TARGET_USER_ID]
        source_user_id = self.process_source_user_id(source_user_id)
        all_user_id = torch.cat([source_user_id, target_user_id])
        rating_matrix = self.rating_matrix[all_user_id.cpu()].to(source_user_id.device)
        rating_matrix_source = self.rating_matrix_source[source_user_id.cpu()].to(source_user_id.device)
        rating_matrix_target = self.rating_matrix_target[target_user_id.cpu()].to(target_user_id.device)

        self.saved_user_id = all_user_id

        z = self.forward(rating_matrix, rating_matrix_source, rating_matrix_target, all_user_id)

        # CE loss
        ce_loss = -(F.log_softmax(z, 1) * rating_matrix).sum(1).mean()
        
        # Compactness loss
        disen_loss = 0
        # disen_loss = self.compactness(torch.stack(self.h_list, dim=1))
        for h in self.h_list:
            disen_loss += self.vicreg_covariance(h)

        if self.regs[0] != 0 or self.regs[1] != 0:
            return ce_loss + self.reg_loss()
        return (ce_loss, self.lmd_disen * disen_loss)
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
        rating_matrix_source = self.rating_matrix_source[user.cpu()].to(user.device)
        rating_matrix_target = self.rating_matrix_target[torch.zeros_like(user)].to(user.device)

        scores = self.forward(rating_matrix, rating_matrix_source, rating_matrix_target, user)
        # scores = scores[:, self.target_num_items - self.overlapped_num_items:]
        return scores.reshape(-1)

    def full_sort_predict_target(self, interaction):
        user = interaction[self.TARGET_USER_ID]

        rating_matrix= self.rating_matrix[user.cpu()].to(user.device)
        rating_matrix_source = self.rating_matrix_source[torch.zeros_like(user)].to(user.device)
        rating_matrix_target = self.rating_matrix_target[user.cpu()].to(user.device)
        
        scores = self.forward(rating_matrix, rating_matrix_source, rating_matrix_target, user)
        # scores = scores[:, :self.target_num_items]
        return scores.reshape(-1)

    

# -*- coding: utf-8 -*-
# @Time   : 2022/3/23
# @Author : Gaowei Zhang
# @Email  : 1462034631@qq.com
import dgl
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

from recbole_cdr.model.layers import Binarize, KernelAttention
from dgl.nn.pytorch.conv import GraphConv
from torch.distributions.binomial import Binomial

def InfoNCE(view1, view2, temperature=0.2):
    view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    cl_loss = -torch.log(pos_score / ttl_score)
    return torch.mean(cl_loss)

class MacridDAE4(CrossDomainRecommender, AutoEncoderMixin):
    r"""BiTGCF uses feature propagation and feature transfer to achieve bidirectional
        knowledge transfer between the two domains.
        We extend the basic BiTGCF model in a symmetrical way to support those datasets that have overlapped items.

    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(MacridDAE4, self).__init__(config, dataset)

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
        self.tau_gsl = config["tau_gsl"]
        self.tau_source = config["tau_source"]
        self.tau_target = config["tau_target"]
        self.source_loss = config["source_loss"]
        self.target_loss = config["target_loss"]
        self.nogb = config["nogb"]
        self.regs = config["reg_weights"]
        self.graph_threshold = config["graph_threshold"]
        self.gnn_layers = config["gnn_layers"]
        self.graph_noise = config["graph_noise"]
        self.cl_weight = config["cl_weight"]

        # ablation
        self.no_mask = config["no_mask"]
        self.no_gsl = config["no_gsl"]

        # define layers and loss
        self.update = 0

        self.encode_layer_dims = [self.total_num_items] + self.layers + [self.lat_dim]
        
        self.encoder = MLPLayers(self.encode_layer_dims, activation="tanh")
        # self.encoder = nn.Linear(self.total_num_items, self.lat_dim)
        self.item_embedding = nn.Embedding(self.total_num_items, self.lat_dim)
        self.item_embedding_source = nn.Embedding(self.source_num_items, self.lat_dim)
        self.item_embedding_target = nn.Embedding(self.target_num_items, self.lat_dim)
        self.l2_loss = EmbLoss()

        # My layers
        self.source_aug_norm = nn.LayerNorm(self.lat_dim)
        self.target_aug_norm = nn.LayerNorm(self.lat_dim)
        self.proj_source = nn.Linear(self.lat_dim, self.lat_dim)
        self.proj_target = nn.Linear(self.lat_dim, self.lat_dim)
        self.mask_source = nn.Parameter(torch.empty(1, self.kfac))
        self.mask_target = nn.Parameter(torch.empty(1, self.kfac))
        self.binary = Binarize.apply
        # self.graph_layer = KernelAttention(
        #     self.lat_dim,
        #     self.lat_dim,
        #     self.kfac,
        #     nb_random_features=256
        # )
        self.graph_conv = GraphConv(1, 2, weight=False, bias=False, allow_zero_in_degree=True)

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.get_sparse_norm_rating_matrix()
        self.graph_construction(self.source_interaction_matrix, self.target_interaction_matrix)
        nn.init.normal_(self.mask_source, 0, 0.01)
        nn.init.normal_(self.mask_target, 0, 0.01)

    def graph_construction(self, source_graph, target_graph):
        all_graph = source_graph + target_graph
        co_occurence = all_graph @ all_graph.T
        co_occurence = co_occurence.multiply(co_occurence > self.graph_threshold)
        self.g = dgl.from_scipy(co_occurence).to(self.device)

    def get_gnn_embeddings(self, emb, noise=True):
        emb_list = [emb]
        for idx in range(self.gnn_layers):
            if noise:
                emb = self.graph_conv(self.g, emb)
                random_noise = torch.rand_like(emb, device=self.device)
                emb += torch.sign(emb) * F.normalize(random_noise, dim=-1) * self.graph_noise
            else:
                emb = self.graph_conv(self.g, emb)
            emb_list.append(emb)
        emb = torch.stack(emb_list, dim=1).mean(1)
        return emb

    def factor_gsl(self, h):
        h = F.normalize(h)
        self.h_factor = self.factor_division(h)
        h_enhanced = self.graph_layer(self.h_factor, self.tau_gsl)
        h = h + h_enhanced
        h = F.normalize(h)
        return h

    def factor_division(self, h):
        return h.reshape(-1, self.kfac, self.lat_dim // self.kfac)
        h_factor_list = []
        for idx in range(self.kfac):
            h_factor = self.subspace_projector[idx](h)
            h_factor_list.append(h_factor)
        h_factor = torch.stack(h_factor_list, dim=1)
        return h_factor
    
    def mlp_layers(self, layer_dims):
        mlp_modules = []
        for i, (d_in, d_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            mlp_modules.append(nn.Linear(d_in, d_out))
            if i != len(layer_dims[:-1]) - 1:
                mlp_modules.append(nn.Tanh())
        return nn.Sequential(*mlp_modules)

    def factor_selection(self, h):
        h_source = self.proj_source(h).reshape(h.shape[0], self.kfac, -1)
        h_target = self.proj_target(h).reshape(h.shape[0], self.kfac, -1)
        mask_source = self.mask_source.unsqueeze(-1)
        mask_target = self.mask_target.unsqueeze(-1)
        # mask_source = self.binary(self.mask_source).unsqueeze(-1)
        # mask_target = self.binary(self.mask_target).unsqueeze(-1)
        h_source_aug = F.normalize((h_source * mask_source).flatten(-2, -1))
        h_target_aug = F.normalize((h_target * mask_target).flatten(-2, -1))
        # h_source_aug = self.source_aug_norm((h_source * mask_source).flatten(-2, -1))
        # h_target_aug = self.target_aug_norm((h_target * mask_target).flatten(-2, -1))
        return h_source_aug, h_target_aug

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

    def compactness(self, emb):
        # emb: [N, K, D]
        emb = F.normalize(emb, dim=-1)
        loss = 0
        emb_all = torch.flatten(emb, 0, 1) # [NK, D]
        NK, D = emb_all.shape
        loss -= 0.5 * torch.logdet(torch.eye(D, device=self.device) + (D / NK / self.epsilon * emb_all.T) @ emb_all)

        N, K, D = emb.shape
        emb = torch.einsum('abc->bca', emb) # [K, D, N]
        loss_factor = torch.einsum('abc,adc->abd', D / N / self.epsilon * emb, emb) # [K, D, D]
        loss_factor = 0.5 * torch.logdet(torch.eye(D, device=self.device).unsqueeze(0) + loss_factor)
        loss += loss_factor.sum()

        return loss

    def forward(self, rating_matrix, user=None):
        items_source = F.normalize(self.item_embedding_source.weight, dim=1)
        items_target = F.normalize(self.item_embedding_target.weight, dim=1)

        rating_matrix = F.normalize(rating_matrix)
        rating_matrix = F.dropout(rating_matrix, self.drop_out, training=self.training)

        # encoder
        x_k = rating_matrix
        h = self.encoder(x_k)

        if self.training and not self.no_gsl:
            input = self.sprase_drop(self.sparse_norm_rating_matrix)
            first_layer = self.encoder.mlp_layers[1]
            input = torch.sparse.mm(input, first_layer.weight.T) + first_layer.bias
            for m in self.encoder.mlp_layers[2:]:
                input = m(input)
            emb_1 = self.get_gnn_embeddings(input)[user]
            emb_2 = self.get_gnn_embeddings(input)[user]
            self.loss_con = InfoNCE(emb_1, emb_2)
        else:
            self.loss_con = torch.tensor(0., device=self.device)

        z = h
        if self.no_mask:
            z_source, z_target = h, h
        else:
            z_source, z_target = self.factor_selection(z)

        # decoder source
        # z_k = F.normalize(z, dim=1)
        logits_k_source = torch.matmul(z_source, items_source.transpose(0, 1)) / self.tau_source

        # decoder target
        # z_k = F.normalize(z, dim=1)
        logits_k_target = torch.matmul(z_target, items_target.transpose(0, 1)) / self.tau_target

        return logits_k_source, logits_k_target

    def process_source_user_id(self, id):
        id[id >= self.overlapped_num_users] += self.target_num_users - self.overlapped_num_users
        return id

    def calculate_loss(self, interaction):
        source_user_id = interaction[self.SOURCE_USER_ID]
        target_user_id = interaction[self.TARGET_USER_ID]
        source_user_id = self.process_source_user_id(source_user_id)
        all_user = torch.cat([source_user_id, target_user_id])
        rating_matrix = self.rating_matrix[all_user.cpu()].to(source_user_id.device)

        rating_matrix_source = self.rating_matrix_source[source_user_id.cpu()].to(self.device)
        rating_matrix_target = self.rating_matrix_target[target_user_id.cpu()].to(self.device)

        z_source, z_target = self.forward(rating_matrix, all_user)

        # CE loss
        ce_loss_source = -(F.log_softmax(z_source[:len(source_user_id)], 1) * rating_matrix_source).sum(1).mean()
        ce_loss_target = -(F.log_softmax(z_target[len(source_user_id):], 1) * rating_matrix_target).sum(1).mean()
        ce_loss = self.source_loss * ce_loss_source + self.target_loss * ce_loss_target

        if self.regs[0] != 0 or self.regs[1] != 0:
            return ce_loss + self.reg_loss()
        return (ce_loss, self.cl_weight * self.loss_con)
    
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

        scores, _ = self.forward(rating_matrix)
        # scores = scores[:, self.target_num_items - self.overlapped_num_items:]
        return scores.reshape(-1)

    def full_sort_predict_target(self, interaction):
        user = interaction[self.TARGET_USER_ID]

        rating_matrix= self.rating_matrix[user.cpu()].to(user.device)
        
        _, scores = self.forward(rating_matrix)
        # scores = scores[:, :self.target_num_items]
        return scores.reshape(-1)

    

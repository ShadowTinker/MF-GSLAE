

import random as rd
import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import EmbLoss, BPRLoss
from recbole.utils import InputType

def sample_cor_samples(n_users, n_items, cor_batch_size):
    r"""This is a function that sample item ids and user ids.

    Args:
        n_users (int): number of users in total
        n_items (int): number of items in total
        cor_batch_size (int): number of id to sample

    Returns:
        list: cor_users, cor_items. The result sampled ids with both as cor_batch_size long.

    Note:
        We have to sample some embedded representations out of all nodes.
        Because we have no way to store cor-distance for each pair.
    """
    cor_users = rd.sample(list(range(n_users)), cor_batch_size)
    cor_items = rd.sample(list(range(n_items)), cor_batch_size)

    return cor_users, cor_items

class DGCF(CrossDomainRecommender):
    r"""BiTGCF uses feature propagation and feature transfer to achieve bidirectional
        knowledge transfer between the two domains.
        We extend the basic BiTGCF model in a symmetrical way to support those datasets that have overlapped items.

    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(DGCF, self).__init__(config, dataset)

        # load dataset info
        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field

        # load parameters info
        self.device = config['device']
        self.embedding_size = config["embedding_size"]
        self.n_factors = config["n_factors"]
        self.n_iterations = config["n_iterations"]
        self.n_layers = config["n_layers"]
        self.reg_weight = config["reg_weight"]
        self.cor_weight = config["cor_weight"]
        inter_num = dataset.source_domain_dataset.inter_num + dataset.target_domain_dataset.inter_num
        n_batch = inter_num // config["train_batch_size"] + 1
        self.cor_batch_size = int(max(self.total_num_users / n_batch, self.total_num_items / n_batch))
        # ensure embedding can be divided into <n_factors> intent
        assert self.embedding_size % self.n_factors == 0

        # generate intermediate data
        self.source_interaction_matrix = dataset.inter_matrix(form='coo', value_field=None, domain='source').astype(np.float32)
        self.target_interaction_matrix = dataset.inter_matrix(form='coo', value_field=None, domain='target').astype(np.float32)
        self.interaction_matrix = (self.source_interaction_matrix + self.target_interaction_matrix).tocoo()

        row = self.interaction_matrix.row.tolist()
        col = self.interaction_matrix.col.tolist()
        col = [item_index + self.total_num_users for item_index in col]
        all_h_list = row + col  # row.extend(col)
        all_t_list = col + row  # col.extend(row)
        num_edge = len(all_h_list)
        edge_ids = range(num_edge)
        self.all_h_list = torch.LongTensor(all_h_list).to(self.device)
        self.all_t_list = torch.LongTensor(all_t_list).to(self.device)
        self.edge2head = torch.LongTensor([all_h_list, edge_ids]).to(self.device)
        self.head2edge = torch.LongTensor([edge_ids, all_h_list]).to(self.device)
        self.tail2edge = torch.LongTensor([edge_ids, all_t_list]).to(self.device)
        val_one = torch.ones_like(self.all_h_list).float().to(self.device)
        num_node = self.total_num_users + self.total_num_items
        self.edge2head_mat = self._build_sparse_tensor(
            self.edge2head, val_one, (num_node, num_edge)
        )
        self.head2edge_mat = self._build_sparse_tensor(
            self.head2edge, val_one, (num_edge, num_node)
        )
        self.tail2edge_mat = self._build_sparse_tensor(
            self.tail2edge, val_one, (num_edge, num_node)
        )
        self.num_edge = num_edge
        self.num_node = num_node

        # define layers and loss
        self.user_embedding = nn.Embedding(self.total_num_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.total_num_items, self.embedding_size)
        self.softmax = torch.nn.Softmax(dim=1)
        self.loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.restore_user_e = None
        self.restore_item_e = None

        # storage variables for full sort evaluation acceleration
        self.source_restore_user_e = None
        self.source_restore_item_e = None
        self.target_restore_user_e = None
        self.target_restore_item_e = None

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ['source_restore_user_e', 'source_restore_item_e', 'target_restore_user_e', 'target_restore_item_e']

    def _build_sparse_tensor(self, indices, values, size):
        # Construct the sparse matrix with indices, values and size.
        return torch.sparse.FloatTensor(indices, values, size).to(self.device)

    def _get_ego_embeddings(self):
        # concat of user embeddings and item embeddings
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        ego_embeddings = torch.cat([user_emb, item_emb], dim=0)
        return ego_embeddings

    def build_matrix(self, A_values):
        r"""Get the normalized interaction matrix of users and items according to A_values.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        Args:
            A_values (torch.cuda.FloatTensor): (num_edge, n_factors)

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            torch.cuda.FloatTensor: Sparse tensor of the normalized interaction matrix. shape: (num_edge, n_factors)
        """
        norm_A_values = self.softmax(A_values)
        factor_edge_weight = []
        for i in range(self.n_factors):
            tp_values = norm_A_values[:, i].unsqueeze(1)
            # (num_edge, 1)
            d_values = torch.sparse.mm(self.edge2head_mat, tp_values)
            # (num_node, num_edge) (num_edge, 1) -> (num_node, 1)
            d_values = torch.clamp(d_values, min=1e-8)
            try:
                assert not torch.isnan(d_values).any()
            except AssertionError:
                self.logger.info("d_values", torch.min(d_values), torch.max(d_values))

            d_values = 1.0 / torch.sqrt(d_values)
            head_term = torch.sparse.mm(self.head2edge_mat, d_values)
            # (num_edge, num_node) (num_node, 1) -> (num_edge, 1)

            tail_term = torch.sparse.mm(self.tail2edge_mat, d_values)
            edge_weight = tp_values * head_term * tail_term
            factor_edge_weight.append(edge_weight)
        return factor_edge_weight

    def forward(self):
        ego_embeddings = self._get_ego_embeddings()
        all_embeddings = [ego_embeddings.unsqueeze(1)]
        # initialize with every factor value as 1
        A_values = torch.ones((self.num_edge, self.n_factors)).to(self.device)
        A_values = Variable(A_values, requires_grad=True)
        for k in range(self.n_layers):
            layer_embeddings = []

            # split the input embedding table
            # .... ego_layer_embeddings is a (n_factors)-length list of embeddings
            # [n_users+n_items, embed_size/n_factors]
            ego_layer_embeddings = torch.chunk(ego_embeddings, self.n_factors, 1)
            for t in range(0, self.n_iterations):
                iter_embeddings = []
                A_iter_values = []
                factor_edge_weight = self.build_matrix(A_values=A_values)
                for i in range(0, self.n_factors):
                    # update the embeddings via simplified graph convolution layer
                    edge_weight = factor_edge_weight[i]
                    # (num_edge, 1)
                    edge_val = torch.sparse.mm(
                        self.tail2edge_mat, ego_layer_embeddings[i]
                    )
                    # (num_edge, dim / n_factors)
                    edge_val = edge_val * edge_weight
                    # (num_edge, dim / n_factors)
                    factor_embeddings = torch.sparse.mm(self.edge2head_mat, edge_val)
                    # (num_node, num_edge) (num_edge, dim) -> (num_node, dim)

                    iter_embeddings.append(factor_embeddings)

                    if t == self.n_iterations - 1:
                        layer_embeddings = iter_embeddings

                    # get the factor-wise embeddings
                    # .... head_factor_embeddings is a dense tensor with the size of [all_h_list, embed_size/n_factors]
                    # .... analogous to tail_factor_embeddings
                    head_factor_embeddings = torch.index_select(
                        factor_embeddings, dim=0, index=self.all_h_list
                    )
                    tail_factor_embeddings = torch.index_select(
                        ego_layer_embeddings[i], dim=0, index=self.all_t_list
                    )

                    # .... constrain the vector length
                    # .... make the following attentive weights within the range of (0,1)
                    # to adapt to torch version
                    head_factor_embeddings = F.normalize(
                        head_factor_embeddings, p=2, dim=1
                    )
                    tail_factor_embeddings = F.normalize(
                        tail_factor_embeddings, p=2, dim=1
                    )

                    # get the attentive weights
                    # .... A_factor_values is a dense tensor with the size of [num_edge, 1]
                    A_factor_values = torch.sum(
                        head_factor_embeddings * torch.tanh(tail_factor_embeddings),
                        dim=1,
                        keepdim=True,
                    )

                    # update the attentive weights
                    A_iter_values.append(A_factor_values)
                A_iter_values = torch.cat(A_iter_values, dim=1)
                # (num_edge, n_factors)
                # add all layer-wise attentive weights up.
                A_values = A_values + A_iter_values

            # sum messages of neighbors, [n_users+n_items, embed_size]
            side_embeddings = torch.cat(layer_embeddings, dim=1)

            ego_embeddings = side_embeddings
            # concatenate outputs of all layers
            all_embeddings += [ego_embeddings.unsqueeze(1)]

        all_embeddings = torch.cat(all_embeddings, dim=1)
        # (num_node, n_layer + 1, embedding_size)
        all_embeddings = torch.mean(all_embeddings, dim=1, keepdim=False)
        # (num_node, embedding_size)

        u_g_embeddings = all_embeddings[: self.total_num_users, :]
        i_g_embeddings = all_embeddings[self.total_num_users :, :]

        return u_g_embeddings, i_g_embeddings
        
    def calculate_loss(self, interaction):
        self.init_restore_e()
        user_all_embeddings, item_all_embeddings = self.forward()
        source_user = interaction[self.SOURCE_USER_ID]
        source_pos = interaction[self.SOURCE_ITEM_ID]
        source_neg = interaction[self.SOURCE_NEG_ITEM_ID]

        target_user = interaction[self.TARGET_USER_ID]
        target_pos = interaction[self.TARGET_ITEM_ID]
        target_neg = interaction[self.TARGET_NEG_ITEM_ID]
        losses = []

        source_u_embeddings = user_all_embeddings[source_user]
        source_i_pos_embeddings = item_all_embeddings[source_pos]
        source_i_neg_embeddings = item_all_embeddings[source_neg]

        target_u_embeddings = user_all_embeddings[target_user]
        target_i_pos_embeddings = item_all_embeddings[target_pos]
        target_i_neg_embeddings = item_all_embeddings[target_neg]

        # calculate BPR Loss in source domain
        pos_scores = torch.mul(source_u_embeddings, source_i_pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(source_u_embeddings, source_i_neg_embeddings).sum(dim=1)
        source_bpr_loss = self.loss(pos_scores, neg_scores)

        # calculate Reg Loss in source domain
        u_ego_embeddings = self.user_embedding(source_user)
        i_pos_ego_embeddings = self.item_embedding(source_pos)
        i_neg_ego_embeddings = self.item_embedding(source_neg)
        source_reg_loss = self.reg_loss(u_ego_embeddings, i_pos_ego_embeddings, i_neg_ego_embeddings)

        source_loss = source_bpr_loss + self.reg_weight * source_reg_loss
        losses.append(source_loss)

        # calculate BPR Loss in target domain
        pos_scores = torch.mul(target_u_embeddings, target_i_pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(target_u_embeddings, target_i_neg_embeddings).sum(dim=1)
        target_bpr_loss = self.loss(pos_scores, neg_scores)

        # calculate Reg Loss in target domain
        u_ego_embeddings = self.user_embedding(target_user)
        i_pos_ego_embeddings = self.item_embedding(target_pos)
        i_neg_ego_embeddings = self.item_embedding(target_neg)
        target_reg_loss = self.reg_loss(u_ego_embeddings,i_pos_ego_embeddings, i_neg_ego_embeddings)

        target_loss = target_bpr_loss + self.reg_weight * target_reg_loss
        losses.append(target_loss)

        # cul regularized
        if self.n_factors > 1 and self.cor_weight > 1e-9:
            cor_users, cor_items = sample_cor_samples(
                self.total_num_users, self.total_num_items, self.cor_batch_size
            )
            cor_users = torch.LongTensor(cor_users).to(self.device)
            cor_items = torch.LongTensor(cor_items).to(self.device)
            cor_u_embeddings = user_all_embeddings[cor_users]
            cor_i_embeddings = item_all_embeddings[cor_items]
            cor_loss = self.cor_weight * self.create_cor_loss(cor_u_embeddings, cor_i_embeddings)
            losses.append(cor_loss)

        return tuple(losses)

    def create_cor_loss(self, cor_u_embeddings, cor_i_embeddings):
        r"""Calculate the correlation loss for a sampled users and items.

        Args:
            cor_u_embeddings (torch.cuda.FloatTensor): (cor_batch_size, n_factors)
            cor_i_embeddings (torch.cuda.FloatTensor): (cor_batch_size, n_factors)

        Returns:
            torch.Tensor : correlation loss.
        """
        cor_loss = None

        ui_embeddings = torch.cat((cor_u_embeddings, cor_i_embeddings), dim=0)
        ui_factor_embeddings = torch.chunk(ui_embeddings, self.n_factors, 1)

        for i in range(0, self.n_factors - 1):
            x = ui_factor_embeddings[i]
            # (M + N, emb_size / n_factor)
            y = ui_factor_embeddings[i + 1]
            # (M + N, emb_size / n_factor)
            if i == 0:
                cor_loss = self._create_distance_correlation(x, y)
            else:
                cor_loss += self._create_distance_correlation(x, y)

        cor_loss /= (self.n_factors + 1.0) * self.n_factors / 2

        return cor_loss

    def _create_distance_correlation(self, X1, X2):
        def _create_centered_distance(X):
            """
            X: (batch_size, dim)
            return: X - E(X)
            """
            # calculate the pairwise distance of X
            # .... A with the size of [batch_size, embed_size/n_factors]
            # .... D with the size of [batch_size, batch_size]
            r = torch.sum(X * X, dim=1, keepdim=True)
            # (N, 1)
            # (x^2 - 2xy + y^2) -> l2 distance between all vectors
            value = r - 2 * torch.mm(X, X.T + r.T)
            zero_value = torch.zeros_like(value)
            value = torch.where(value > 0.0, value, zero_value)
            D = torch.sqrt(value + 1e-8)

            # # calculate the centered distance of X
            # # .... D with the size of [batch_size, batch_size]
            # matrix - average over row - average over col + average over matrix
            D = (
                D
                - torch.mean(D, dim=0, keepdim=True)
                - torch.mean(D, dim=1, keepdim=True)
                + torch.mean(D)
            )
            return D

        def _create_distance_covariance(D1, D2):
            # calculate distance covariance between D1 and D2
            n_samples = float(D1.size(0))
            value = torch.sum(D1 * D2) / (n_samples * n_samples)
            zero_value = torch.zeros_like(value)
            value = torch.where(value > 0.0, value, zero_value)
            dcov = torch.sqrt(value + 1e-8)
            return dcov

        D1 = _create_centered_distance(X1)
        D2 = _create_centered_distance(X2)

        dcov_12 = _create_distance_covariance(D1, D2)
        dcov_11 = _create_distance_covariance(D1, D1)
        dcov_22 = _create_distance_covariance(D2, D2)

        # calculate the distance correlation
        value = dcov_11 * dcov_22
        zero_value = torch.zeros_like(value)
        value = torch.where(value > 0.0, value, zero_value)
        dcor = dcov_12 / (torch.sqrt(value) + 1e-10)
        return dcor

    def full_sort_predict_source(self, interaction):
        user = interaction[self.SOURCE_USER_ID]

        restore_user_e, restore_item_e, _, _ = self.get_restore_e()
        u_embeddings = restore_user_e[user]
        i_embeddings = restore_item_e[self.target_num_items - self.overlapped_num_items:]

        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))
        return scores.view(-1)

    def full_sort_predict_target(self, interaction):
        user = interaction[self.TARGET_USER_ID]

        _, _, restore_user_e, restore_item_e = self.get_restore_e()
        u_embeddings = restore_user_e[user]
        i_embeddings = restore_item_e[:self.target_num_items]

        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))
        return scores.view(-1)

    def init_restore_e(self):
        # clear the storage variable when training
        if self.source_restore_user_e is not None or self.source_restore_item_e is not None:
            self.source_restore_user_e, self.source_restore_item_e = None, None
        if self.target_restore_user_e is not None or self.target_restore_item_e is not None:
            self.target_restore_user_e, self.target_restore_item_e = None, None

    def get_restore_e(self):
        if self.source_restore_user_e is None or self.source_restore_item_e is None or self.target_restore_user_e is None or self.target_restore_item_e is None:
            self.source_restore_user_e, self.source_restore_item_e = self.forward()
            self.target_restore_user_e, self.target_restore_item_e = self.source_restore_user_e, self.source_restore_item_e
        return self.source_restore_user_e, self.source_restore_item_e, self.target_restore_user_e, self.target_restore_item_e

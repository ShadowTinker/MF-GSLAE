import math
import torch
import random
import numpy as np
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul
import torch.nn as nn
from torch_geometric.utils import degree
from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import MLPLayers

BIG_CONSTANT = 1e8

class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, real_value_mask : torch.Tensor):
        binarized_mask = real_value_mask.clone()
        binarized_mask[binarized_mask > 0] = 1.
        binarized_mask[binarized_mask < 0] = 0.
        binarized_mask = binarized_mask.requires_grad_(True)
        return binarized_mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

# class Binarize(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, real_value_mask : torch.Tensor, budget = None):
#         # binarized_mask = (real_value_mask >= 0).float().requires_grad_(True)
#         if budget == None:
#             binarized_mask = real_value_mask.clone()
#             binarized_mask[binarized_mask > 0] = 1.
#             binarized_mask[binarized_mask < 0] = 0.
#         else:
#             _, idx = torch.topk(real_value_mask, budget)
#             binarized_mask = torch.zeros_like(real_value_mask, device=real_value_mask.device)
#             binarized_mask[0, idx] = 1.
#         binarized_mask = binarized_mask.requires_grad_(True)
#         return binarized_mask

#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output, None

# class Binarize(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, real_value_mask):
#         grad_direction = (real_value_mask > 0).float()
#         grad_direction[grad_direction == 0] = -1
#         ctx.save_for_backward(real_value_mask, grad_direction)
#         binarized_mask = real_value_mask.clone()
#         binarized_mask[binarized_mask > 0] = 1.
#         binarized_mask[binarized_mask <= 0] = 1.
#         binarized_mask = binarized_mask.requires_grad_(True)
#         return binarized_mask
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_input = None
#         _, grad_direction = ctx.saved_tensors
#         if ctx.needs_input_grad[0]:
#             grad_input = grad_direction * grad_output
#         return grad_input, None

def softmax_kernel(data, is_query, projection_matrix, eps=1e-4):
    data_normalizer = (data.shape[-1] ** -0.25)
    ratio = (projection_matrix.shape[0] ** -0.5)

    data_dash = torch.einsum("nhd,md->nhm", (data_normalizer * data), projection_matrix) # perform projection
    diag_data = (data ** 2).sum(-1)

    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(-1)
    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=-1, keepdim=True).detach()) + eps
        )
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=(-1, -2), keepdim=True).detach()) + eps
        )
    return data_dash

class KernelAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_factor,
                    nb_random_features=10, weight=True):
        super(KernelAttention, self).__init__()
        if weight:
            self.Wk = nn.ModuleList()
            self.Wq = nn.ModuleList()
            self.Wv = nn.ModuleList()
            for _ in range(num_factor):
                self.Wk.append(nn.Linear(in_channels // num_factor, out_channels // num_factor))
                self.Wq.append(nn.Linear(in_channels // num_factor, out_channels // num_factor))
                self.Wv.append(nn.Linear(in_channels // num_factor, out_channels // num_factor))

        self.out_channels = out_channels
        self.num_factor = num_factor
        self.nb_random_features = nb_random_features
        self.weight = weight

    def reset_parameters(self):
        self.apply(xavier_normal_initialization)

    def forward(self, z, tau):
        query, key, value = torch.zeros_like(z, device=z.device), torch.zeros_like(z, device=z.device), torch.zeros_like(z, device=z.device)
        for head in range(self.num_factor):
            query[:, head] = self.Wq[head](z[:, head])
            key[:, head] = self.Wk[head](z[:, head])
            value[:, head] = self.Wv[head](z[:, head])

        dim = query.shape[-1]
        projection_matrix = create_projection_matrix(self.nb_random_features, dim).to(query.device)
        z_next = kernelized_softmax(query, key, value, projection_matrix, tau)

        z_next = z_next.flatten(-2, -1)
        return z_next.squeeze()

def adj_mul(adj_i, adj, N):
    adj_i_sp = torch.sparse_coo_tensor(adj_i, torch.ones(adj_i.shape[1], dtype=torch.float).to(adj.device), (N, N))
    adj_sp = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1], dtype=torch.float).to(adj.device), (N, N))
    adj_j = torch.sparse.mm(adj_i_sp, adj_sp)
    adj_j = adj_j.coalesce().indices()
    return adj_j

def create_projection_matrix(m, d):
    nb_full_blocks = int(m/d)
    block_list = []
    for _ in range(nb_full_blocks):
        unstructured_block = torch.randn((d, d))
        q, _ = torch.linalg.qr(unstructured_block)
        block_list.append(q.T)
    final_matrix = torch.vstack(block_list)
    multiplier = torch.norm(torch.randn((m, d)), dim=1)
    return torch.matmul(torch.diag(multiplier), final_matrix)

def kernelized_softmax(query, key, value, projection_matrix=None, tau=0.5):
    query = query / math.sqrt(tau)
    key = key / math.sqrt(tau)

    query_kernel = softmax_kernel(query, True, projection_matrix)
    key_kernel = softmax_kernel(key, False, projection_matrix)

    kvs = torch.einsum("nhm,nhd->hmd", key_kernel, value)
    numerator = torch.einsum("nhm,hmd->nhd", query_kernel, kvs)
    denominator = (query_kernel * key_kernel.sum(0, keepdim=True)).sum(-1, keepdim=True)

    z_output = numerator / denominator
    return z_output


# @Time   : 2022/2/13
# @Author : Hui Yu
# @Email  : ishyu@outlook.com

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads))
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class FilterLayer(nn.Module):
    def __init__(self, item_size, lat_dim):
        super(FilterLayer, self).__init__()
        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.linear = nn.Parameter(torch.randn(1, lat_dim // 2 + 1, dtype=torch.float32))
        nn.init.xavier_normal_(self.linear.data)
        # self.LayerNorm = nn.LayerNorm(item_size, eps=1e-12)
        self.out_dropout = nn.Dropout(0.5)
        self.LayerNorm = LayerNorm(lat_dim, eps=1e-12)

    def forward(self, input_tensor):
        # input_tensor: [batch, hidden]
        batch, hidden = input_tensor.shape
        x = torch.fft.rfft2(input_tensor, norm='ortho')
        x = x * self.linear
        hidden_states = torch.fft.irfft2(x, norm='ortho').real
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

# class FilterLayer(nn.Module):
#     def __init__(self, args):
#         super(FilterLayer, self).__init__()
#         # 做完self-attention 做一个前馈全连接 LayerNorm 输出
#         self.complex_weight = nn.Parameter(torch.randn(1, args.max_seq_length//2 + 1, args.hidden_size, 2, dtype=torch.float32) * 0.02)
#         self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
#         self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)


#     def forward(self, input_tensor):
#         # [batch, seq_len, hidden]
#         batch, seq_len, hidden = input_tensor.shape
#         x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
#         weight = torch.view_as_complex(self.complex_weight)
#         x = x * weight
#         sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')
#         hidden_states = self.out_dropout(sequence_emb_fft)
#         hidden_states = self.LayerNorm(hidden_states + input_tensor)

#         return hidden_states

class Intermediate(nn.Module):
    def __init__(self, lat_dim):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(lat_dim, lat_dim * 4)
        self.intermediate_act_fn = gelu

        self.dense_2 = nn.Linear(4 * lat_dim, lat_dim)
        self.LayerNorm = LayerNorm(lat_dim, eps=1e-12)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class Layer(nn.Module):
    def __init__(self, item_size, lat_dim):
        super(Layer, self).__init__()
        # self.no_filters = args.no_filters
        # self.attention = SelfAttention(item_size, lat_dim)
        self.filterlayer = FilterLayer(item_size, lat_dim)
        self.intermediate = Intermediate(lat_dim)

    def forward(self, hidden_states, attention_mask=None):
        # hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = self.filterlayer(hidden_states)

        intermediate_output = self.intermediate(hidden_states)
        return intermediate_output

class Encoder(nn.Module):
    def __init__(self, item_size, lat_dim):
        super(Encoder, self).__init__()
        layer = Layer(item_size, lat_dim)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(1)])

    def forward(self, hidden_states, attention_mask=None):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return F.normalize(hidden_states)
    


class MultiHeadAttention(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)

        attention_scores = attention_scores / self.sqrt_attention_head_size
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        
        return hidden_states, attention_probs


class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.

    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer

    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer

    """

    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": F.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class TransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
        self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
        layer_norm_eps
    ):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        attention_output, att = self.multi_head_attention(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output, att


class TransformerEncoder(nn.Module):
    r""" One TransformerEncoder consists of several TransformerLayers.

        - n_layers(num): num of transformer layers in transformer encoder. Default: 2
        - n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        - hidden_size(num): the input and output hidden size. Default: 64
        - inner_size(num): the dimensionality in feed-forward layer. Default: 256
        - hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        - attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        - hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12

    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act='gelu',
        layer_norm_eps=1e-12
    ):

        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer(
            n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask=0, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        att_list = []
        for layer_module in self.layer:
            hidden_states, att = layer_module(hidden_states, attention_mask)
            att_list.append(att)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, att_list
    
class MyTrmEncoderLayer(torch.nn.TransformerEncoderLayer):
    def forward(self, src, src_mask = None, src_key_padding_mask = None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(src_key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        why_not_sparsity_fast_path = ''
        if not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first :
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif not self.self_attn._qkv_same_embed_dim :
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm1.eps == self.norm2.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src_mask is not None:
            why_not_sparsity_fast_path = "src_mask is not supported for fastpath"
        elif src.is_nested and src_key_padding_mask is not None:
            why_not_sparsity_fast_path = "src_key_padding_mask is not supported with NestedTensor input for fastpath"
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )

            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all((x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if not why_not_sparsity_fast_path:
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    self.norm_first,
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    # TODO: if src_mask and src_key_padding_mask merge to single 4-dim mask
                    src_mask if src_mask is not None else src_key_padding_mask,
                    1 if src_key_padding_mask is not None else
                    0 if src_mask is not None else
                    None,
                )


        x = src
        if self.norm_first:
            x_att, attention = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + x_att
            x = x + self._ff_block(self.norm2(x))
        else:
            x_att, attention = self._sa_block(x, src_mask, src_key_padding_mask)
            x = self.norm1(x + x_att)
            x = x + x_att
            x = self.norm2(x + self._ff_block(x))

        return x, attention

    # self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask):
        x, attention = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True)
        return self.dropout1(x), attention

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
    
class MyTransformerEncoder(nn.TransformerEncoder):
    r"""TransformerEncoder is a stack of N encoder layers. Users can build the
    BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def forward(self, src, mask = None, src_key_padding_mask = None):
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(src_key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")
        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ''
        str_first_layer = "self.layers[0]"
        if not isinstance(first_layer, torch.nn.TransformerEncoderLayer):
            why_not_sparsity_fast_path = f"{str_first_layer} was not TransformerEncoderLayer"
        elif first_layer.norm_first :
            why_not_sparsity_fast_path = f"{str_first_layer}.norm_first was True"
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not first_layer.self_attn.batch_first:
            why_not_sparsity_fast_path = f" {str_first_layer}.self_attn.batch_first was not True"
        elif not first_layer.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = f"{str_first_layer}.self_attn._qkv_same_embed_dim was not True"
        elif not first_layer.activation_relu_or_gelu:
            why_not_sparsity_fast_path = f" {str_first_layer}.activation_relu_or_gelu was not True"
        elif not (first_layer.norm1.eps == first_layer.norm2.eps) :
            why_not_sparsity_fast_path = f"{str_first_layer}.norm1.eps was not equal to {str_first_layer}.norm2.eps"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif not self.enable_nested_tensor:
            why_not_sparsity_fast_path = "enable_nested_tensor was not True"
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        elif (((not hasattr(self, "mask_check")) or self.mask_check)
                and not torch._nested_tensor_from_mask_left_aligned(src, src_key_padding_mask.logical_not())):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = "src_key_padding_mask and mask were both supplied"
        elif first_layer.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )

            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not (src.is_cuda or 'cpu' in str(src.device)):
                why_not_sparsity_fast_path = "src is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not(), mask_check=False)
                src_key_padding_mask_for_layers = None
        att_list = []
        for mod in self.layers:
            output, att = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask_for_layers)
            att_list.append(att)

        if convert_to_nested:
            output = output.to_padded_tensor(0.)

        if self.norm is not None:
            output = self.norm(output)

        return output, att_list
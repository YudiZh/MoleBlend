import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.modules import (
    FairseqDropout,
)

from fairseq import utils
from fairseq.modules import LayerNorm, Fp32LayerNorm

from torch import Tensor
from typing import Callable, Tuple

from einops import rearrange, einsum


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class AtomFeature(nn.Module):
    """
    Compute atom features for each atom in the molecule.
    """

    def __init__(self, num_heads, num_atoms, num_in_degree, num_out_degree, hidden_dim, n_layers, no_2d=False):
        super(AtomFeature, self).__init__()
        self.num_heads = num_heads
        self.num_atoms = num_atoms
        self.no_2d = no_2d

        self.atom_encoder = nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_dim, padding_idx=0)

        self.graph_token = nn.Embedding(1, hidden_dim)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data, mask_2d=None):
        x, in_degree, out_degree = batched_data['x'], batched_data['in_degree'], batched_data['out_degree']
        n_graph, n_node = x.size()[:2]  
        node_feature = self.atom_encoder(x).sum(dim=-2) 

        degree_feature = None
        if not self.no_2d:  
            degree_feature = self.in_degree_encoder(in_degree) + self.out_degree_encoder(
                out_degree)  
            if mask_2d is not None:
                degree_feature = degree_feature * mask_2d[:, None, None]

        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)  

        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)  

        return graph_node_feature, degree_feature


class MoleculeAttnBias(nn.Module):
    """
    Compute attention bias for each head.
    """

    def __init__(self, num_heads, num_atoms, num_edges, num_spatial, num_edge_dis, hidden_dim, edge_type,
                 multi_hop_max_dist, n_layers, no_2d=False):
        super(MoleculeAttnBias, self).__init__()
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist
        self.no_2d = no_2d

        self.edge_encoder = nn.Embedding(num_edges + 1, num_heads, padding_idx=0)

        self.edge_type = edge_type
        if self.edge_type == 'multi_hop':
            self.edge_dis_encoder = nn.Embedding(
                num_edge_dis * num_heads * num_heads, 1)
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)  # 512 x 32

        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data, mask_2d=None):
        attn_bias, spatial_pos, x = batched_data['attn_bias'], batched_data['spatial_pos'], batched_data['x']
        edge_input, attn_edge_type = batched_data['edge_input'], batched_data['attn_edge_type']

        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1) 


        if not self.no_2d:  
            spatial_pos_bias_nomask = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1,
                                                                                    2)  
            if mask_2d is not None:
                spatial_pos_bias = spatial_pos_bias_nomask * mask_2d[:, None, None, None]
            else:
                spatial_pos_bias = spatial_pos_bias_nomask
        else:
            spatial_pos_bias_nomask = None
            spatial_pos_bias = None

        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        if not self.no_2d:

            # edge feature
            if self.edge_type == 'multi_hop':
                spatial_pos_ = spatial_pos.clone()
                spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
                spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
                if self.multi_hop_max_dist > 0:
                    spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                    edge_input = edge_input[:, :, :, :self.multi_hop_max_dist, :]  
                edge_input = self.edge_encoder(edge_input).mean(-2)  
                max_dist = edge_input.size(-2)
                edge_input_flat = edge_input.permute(
                    3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)  
                edge_input_flat = torch.bmm(edge_input_flat, self.edge_dis_encoder.weight.reshape(
                    -1, self.num_heads, self.num_heads)[:max_dist, :, :]) 
                edge_input = edge_input_flat.reshape(
                    max_dist, n_graph, n_node, n_node, self.num_heads).permute(1, 2, 3, 0,
                                                                               4)  
                edge_input = (edge_input.sum(-2) /
                              (spatial_pos_.float().unsqueeze(-1))).permute(0, 3, 1,
                                                                            2)  

            else:
                edge_input = self.edge_encoder(
                    attn_edge_type).mean(-2).permute(0, 3, 1, 2)

            edge_input_nomask = edge_input
            if mask_2d is not None:
                edge_input = edge_input_nomask * mask_2d[:, None, None, None]  # [B, C, n_atom, n_atom]
            else:
                edge_input = edge_input_nomask
        else:
            edge_input_nomask = None
            edge_input = None

        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        return graph_attn_bias, spatial_pos_bias, spatial_pos_bias_nomask, edge_input, edge_input_nomask





class Molecule3DBias(nn.Module):
    """
        Compute 3D attention bias according to the position information for each head.
        """

    def __init__(self, num_heads, num_edges, n_layers, embed_dim, num_kernel, no_share_rpe=False):
        super(Molecule3DBias, self).__init__()
        self.num_heads = num_heads
        self.num_edges = num_edges
        self.n_layers = n_layers
        self.no_share_rpe = no_share_rpe
        self.num_kernel = num_kernel
        self.embed_dim = embed_dim

        rpe_heads = self.num_heads * self.n_layers if self.no_share_rpe else self.num_heads
        self.gbf = GaussianLayer(self.num_kernel, num_edges)
        self.gbf_proj = NonLinear(self.num_kernel, rpe_heads)

        if self.num_kernel != self.embed_dim:
            self.edge_proj = nn.Linear(self.num_kernel, self.embed_dim)
        else:
            self.edge_proj = None

    def forward(self, batched_data):
        pos, x, node_type_edge = batched_data['pos'], batched_data['x'], batched_data['node_type_edge']

        padding_mask = x.eq(0).all(dim=-1) 
        n_graph, n_node, _ = pos.shape
        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)  
        dist = delta_pos.norm(dim=-1).view(-1, n_node, n_node)  
        delta_pos /= dist.unsqueeze(-1) + 1e-5  
        edge_feature = self.gbf(dist,
                                torch.zeros_like(dist).long() if node_type_edge is None else node_type_edge.long())
        gbf_result = self.gbf_proj(edge_feature)
        graph_attn_bias = gbf_result 

        graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous() 
        graph_attn_bias.masked_fill_(
            padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
        )

        edge_feature = edge_feature.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool), 0.0
        )

        sum_edge_features = edge_feature.sum(dim=-2)
        merge_edge_features = self.edge_proj(sum_edge_features) 

        return graph_attn_bias, merge_edge_features, delta_pos


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=512 * 3):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1, padding_idx=0)
        self.bias = nn.Embedding(edge_types, 1, padding_idx=0)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_types):
        mul = self.mul(edge_types).sum(dim=-2)
        bias = self.bias(edge_types).sum(dim=-2)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-2
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(NonLinear, self).__init__()

        if hidden is None:
            hidden = input
        self.layer1 = nn.Linear(input, hidden)
        self.layer2 = nn.Linear(hidden, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x


class AtomTaskHead(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.k_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.v_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.scaling = (embed_dim // num_heads) ** -0.5
        self.force_proj1: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)
        self.force_proj2: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)
        self.force_proj3: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)

        self.dropout_module = FairseqDropout(
            0.1, module_name=self.__class__.__name__
        )

    def forward(
            self,
            query: Tensor,
            attn_bias: Tensor,
            delta_pos: Tensor,
    ) -> Tensor:
        query = query.contiguous().transpose(0, 1)  
        bsz, n_node, _ = query.size()
        q = (
                self.q_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
                * self.scaling
        )  
        k = self.k_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
        attn = q @ k.transpose(-1, -2)  
        attn_probs_float = utils.softmax(
            attn.view(-1, n_node, n_node) + attn_bias.contiguous().view(-1, n_node, n_node), dim=-1, onnx_trace=False)
        attn_probs = attn_probs_float.type_as(attn)
        attn_probs = self.dropout_module(attn_probs).view(bsz, self.num_heads, n_node, n_node)
        rot_attn_probs = attn_probs.unsqueeze(-1) * delta_pos.unsqueeze(1).type_as(
            attn_probs
        )  
        rot_attn_probs = rot_attn_probs.permute(0, 1, 4, 2, 3)
        x = rot_attn_probs @ v.unsqueeze(2)  
        x = x.permute(0, 3, 2, 1, 4).contiguous().view(bsz, n_node, 3, -1) 
        f1 = self.force_proj1(x[:, :, 0, :]).view(bsz, n_node, 1)
        f2 = self.force_proj2(x[:, :, 1, :]).view(bsz, n_node, 1)
        f3 = self.force_proj3(x[:, :, 2, :]).view(bsz, n_node, 1)
        cur_force = torch.cat([f1, f2, f3], dim=-1).float()
        return cur_force


class TransformN2Head(nn.Module):
    def __init__(
            self,
            d_input,
            d_hidden,
    ):
        super(TransformN2Head, self).__init__()
        # single head 
        self.d_input = d_input
        self.d_hidden = d_hidden

        self.column_head = nn.Linear(d_input, d_hidden)
        self.column_act_fn = nn.GELU()
        self.column_LayerNorm = nn.LayerNorm(d_hidden, eps=1e-12)

        self.row_head = nn.Linear(d_input, d_hidden)
        self.row_act_fn = nn.GELU()
        self.row_LayerNorm = nn.LayerNorm(d_hidden, eps=1e-12)

        self.scaling = d_input ** -0.5

    def forward(
            self,
            hidden_states,
    ):
        """

        :param hidden_states: [n_atom, B, C], output of transformer
        :return: outer_prod: [B, n_atom, n_atom, n_hid**2]
        """

        hidden_states = rearrange(hidden_states, 'n_atom B C -> B n_atom C')  

        row_states = self.row_head(hidden_states)  
        row_states = self.row_act_fn(row_states)
        row_states = self.row_LayerNorm(row_states)
        column_states = self.column_head(hidden_states)  
        column_states = self.column_act_fn(column_states)
        column_states = self.column_LayerNorm(column_states)

        row_states = rearrange(row_states, 'B n_atom d_hid -> B n_atom d_hid 1')
        column_states = rearrange(column_states, 'B n_atom d_hid -> B 1 d_hid n_atom')
        outer_prod = einsum(row_states, column_states, 'B n1 c1 d1, B d1 c2 n2 -> B n1 c1 c2 n2')
        outer_prod = rearrange(outer_prod, 'B n1 c1 c2 n2 -> B n1 n2 (c1 c2)')

        return outer_prod  


class PredictSPDHead(nn.Module):
    def __init__(self, d_input, d_hidden, max_dist):
        super(PredictSPDHead, self).__init__()
        self.transform_n2_head = TransformN2Head(
            d_input=d_input,
            d_hidden=d_hidden,
        )
        self.max_dist = max_dist
        self.linear = nn.Linear(d_hidden ** 2, max_dist)

    def forward(self, x, ):
        """

        :param x: [n_atom, B, C]
        :return:
        """
        relation = self.transform_n2_head(x)  
        logit = self.linear(relation)  

        return logit  


class PredictEdgeHead(nn.Module):
    def __init__(self, d_input, d_hidden, feature1_num, feature2_num, feature3_num, max_dist=None):
        super().__init__()
        self.transform_n2_head = TransformN2Head(
            d_input=d_input,
            d_hidden=d_hidden,
        )
        self.linear_map_feature = nn.Linear(d_hidden ** 2, 256) 
        self.max_dist = max_dist
        self.feature1_num = feature1_num
        self.feature2_num = feature2_num
        self.feature3_num = feature3_num
        
        assert max_dist is not None
        self.linear_map_feature: Callable[[Tensor], Tensor] = nn.Linear(d_hidden ** 2, feature1_num*feature2_num*feature3_num*max_dist)  
        
    
    def forward(self, x):
        relation = self.transform_n2_head(x)  
        edge_type_hidden_states = self.linear_map_feature(relation)  
        B, n_node = edge_type_hidden_states.size(0), edge_type_hidden_states.size(1)
        edge_feature_pred = edge_type_hidden_states.view(B, n_node, n_node, self.max_dist, self.feature1_num*self.feature2_num*self.feature3_num)
        return edge_feature_pred
        


class Predict3DHead(nn.Module):
    def __init__(self, d_input, d_hidden, pred_hidden_dim):
        super().__init__()
        self.transform_n2_head = TransformN2Head(
            d_input=d_input,
            d_hidden=d_hidden,
        )
        self.linear = nn.Linear(d_hidden ** 2, pred_hidden_dim)
        self.force_proj1: Callable[[Tensor], Tensor] = nn.Linear(pred_hidden_dim, 1)
        self.force_proj2: Callable[[Tensor], Tensor] = nn.Linear(pred_hidden_dim, 1)
        self.force_proj3: Callable[[Tensor], Tensor] = nn.Linear(pred_hidden_dim, 1)

    def forward(self, x):
        relation = self.transform_n2_head(x)  
        delta_pos_hidden_states = self.linear(relation)  
        bsz, n_node = delta_pos_hidden_states.size(0), delta_pos_hidden_states.size(1)
        f1 = self.force_proj1(delta_pos_hidden_states).view(bsz, n_node, n_node, 1)
        f2 = self.force_proj2(delta_pos_hidden_states).view(bsz, n_node, n_node, 1)
        f3 = self.force_proj3(delta_pos_hidden_states).view(bsz, n_node, n_node, 1)
        delta_pos_pred = torch.cat([f1, f2, f3], dim=-1).float()
        return delta_pos_pred  


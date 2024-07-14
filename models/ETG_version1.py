import math
from typing import Dict, Optional

import torch

from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import ExtractIr, FullyConnectedNet, Gate
from e3nn.o3 import FullyConnectedTensorProduct, TensorProduct
from e3nn.util.jit import compile_mode
from e3nn.math import soft_unit_step
import torch
from e3nn import nn
from torch_scatter import scatter
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_cluster import radius_graph
from torch_geometric.data import Data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class GraphSelfAttention(torch.nn.Module):
    def __init__(self, irreps_input, irreps_query, irreps_key, irreps_output, max_radius, number_of_basis=10):
        super(GraphSelfAttention, self).__init__()
        self.irreps_input = irreps_input
        self.irreps_query = irreps_query
        self.irreps_key = irreps_key
        self.irreps_output = irreps_output
        self.irreps_sh = o3.Irreps.spherical_harmonics(3)
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis

        self.h_q = o3.Linear(self.irreps_input, self.irreps_query)
        self.tp_k = o3.FullyConnectedTensorProduct(self.irreps_input, self.irreps_sh, self.irreps_key,
                                                   shared_weights=False)
        self.fc_k = nn.FullyConnectedNet([number_of_basis, 16, self.tp_k.weight_numel], act=torch.nn.functional.silu)

        self.tp_v = o3.FullyConnectedTensorProduct(self.irreps_input, self.irreps_sh, self.irreps_output,
                                                   shared_weights=False)
        self.fc_v = nn.FullyConnectedNet([number_of_basis, 16, self.tp_v.weight_numel], act=torch.nn.functional.silu)
        self.dot = o3.FullyConnectedTensorProduct(self.irreps_query, self.irreps_key, "0e")

    def forward(self, x, pos):
        pos = pos
        f = x
        #print(f.device)
        edge_src, edge_dst = radius_graph(pos, self.max_radius)
        edge_vec = pos[edge_src] - pos[edge_dst]
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedded = soft_one_hot_linspace(
            edge_length,
            start=0.0,
            end=self.max_radius,
            number=self.number_of_basis,
            basis='smooth_finite',
            cutoff=True  # goes (smoothly) to zero at `start` and `end`
        )
        edge_length_embedded = edge_length_embedded.mul(self.number_of_basis ** 0.5)
        edge_weight_cutoff = soft_unit_step(10 * (1 - edge_length / self.max_radius))


        edge_sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, True, normalization='component')

        h_q = o3.Linear(self.irreps_input, self.irreps_query)
        f = f.to('cpu')
        q = h_q(f)
        f = f.to(device)
        q = q.to(device)
        k = self.tp_k(f[edge_src], edge_sh, self.fc_k(edge_length_embedded))
        v = self.tp_v(f[edge_src], edge_sh, self.fc_v(edge_length_embedded))

        exp = edge_weight_cutoff[:, None] * self.dot(q[edge_dst], k).exp()
        z = scatter(exp, edge_dst, dim=0, dim_size=len(f))
        z[z == 0] = 1
        alpha = exp / z[edge_dst]

        return scatter(alpha.relu().sqrt() * v, edge_dst, dim=0, dim_size=len(f))





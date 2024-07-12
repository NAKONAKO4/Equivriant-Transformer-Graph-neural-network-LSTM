import torch
from torch.nn import Parameter
import torch.nn.functional as F
from ETG_version1 import GraphSelfAttention
from e3nn import o3
from torch_geometric.nn.inits import glorot, zeros
class ETGplusLSTM(torch.nn.Module):
    def __init__(self,
                 irreps_input: o3.Irreps,
                 irreps_query: o3.Irreps,
                 irreps_key: o3.Irreps,
                 irreps_output: o3.Irreps,
                 max_radius: float,
                 bias: bool = True,):
        super(ETGplusLSTM, self).__init__()
        self.irreps_input = irreps_input
        self.irreps_query = irreps_query
        self.irreps_key = irreps_key
        self.irreps_output = irreps_output
        self.max_radius = max_radius
        self.bias = bias
        self._create_parameters_and_layers()
        self._set_parameters()

    def _create_input_gate(self):
        self.etg_x_i = torch.nn.ModuleList()
        self.etg_h_i = torch.nn.ModuleList()
        self.etg_x_i.append(GraphSelfAttention(self.irreps_input,
                                               self.irreps_query,
                                               self.irreps_key,
                                               self.irreps_output,
                                               self.max_radius,))
        self.etg_h_i.append(GraphSelfAttention(self.irreps_input,
                                               self.irreps_query,
                                               self.irreps_key,
                                               self.irreps_output,
                                               self.max_radius, ))
        self.w_c_i = Parameter(torch.Tensor(1, self.irreps_output))
        self.b_i = Parameter(torch.Tensor(1, self.irreps_output))
    def _create_forget_gate(self):
        self.etg_x_f = torch.nn.ModuleList()
        self.etg_h_f = torch.nn.ModuleList()
        self.etg_x_f.append(GraphSelfAttention(self.irreps_input,
                                               self.irreps_query,
                                               self.irreps_key,
                                               self.irreps_output,
                                               self.max_radius,))
        self.etg_h_f.append(GraphSelfAttention(self.irreps_input,
                                               self.irreps_query,
                                               self.irreps_key,
                                               self.irreps_output,
                                               self.max_radius, ))
        self.w_c_f = Parameter(torch.Tensor(1, self.irreps_output))
        self.b_f = Parameter(torch.Tensor(1, self.irreps_output))

    def _create_cell_gate(self):
        self.etg_x_c = torch.nn.ModuleList()
        self.etg_h_c = torch.nn.ModuleList()
        self.etg_x_c.append(GraphSelfAttention(self.irreps_input,
                                               self.irreps_query,
                                               self.irreps_key,
                                               self.irreps_output,
                                               self.max_radius,))
        self.etg_h_c.append(GraphSelfAttention(self.irreps_input,
                                               self.irreps_query,
                                               self.irreps_key,
                                               self.irreps_output,
                                               self.max_radius, ))
        self.b_c = Parameter(torch.Tensor(1, self.irreps_output))

    def _create_output_gate(self):
        self.etg_x_o = torch.nn.ModuleList()
        self.etg_h_o = torch.nn.ModuleList()
        self.etg_x_o.append(GraphSelfAttention(self.irreps_input,
                                               self.irreps_query,
                                               self.irreps_key,
                                               self.irreps_output,
                                               self.max_radius,))
        self.etg_h_o.append(GraphSelfAttention(self.irreps_input,
                                               self.irreps_query,
                                               self.irreps_key,
                                               self.irreps_output,
                                               self.max_radius, ))
        self.w_c_o = Parameter(torch.Tensor(1, self.irreps_output))
        self.b_o = Parameter(torch.Tensor(1, self.irreps_output))

    def _create_parameters_and_layers(self):
        self._create_input_gate()
        self._create_forget_gate()
        self._create_cell_gate()
        self._create_output_gate()

    def _set_parameters(self):
        glorot(self.w_c_i)
        glorot(self.w_c_f)
        glorot(self.w_c_o)
        zeros(self.b_i)
        zeros(self.b_f)
        zeros(self.b_c)
        zeros(self.b_o)

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _set_cell_state(self, X, C):
        if C is None:
            C = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return C

    def _calculate_input_gate(self, X, edge_index, H, C):
        I = self.etg_x_i[0](X, edge_index)
        I = I + self.etg_h_i[0](H, edge_index)
        I = I + (self.w_c_i * C)
        I = I + self.b_i
        I = torch.sigmoid(I)
        return I

    def _calculate_forget_gate(self, X, edge_index, H, C):
        F = self.etg_x_f[0](X, edge_index)
        F = F + self.etg_h_f[0](H, edge_index)
        F = F + (self.w_c_f * C)
        F = F + self.b_f
        F = torch.sigmoid(F)
        return F

    def _calculate_cell_state(self, X, edge_index, H, C, I, F):
        T = self.etg_x_c[0](X, edge_index)
        T = T + self.etg_h_c[0](H, edge_index)
        T = T + self.b_c
        T = torch.tanh(T)
        C = F * C + I * T
        return C

    def _calculate_output_gate(self, X, edge_index, H, C):
        O = self.etg_x_o[0](X, edge_index)
        O = O + self.etg_h_o[0](H, edge_index)
        O = O + (self.w_c_o * C)
        O = O + self.b_o
        O = torch.sigmoid(O)
        return O

    def _calculate_hidden_state(self, O, C):
        H = O * torch.tanh(C)
        return H

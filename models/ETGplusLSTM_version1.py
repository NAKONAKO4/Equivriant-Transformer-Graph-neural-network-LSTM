import torch
from torch.nn import Parameter
import torch.nn.functional as F
from .ETG_version1  import GraphSelfAttention
from e3nn import o3
from torch_geometric.nn.inits import glorot, zeros
from torch.nn import CrossEntropyLoss

HIDDEN_CHANNELS = 128
NUM_FEATURES = 11
POSITIVE_RATE = 0.125 #hyperparameter
NUM_CLASSES = 2
LR = 1e-3
DROPOUT = 0.3
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
        self.etg_h_i.append(GraphSelfAttention(self.irreps_output,
                                               self.irreps_query,
                                               self.irreps_key,
                                               self.irreps_output,
                                               self.max_radius, ))
        self.w_c_i = Parameter(torch.Tensor(1, self.irreps_output.dim))
        self.b_i = Parameter(torch.Tensor(1, self.irreps_output.dim))
    def _create_forget_gate(self):
        self.etg_x_f = torch.nn.ModuleList()
        self.etg_h_f = torch.nn.ModuleList()
        self.etg_x_f.append(GraphSelfAttention(self.irreps_input,
                                               self.irreps_query,
                                               self.irreps_key,
                                               self.irreps_output,
                                               self.max_radius,))
        self.etg_h_f.append(GraphSelfAttention(self.irreps_output,
                                               self.irreps_query,
                                               self.irreps_key,
                                               self.irreps_output,
                                               self.max_radius, ))
        self.w_c_f = Parameter(torch.Tensor(1, self.irreps_output.dim))
        self.b_f = Parameter(torch.Tensor(1, self.irreps_output.dim))

    def _create_cell_gate(self):
        self.etg_x_c = torch.nn.ModuleList()
        self.etg_h_c = torch.nn.ModuleList()
        self.etg_x_c.append(GraphSelfAttention(self.irreps_input,
                                               self.irreps_query,
                                               self.irreps_key,
                                               self.irreps_output,
                                               self.max_radius,))
        self.etg_h_c.append(GraphSelfAttention(self.irreps_output,
                                               self.irreps_query,
                                               self.irreps_key,
                                               self.irreps_output,
                                               self.max_radius, ))
        self.b_c = Parameter(torch.Tensor(1, self.irreps_output.dim))

    def _create_output_gate(self):
        self.etg_x_o = torch.nn.ModuleList()
        self.etg_h_o = torch.nn.ModuleList()
        self.etg_x_o.append(GraphSelfAttention(self.irreps_input,
                                               self.irreps_query,
                                               self.irreps_key,
                                               self.irreps_output,
                                               self.max_radius,))
        self.etg_h_o.append(GraphSelfAttention(self.irreps_output,
                                               self.irreps_query,
                                               self.irreps_key,
                                               self.irreps_output,
                                               self.max_radius, ))
        self.w_c_o = Parameter(torch.Tensor(1, self.irreps_output.dim))
        self.b_o = Parameter(torch.Tensor(1, self.irreps_output.dim))

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
            H = torch.zeros(X.shape[0], self.irreps_output.dim).to(X.device)
        return H

    def _set_cell_state(self, X, C):
        if C is None:
            C = torch.zeros(X.shape[0], self.irreps_output.dim).to(X.device)
        return C

    def _calculate_input_gate(self, X, pos, H, C):
        I = self.etg_x_i[0](X, pos)
        I = I + self.etg_h_i[0](H, pos)
        I = I + (self.w_c_i * C)
        I = I + self.b_i
        I = torch.sigmoid(I)
        return I

    def _calculate_forget_gate(self, X, pos, H, C):
        F = self.etg_x_f[0](X, pos)
        F = F + self.etg_h_f[0](H, pos)
        F = F + (self.w_c_f * C)
        F = F + self.b_f
        F = torch.sigmoid(F)
        return F

    def _calculate_cell_state(self, X, pos, H, C, I, F):
        T = self.etg_x_c[0](X, pos)
        T = T + self.etg_h_c[0](H, pos)
        T = T + self.b_c
        T = torch.tanh(T)
        C = F * C + I * T
        return C

    def _calculate_output_gate(self, X, pos, H, C):
        O = self.etg_x_o[0](X, pos)
        O = O + self.etg_h_o[0](H, pos)
        O = O + (self.w_c_o * C)
        O = O + self.b_o
        O = torch.sigmoid(O)
        return O

    def _calculate_hidden_state(self, O, C):
        H = O * torch.tanh(C)
        return H
    def forward(
        self,
        X: torch.FloatTensor,
        pos: torch.Tensor,
        H: torch.FloatTensor = None,
        C: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If the hidden state and cell state
        matrices are not present when the forward pass is called these are
        initialized with zeros.

        Arg types:
            X(PyTorch Float Tensor): Node features.
            edge_index(PyTorch Long Tensor): Graph edge indices.
            edge_weight(PyTorch Long Tensor, optional): Edge weight vector.
            H(PyTorch Float Tensor, optional): Hidden state matrix for all nodes.
            C(PyTorch Float Tensor, optional): Cell state matrix for all nodes.

        Return types:
            H(PyTorch Float Tensor): Hidden state matrix for all nodes.
            C(PyTorch Float Tensor): Cell state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        C = self._set_cell_state(X, C)
        I = self._calculate_input_gate(X, pos, H, C)
        F = self._calculate_forget_gate(X, pos, H, C)
        C = self._calculate_cell_state(X, pos, H, C, I, F)
        O = self._calculate_output_gate(X, pos, H, C)
        H = self._calculate_hidden_state(O, C)
        return H, C


class MLPModel(torch.nn.Module):
    def __init__(
            self,
            in_size=128,
            hidden_size=256,
            out_size=2
    ):
        super(MLPModel, self).__init__()

        self.hidden_size = hidden_size
        self.in_size = in_size

        self.encoder1 = torch.nn.Linear(self.in_size, self.hidden_size)
        self.encoder2 = torch.nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.sigmoid = torch.nn.Sigmoid()

        self.fc = torch.nn.Linear(self.hidden_size // 2, self.hidden_size // 4)
        self.fc_out = torch.nn.Linear(hidden_size // 4, out_size)
        self.leaky_relu = torch.nn.LeakyReLU()

    def forward(self, x):
        hidden = self.encoder1(x)
        hidden = self.leaky_relu(hidden)

        hidden = self.encoder2(hidden)
        hidden = self.leaky_relu(hidden)

        hidden = self.fc(hidden)
        hidden = self.leaky_relu(hidden)

        return self.fc_out(hidden).squeeze()


class ETG_LSTM_final(torch.nn.Module):
    def __init__(
            self,
            in_channels=NUM_FEATURES,
            hidden_channels=HIDDEN_CHANNELS,
            num_classes=NUM_CLASSES,
            dropout=DROPOUT,
            lr=LR,
            positive_rate=POSITIVE_RATE
    ):
        super(ETG_LSTM_final, self).__init__()
        self.recurrent = ETGplusLSTM(o3.Irreps("11x0e"),
                     o3.Irreps("11x0e"),
                     o3.Irreps("11x0e"),
                     o3.Irreps("22x0e"),
                     10)
        self.dropout = dropout
        self.MLP = MLPModel(in_size=22, hidden_size=44, out_size=num_classes)
        self.optimizer = torch.optim.Adam([
            dict(params=self.recurrent.parameters(), weight_decay=1e-2),
            dict(params=self.MLP.parameters(), weight_decay=5e-4)
        ], lr=lr)
        weights = [1, (1 - positive_rate) / positive_rate]
        class_weights = torch.FloatTensor(weights)
        self.criterion = CrossEntropyLoss(weight=class_weights)

    def forward(self, x, pos, h=None, c=None, output=False):
        x = F.dropout(x, self.dropout, training=self.training)
        h, c = self.recurrent(x, pos, h, c)
        h = F.relu(h)
        if output == True:
            h = F.dropout(h, self.dropout, training=self.training)
            h = self.MLP(h)
        return h, c

import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.inits import glorot, zeros
from Eqvrtransformerlstm import Network
from e3nn import o3
HIDDEN_CHANNELS = 128
NUM_FEATURES = 11
POSITIVE_RATE = 0.125  # hyperparameter
NUM_CLASSES = 2
LR = 1e-3
DROPOUT = 0.3


class GConvLSTM(torch.nn.Module):

    def __init__(
            self,
            irreps_in: o3.Irreps,
            irreps_out: o3.Irreps,
            irreps_hidden: o3.Irreps,
            irreps_node_attr: o3.Irreps,
            irreps_edge_attr: o3.Irreps,
            layers: int,

            max_radius: float,
            num_neighbors: float,
            num_nodes: int,
            radial_layers: int,
            radial_neurons: int,
            num_graphs: int,
            in_channels: int,
            out_channels: int,
            bias: bool = True,

    ):
        super(GConvLSTM, self).__init__()

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.irreps_hidden = irreps_hidden
        self.irreps_node_attr = irreps_node_attr
        self.irreps_edge_attr = irreps_edge_attr
        self.layers = layers
        self.radial_layers = radial_layers
        self.radial_neurons = radial_neurons
        self.num_nodes = num_nodes

        self.max_radius = max_radius
        self.num_neighbors = num_neighbors
        self.num_graphs = num_graphs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self._create_parameters_and_layers()
        self._set_parameters()

    def _create_input_gate_parameters_and_layers(self):
        self.conv_x_i = torch.nn.ModuleList()
        self.conv_x_i.append(SAGEConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            normalize=True,
            bias=self.bias,
        ))
        self.conv_x_i.append(Network(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            irreps_hidden=self.irreps_hidden,
            irreps_node_attr=self.irreps_node_attr,
            irreps_edge_attr=self.irreps_edge_attr,
            layers=self.layers,
            radial_layers=self.radial_layers,
            radial_neurons=self.radial_neurons,
            num_neighbors=self.num_neighbors,
            num_nodes=self.num_nodes,
            max_radius=self.max_radius,
        ))
        self.conv_h_i = torch.nn.ModuleList()
        self.conv_h_i.append(SAGEConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            normalize=True,
            bias=self.bias,
        ))
        self.conv_h_i.append(Network(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            irreps_hidden=self.irreps_hidden,
            irreps_node_attr=self.irreps_node_attr,
            irreps_edge_attr=self.irreps_edge_attr,
            layers=self.layers,
            radial_layers=self.radial_layers,
            radial_neurons=self.radial_neurons,
            num_neighbors=self.num_neighbors,
            num_nodes=self.num_nodes,
            max_radius=self.max_radius,
        ))
        self.w_c_i = Parameter(torch.Tensor(1, self.out_channels))
        self.b_i = Parameter(torch.Tensor(1, self.out_channels))

    def _create_forget_gate_parameters_and_layers(self):
        self.conv_x_f = torch.nn.ModuleList()
        self.conv_x_f.append(SAGEConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            normalize=True,
            bias=self.bias,
        ))
        self.conv_x_f.append(Network(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            irreps_hidden=self.irreps_hidden,
            irreps_node_attr=self.irreps_node_attr,
            irreps_edge_attr=self.irreps_edge_attr,
            layers=self.layers,
            radial_layers=self.radial_layers,
            radial_neurons=self.radial_neurons,
            num_neighbors=self.num_neighbors,
            num_nodes=self.num_nodes,
            max_radius=self.max_radius,
        ))
        self.conv_h_f = torch.nn.ModuleList()
        self.conv_h_f.append(SAGEConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            normalize=True,
            bias=self.bias,
        ))
        self.conv_h_f.append(Network(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            irreps_hidden=self.irreps_hidden,
            irreps_node_attr=self.irreps_node_attr,
            irreps_edge_attr=self.irreps_edge_attr,
            layers=self.layers,
            radial_layers=self.radial_layers,
            radial_neurons=self.radial_neurons,
            num_neighbors=self.num_neighbors,
            num_nodes=self.num_nodes,
            max_radius=self.max_radius,
        ))
        self.w_c_f = Parameter(torch.Tensor(1, self.out_channels))
        self.b_f = Parameter(torch.Tensor(1, self.out_channels))

    def _create_cell_state_parameters_and_layers(self):
        self.conv_x_c = torch.nn.ModuleList()
        self.conv_x_c.append(SAGEConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            normalize=True,
            bias=self.bias,
        ))
        self.conv_x_c.append(Network(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            irreps_hidden=self.irreps_hidden,
            irreps_node_attr=self.irreps_node_attr,
            irreps_edge_attr=self.irreps_edge_attr,
            layers=self.layers,
            radial_layers=self.radial_layers,
            radial_neurons=self.radial_neurons,
            num_neighbors=self.num_neighbors,
            num_nodes=self.num_nodes,
            max_radius=self.max_radius,
        ))
        self.conv_h_c = torch.nn.ModuleList()
        self.conv_h_c.append(SAGEConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            normalize=True,
            bias=self.bias,
        ))
        self.conv_h_c.append(Network(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            irreps_hidden=self.irreps_hidden,
            irreps_node_attr=self.irreps_node_attr,
            irreps_edge_attr=self.irreps_edge_attr,
            layers=self.layers,
            radial_layers=self.radial_layers,
            radial_neurons=self.radial_neurons,
            num_neighbors=self.num_neighbors,
            num_nodes=self.num_nodes,
            max_radius=self.max_radius,
        ))
        self.b_c = Parameter(torch.Tensor(1, self.out_channels))

    def _create_output_gate_parameters_and_layers(self):
        self.conv_x_o = torch.nn.ModuleList()
        self.conv_x_o.append(SAGEConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            normalize=True,
            bias=self.bias,
        ))
        self.conv_x_o.append(Network(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            irreps_hidden=self.irreps_hidden,
            irreps_node_attr=self.irreps_node_attr,
            irreps_edge_attr=self.irreps_edge_attr,
            layers=self.layers,
            radial_layers=self.radial_layers,
            radial_neurons=self.radial_neurons,
            num_neighbors=self.num_neighbors,
            num_nodes=self.num_nodes,
            max_radius=self.max_radius,
        ))
        self.conv_h_o = torch.nn.ModuleList()
        self.conv_h_o.append(SAGEConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            normalize=True,
            bias=self.bias,
        ))
        self.conv_h_o.append(Network(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            irreps_hidden=self.irreps_hidden,
            irreps_node_attr=self.irreps_node_attr,
            irreps_edge_attr=self.irreps_edge_attr,
            layers=self.layers,
            radial_layers=self.radial_layers,
            radial_neurons=self.radial_neurons,
            num_neighbors=self.num_neighbors,
            num_nodes=self.num_nodes,
            max_radius=self.max_radius,
        ))
        self.w_c_o = Parameter(torch.Tensor(1, self.out_channels))
        self.b_o = Parameter(torch.Tensor(1, self.out_channels))

    def _create_parameters_and_layers(self):
        self._create_input_gate_parameters_and_layers()
        self._create_forget_gate_parameters_and_layers()
        self._create_cell_state_parameters_and_layers()
        self._create_output_gate_parameters_and_layers()

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
        I = self.conv_x_i[0](X, edge_index)
        I = I + self.conv_h_i[0](H, edge_index)
        I = I + (self.w_c_i * C)
        I = I + self.b_i
        I = torch.sigmoid(I)
        return I

    def _calculate_forget_gate(self, X, edge_index, H, C):
        F = self.conv_x_f[0](X, edge_index)
        F = F + self.conv_h_f[0](H, edge_index)
        F = F + (self.w_c_f * C)
        F = F + self.b_f
        F = torch.sigmoid(F)
        return F

    def _calculate_cell_state(self, X, edge_index, H, C, I, F):
        T = self.conv_x_c[0](X, edge_index)
        T = T + self.conv_h_c[0](H, edge_index)
        T = T + self.b_c
        T = torch.tanh(T)
        C = F * C + I * T
        return C

    def _calculate_output_gate(self, X, edge_index, H, C):
        O = self.conv_x_o[0](X, edge_index)
        O = O + self.conv_h_o[0](H, edge_index)
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
            edge_index: torch.LongTensor,
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
        I = self._calculate_input_gate(X, edge_index, H, C)
        F = self._calculate_forget_gate(X, edge_index, H, C)
        C = self._calculate_cell_state(X, edge_index, H, C, I, F)
        O = self._calculate_output_gate(X, edge_index, H, C)
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


class GraphSAGE_LSTM(torch.nn.Module):
    def __init__(
            self,
            in_channels=NUM_FEATURES,
            hidden_channels=HIDDEN_CHANNELS,
            num_classes=NUM_CLASSES,
            dropout=DROPOUT,
            lr=LR,
            positive_rate=POSITIVE_RATE
    ):
        super(GraphSAGE_LSTM, self).__init__()
        self.recurrent = GConvLSTM(in_channels, hidden_channels)
        self.dropout = dropout
        self.MLP = MLPModel(in_size=hidden_channels, hidden_size=256, out_size=num_classes)
        self.optimizer = torch.optim.Adam([
            dict(params=self.recurrent.parameters(), weight_decay=1e-2),
            dict(params=self.MLP.parameters(), weight_decay=5e-4)
        ], lr=lr)
        weights = [1, (1 - positive_rate) / positive_rate]
        class_weights = torch.FloatTensor(weights)
        self.criterion = CrossEntropyLoss(weight=class_weights)

    def forward(self, x, edge_index, h=None, c=None, output=False):
        x = F.dropout(x, self.dropout, training=self.training)
        h, c = self.recurrent(x, edge_index, h, c)
        h = F.relu(h)
        if output == True:
            h = F.dropout(h, self.dropout, training=self.training)
            h = self.MLP(h)
        return h, c


class GL_Pair(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.5):
        super(GL_Pair, self).__init__()
        self.recurrent = GConvLSTM(in_channels, hidden_channels)
        self.dropout = dropout
        # delete mlp

    def forward(self, x, edge_index, h=None, c=None, output=False):
        x = F.dropout(x, self.dropout, training=self.training)
        h, c = self.recurrent(x, edge_index, h, c)
        h = F.relu(h)
        if output == True:
            h = F.dropout(h, self.dropout, training=self.training)
        return h, c

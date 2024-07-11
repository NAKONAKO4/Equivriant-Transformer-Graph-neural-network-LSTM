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

    def forward(self, data):
        pos = data.pos
        f = data.x
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

        q = h_q(f)
        k = self.tp_k(f[edge_src], edge_sh, self.fc_k(edge_length_embedded))
        v = self.tp_v(f[edge_src], edge_sh, self.fc_v(edge_length_embedded))

        exp = edge_weight_cutoff[:, None] * self.dot(q[edge_dst], k).exp()
        z = scatter(exp, edge_dst, dim=0, dim_size=len(f))
        z[z == 0] = 1
        alpha = exp / z[edge_dst]

        return scatter(alpha.relu().sqrt() * v, edge_dst, dim=0, dim_size=len(f))



model=GraphSelfAttention("15x0e",
                         "15x0e",
                         "15x0e",
                         "15x0e",
                         1.3)
import torch
from torch_geometric.data import Data, Dataset, DataLoader
from torch.utils.data import SubsetRandomSampler
from e3nn import o3

def random_graph_dataset(num_graphs, num_nodes_per_graph=20):
    irreps_input = o3.Irreps("15x0e")
    dataset = []
    for i in range(num_graphs):
        pos = torch.randn(num_nodes_per_graph, 3)
        x = irreps_input.randn(num_nodes_per_graph, -1)
        data = Data(pos=pos, x=x)
        dataset.append(data)
    return dataset

# Parameters
num_graphs = 100
num_nodes_per_graph = 20
batch_size = 10
test_split = 0.2  # 20% of the dataset will be used for testing

# Create dataset
dataset = random_graph_dataset(num_graphs, num_nodes_per_graph)

# Split dataset into train and test sets using SubsetRandomSampler
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(test_split * dataset_size)
train_indices, test_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

# Create DataLoader for train and test sets
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm  # For progress bar

optimizer = Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()  # Example loss function, adjust as needed

# Training loop
num_epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    epoch_loss = 0.0

    # Iterate over batches of train_loader
    for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        output = model(batch)

        # Compute loss
        loss = criterion(output, batch.x)  # Adjust y to match your dataset's target

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Calculate average epoch loss
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f'Train Loss after epoch {epoch + 1}: {avg_epoch_loss:.4f}')

    # Validation (optional)
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Validation', leave=False):
            batch = batch.to(device)
            output = model(batch)
            loss = criterion(output, batch.x)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(test_loader)
    print(f'Validation Loss after epoch {epoch + 1}: {avg_val_loss:.4f}')

print('Training finished!')
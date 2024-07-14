from torch_geometric.data import Data, Dataset, DataLoader
from torch.utils.data import SubsetRandomSampler
from e3nn import o3
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
from ETG_version1 import GraphSelfAttention
from ETGplusLSTM_version1 import ETGplusLSTM
from ETGplusLSTM_version1 import ETG_LSTM_final

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def random_graph_dataset(num_graphs, num_nodes_per_graph=20):
    irreps_input = o3.Irreps("15x0e")
    dataset = []
    for i in range(num_graphs):
        pos = torch.randn(num_nodes_per_graph, 3)
        x = irreps_input.randn(num_nodes_per_graph, -1)
        pos = pos.to(device)
        x = x.to(device)
        data = Data(pos=pos, x=x)
        dataset.append(data)
    return dataset

# Parameters
num_graphs = 100
num_nodes_per_graph = 20
batch_size = 10
test_split = 0.2  # 20% of the dataset will be used for testing

model=GraphSelfAttention("15x0e",
                         "15x0e",
                         "15x0e",
                         "15x0e",
                         1.3)
model1 = ETGplusLSTM(o3.Irreps("15x0e"),
                     o3.Irreps("15x0e"),
                     o3.Irreps("15x0e"),
                     o3.Irreps("128x0e"),
                     1.3)
model2=ETG_LSTM_final().to(device)
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

from torch.optim import Adam
from tqdm import tqdm  # For progress bar

optimizer = Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()  # Example loss function, adjust as needed

# Training loop
num_epochs = 50
model.to(device)
model1.to(device)
print(dataset)
'''
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
'''
pos=dataset[0].pos
x=dataset[0].x
h=None
c=None
h,c = model2(x, pos, h, c, output=True)
print(h)
print(c)



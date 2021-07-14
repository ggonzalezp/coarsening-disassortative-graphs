import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import (GraphConv, SAGPooling, global_mean_pool,
                                JumpingKnowledge)
from torch_geometric.datasets import TUDataset
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool, dense_mincut_pool
from collections import Counter
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
import os.path as osp
from math import ceil
import torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score
from glob import glob





pooling = 'sagpool' #(options: sagpool, topk)
dataset_name = 'ENZYMES'



#Dataset
dataset = TUDataset(root='../datasets', name=dataset_name, use_node_attr=True)




#Splitting
dataset = dataset.shuffle()
n = (len(dataset) + 9) // 10
test_dataset = dataset[:n]
val_dataset = dataset[n:2 * n]
train_dataset = dataset[2 * n:]
test_loader = DataLoader(test_dataset, batch_size=20)
val_loader = DataLoader(val_dataset, batch_size=20)
train_loader = DataLoader(train_dataset, batch_size=20)






########
#Model
########

class Net(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, ratio=0.8):
        super(Net, self).__init__()
        self.conv1 = GraphConv(dataset.num_features, hidden, aggr='mean')
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.convs.extend([
            GraphConv(hidden, hidden, aggr='mean')
            for i in range(num_layers - 1)
        ])
        self.pools.extend(
            [SAGPooling(hidden, ratio) for i in range((num_layers) // 2)])
        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)
    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        xs = [global_mean_pool(x, batch)]
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            xs += [global_mean_pool(x, batch)]
            if i % 2 == 0 and i < len(self.convs) - 1:
                pool = self.pools[i // 2]
                x, edge_index, _, batch, _, _ = pool(x, edge_index,
                                                     batch=batch)
        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)
    def __repr__(self):
        return self.__class__.__name__






###########
#Training
###########
#Optimizer, model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(dataset, 3, 64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)




#Training function - supervised graph classification
def train(epoch):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y.view(-1))
        loss.backward()
        loss_all += data.y.size(0) * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).max(dim=-1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


best_val_acc = test_acc = 0
for epoch in range(1, 151):
    train_loss = train(epoch)
    val_acc = test(val_loader)
    if val_acc > best_val_acc:
        test_acc = test(test_loader)
        best_val_acc = val_acc
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
          f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')







import os
os.system('rm -r ../datasets/'+ dataset_name + '/processed')










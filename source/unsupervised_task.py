##Models
from torch_geometric.datasets import TUDataset
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool, dense_mincut_pool
from collections import Counter
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
import os.path as osp
from math import ceil
import torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score
from glob import glob


pooling = 'diffpool' #(options: diffpool, mincut)
dataset_name = 'ENZYMES'






max_nodes_selector = {}
paths = glob('../datasets/*')
for path in paths:
	dataset = path.split('/')[-1]
	gi = pd.read_csv(osp.join(path, dataset + '_graph_indicator.txt'), header = None)
	gc = Counter(gi[0])
	max_nodes_selector[dataset] = max(gc.values())



#Pooling selector
pooling_selector = {
	'diffpool': dense_diff_pool,
	'mincut': dense_mincut_pool
}








#Dataset
max_nodes = max_nodes_selector[dataset_name]

class MyFilter(object):
	def __call__(self, data):
		# data.node_attr = data.x[:, :dataset.num_node_attributes]
		# data.node_label = data.x[:, dataset.num_node_attributes:]
		data.num_nodes <= max_nodes
		return data

class MyTransform(object):
	def __call__(self, data):
		data.node_attr = data.x[:, :dataset.num_node_attributes]
		data.node_label = data.x[:, dataset.num_node_attributes:]
		return data

dataset = TUDataset(root='../datasets', name=dataset_name, transform=T.Compose([T.ToDense(max_nodes), MyTransform()]),
					pre_filter=MyFilter(), use_node_attr=True)




#Splitting
dataset = dataset.shuffle()
n = (len(dataset) + 9) // 10
test_dataset = dataset[:n]
val_dataset = dataset[n:2 * n]
train_dataset = dataset[2 * n:]
test_loader = DenseDataLoader(test_dataset, batch_size=20)
val_loader = DenseDataLoader(val_dataset, batch_size=20)
train_loader = DenseDataLoader(train_dataset, batch_size=20)






########
#Model
########

#GNN to compute transformed node features for pooling (for assignation matrix)
class GNN(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels,
				 normalize=False, lin=True):
		super(GNN, self).__init__()
		self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
		self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
		self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
		self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
		self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
		self.bn3 = torch.nn.BatchNorm1d(out_channels)
		if lin is True:
			self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
									   out_channels)
		else:
			self.lin = None
	def bn(self, i, x):
		batch_size, num_nodes, num_channels = x.size()
		x = x.view(-1, num_channels)
		x = getattr(self, 'bn{}'.format(i))(x)
		x = x.view(batch_size, num_nodes, num_channels)
		return x
	def forward(self, x, adj, mask=None):
		batch_size, num_nodes, in_channels = x.size()
		x0 = x
		x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask)))
		x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask)))
		x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask)))
		x = torch.cat([x1, x2, x3], dim=-1)
		if self.lin is not None:
			x = F.relu(self.lin(x))
		return x




#Net to compute pooling in an unsupervised manner
class Net(torch.nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		num_nodes = 3 #3 clusters
		self.gnn1_pool = GNN(dataset.num_node_attributes, 64, num_nodes)
		self.gnn1_embed = GNN(dataset.num_node_attributes, 64, 64, lin=False)
		self.pooling = pooling_selector[pooling]
	def forward(self, x, adj, mask=None):
		s = self.gnn1_pool(x, adj, mask)
		x = self.gnn1_embed(x, adj, mask)
		x, adj, l1, e1 = self.pooling(x, adj, s, mask)
		return torch.softmax(s, dim=-1), l1, e1	#returns assignation matrix, and auxiliary losses














###########
#Training
###########
#Optimizer, model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)




#Trains using auxiliary losses only
def train(epoch):
	model.train()
	loss_all = 0
	for data in train_loader:
		data = data.to(device)
		optimizer.zero_grad()
		s, l1, e1 = model(data.node_attr, data.adj, data.mask)
		loss = l1 + e1
		loss.backward()
		loss_all += data.y.size(0) * loss.item()
		optimizer.step()
	return loss_all / len(train_dataset)


#Measures mutual information of clustering computed by pooling and ground truth
@torch.no_grad()
def test(loader):
	model.eval()
	correct = 0
	for data in loader:
		data = data.to(device)
		pred_node_label = model(data.node_attr, data.adj, data.mask)[0].max(dim=-1)[1].detach().cpu().numpy()
		truth_node_labels = data.node_label.max(dim=-1)[1].detach().cpu().numpy()
		nmi = normalized_mutual_info_score(truth_node_labels.flatten(), pred_node_label.flatten())
	return nmi



best_val_acc = test_acc = 0
for epoch in range(1, 151):
	train_loss = train(epoch)
	val_acc = test(val_loader)
	if val_acc > best_val_acc:
		test_acc = test(test_loader)
		best_val_acc = val_acc
	print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.8f}, '
		  f'Val NMI: {val_acc:.4f}, Test NMI: {test_acc:.4f}')




import os
os.system('rm -r ../datasets/'+ dataset_name + '/processed')




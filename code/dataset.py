import numpy as np
import networkx as nx

import torch
import torch_geometric as tg
import torch_geometric.transforms as T
from torch_geometric.datasets import WikiCS, WebKB, Actor, WikipediaNetwork, DeezerEurope, Twitch


def read_graphs(data_name, dir):
    # set node id start from 0
    raw_edges = np.loadtxt('{}/{}_A.txt'.format(dir, data_name), delimiter=',', dtype=int) - 1
    G = nx.from_edgelist(raw_edges)
    return G


def read_graph_indicator(data_name, dir):
    # set node id start from 0
    indicator = np.loadtxt('{}/{}_graph_indicator.txt'.format(dir, data_name), delimiter=',', dtype=int) - 1
    return indicator


def read_graph_label(data_name, dir):
    graph_labels = np.loadtxt('{}/{}_graph_labels.txt'.format(dir, data_name), dtype=int)
    graph_labels[graph_labels == -1] = 0
    return graph_labels


def read_node_attribute(data_name, dir):
    node_attributes = np.loadtxt('{}/{}_node_attributes.txt'.format(dir, data_name), delimiter=',', dtype=np.float32)
    return node_attributes


def read_node_label(data_name, dir):
    node_labels = np.loadtxt('{}/{}_node_labels.txt'.format(dir, data_name), dtype=int)
    return node_labels


def load_TUDDataset(data_name):
    if data_name in ['ENZYMES', 'PROTEINS', 'PROTEINS_full']:
        available_node_attributes = True
    elif data_name in ['DD', 'KKI', 'OHSU', 'Peking_1']:
        available_node_attributes = False
    root = '../datasets/{}'.format(data_name)

    G = read_graphs(data_name=data_name, dir=root)
    indicator = read_graph_indicator(data_name=data_name, dir=root)
    graph_labels = read_graph_label(data_name=data_name, dir=root)
    node_labels = read_node_label(data_name=data_name, dir=root)
    if available_node_attributes:
        node_attributes = read_node_attribute(data_name=data_name, dir=root)
    else:
        node_attributes = np.zeros((G.number_of_nodes(), 1), dtype=np.float32)

    assert indicator.shape[0] == node_attributes.shape[0] == node_labels.shape[0]
    n_graphs = np.unique(indicator).shape[0]

    data_list = []
    for graph_idx in range(n_graphs):
        node_idx = np.where(indicator == graph_idx)[0]
        node_id_mapping = dict(zip(node_idx, range(node_idx.shape[0])))
        g = G.subgraph(node_idx)
        g = nx.relabel_nodes(g, node_id_mapping)
        x = node_attributes[node_idx]
        node_y = node_labels[node_idx]
        g_y = graph_labels[graph_idx]
        data = tg.utils.from_networkx(G=g)
        data.x = torch.FloatTensor(x)
        data.node_y = torch.LongTensor(node_y)
        data.g_y = torch.LongTensor(g_y)
        data_list.append(data)
    return data_list


def load_pyg_dataset(
        data_name: str, quiet: bool = False, device='cpu'
):
    path = '../datasets/PyG_data/' + data_name + '/'
    if data_name in ['WikiCS']:
        data = WikiCS(path).data
    elif data_name in ["Cornell", "Texas", "Wisconsin"]:
        data = WebKB(path, data_name).data
        data.y = data.y.long()
    elif data_name in ["Chameleon", "Squirrel", "Crocodile"]:
        data = WikipediaNetwork(path, data_name.lower()).data
        data.y = data.y.long()
    elif data_name in ['Actor']:
        data = Actor(path, transform=T.NormalizeFeatures()).data
    elif data_name in ['DeezerEurope']:
        data = DeezerEurope(path, transform=T.NormalizeFeatures()).data
    elif 'Twitch' in data_name:
        path = '../datasets/PyG_data/' + data_name.split('-')[0] + '/'
        data = Twitch(path, name=data_name.split('-')[1], transform=T.NormalizeFeatures()).data
    else:
        data = None
        print('Wrong data name!')
        pass

    data.num_classes = data.y.unique().shape[0]
    data = data.to(device)
    if not quiet:
        print('The obtained data {} has {} nodes, {} edges, {} features, {} labels, '.
              format(data_name, data.num_nodes, data.num_edges, data.num_features, data.num_classes))
    return data

# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
from collections import defaultdict
import networkx as nx
import torch
from torch_geometric.loader import DataLoader
from torch_geometric import utils
from models.sat_finetune import GraphTransformer,GraphDataset
from torch_geometric.data import Data, Dataset
import csv
from rdkit import Chem
from descriptor import atom_features, bond_features, etype_features
import os
from torch_scatter import scatter_add
import torch_geometric.utils as utils
import numpy as np
import yaml


def load_args():
    parser = argparse.ArgumentParser(
        description='Model visualization: SAT',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--visu', action='store_true', help='perform visualization')
    parser.add_argument('--graph-idx', type=int, default=-1,
                        help='graph to interpret')
    parser.add_argument('--outpath', type=str,
                        default='pictures',
                        help='visualization output path')
    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    return args


def load_model(datapath):
    state_dict = torch.load(datapath)
    config = yaml.load(open("config_finetune.yaml", "r"), Loader=yaml.FullLoader)
    model = GraphTransformer(**config["sat"])
    model.load_state_dict(state_dict)
    return model


def read_smiles(data_path, target):
    smiles_data, labels = [], []
    with open(data_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')

        for i, row in enumerate(csv_reader, start=1):
            smiles = row['smiles']
            label = row[target]
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None and label != '':
                smiles_data.append(smiles)
                labels.append(float(label))

    print(len(smiles_data))
    return smiles_data, labels


class MolTestDataset(Dataset):
    def __init__(self, data_path, target):
        super().__init__()
        self.smiles_data, self.labels = read_smiles(data_path, target)

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        nodes = []
        edges = []
        edge_attrs = []

        for atom in mol.GetAtoms():
            node_feat = atom_features(atom)
            nodes.append(node_feat)

        for bond in mol.GetBonds():
            bond_feat = bond_features(bond)
            etype_feat = etype_features(bond)
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            edge_attrs.append(etype_feat + bond_feat)

        x = torch.tensor(np.vstack(nodes), dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.vstack(edge_attrs), dtype=torch.float)
        y = torch.tensor(self.labels[index], dtype=torch.float).view(1, -1)

        smiles = self.smiles_data[index]  # append smiles
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, smiles=smiles)  # smiles to data
        return data

    def __len__(self):
        return len(self.smiles_data)


class PositionEncoding(object):
    def apply_to(self, dataset):
        dataset.abs_pe_list = []
        for i, g in enumerate(dataset):
            pe = self.compute_pe(g)
            dataset.abs_pe_list.append(pe)
        return dataset


class LapEncoding(PositionEncoding):
    def __init__(self, dim, use_edge_attr=False, normalization=None):
        """
        normalization: for Laplacian None. sym or rw
        """
        self.pos_enc_dim = dim
        self.normalization = normalization
        self.use_edge_attr = use_edge_attr

    def compute_pe(self, graph):
        edge_attr = graph.edge_attr if self.use_edge_attr else None
        edge_index, edge_attr = utils.get_laplacian(
            graph.edge_index, edge_attr, normalization=self.normalization,
            num_nodes=graph.num_nodes)
        L = utils.to_scipy_sparse_matrix(edge_index, edge_attr).tocsc()
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort() # increasing order
        EigVal, EigVec = np.real(EigVal[idx]), np.real(EigVec[:,idx])
        return torch.from_numpy(EigVec[:, 1:self.pos_enc_dim+1]).float()


class RWEncoding(PositionEncoding):
    def __init__(self, dim, use_edge_attr=False, normalization=None):
        """
        normalization: for Laplacian None. sym or rw
        """
        self.pos_enc_dim = dim
        self.use_edge_attr = use_edge_attr

    def compute_pe(self, graph):
        W0 = normalize_adj(graph.edge_index, num_nodes=graph.num_nodes).tocsc()
        W = W0
        vector = torch.zeros((graph.num_nodes, self.pos_enc_dim))
        vector[:, 0] = torch.from_numpy(W0.diagonal())
        for i in range(self.pos_enc_dim - 1):
            W = W.dot(W0)
            vector[:, i + 1] = torch.from_numpy(W.diagonal())
        return vector.float()


def normalize_adj(edge_index, edge_weight=None, num_nodes=None):
    edge_index, edge_weight = utils.remove_self_loops(edge_index, edge_weight)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1),
                                 device=edge_index.device)
    num_nodes = utils.num_nodes.maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = 1.0 / deg
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    edge_weight = deg_inv[row] * edge_weight
    return utils.to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes=num_nodes)


POSENCODINGS = {
    'lap': LapEncoding,
    'rw': RWEncoding,
}


def compute_attn(datapath, dataset, graph_idx):
    model = load_model(datapath)
    model.eval()

    graph = dataset[graph_idx]
    y_true = graph.y.squeeze().item()


    graph_dset = GraphDataset([dataset[graph_idx]], degree=True, k_hop=3, se="khopgnn", use_subgraph_edge_attr=True)

    graph_loader = DataLoader(graph_dset, batch_size=1, shuffle=False)

    abs_pe_method = POSENCODINGS['rw']
    abs_pe_encoder = abs_pe_method(20, normalization='sym')
    if abs_pe_encoder is not None:
        abs_pe_encoder.apply_to(graph_dset)

    attns = []

    def get_attns(module, input, output):
        attns.append(output[1])

    for i in range(1):
        model.encoder.layers[i].self_attn.register_forward_hook(get_attns)

    for g in graph_loader:
        with torch.no_grad():
            y_pred = model(g, return_attn=True)
            y_pred = y_pred.argmax(dim=-1)
            y_pred = y_pred.item()
    print('Ground truth: ', y_true, 'Prediction: ', y_pred)

    attn = attns[-1].mean(dim=-1)[-1]
    return attn


def draw_graph_with_attn(
        graph,
        outdir,
        filename,
        nodecolor=["tag", "attn"],
        dpi=300,
        edge_vmax=None,
        args=None,
        eps=1e-6,):

    if len(graph.edges) == 0:
        return
    import matplotlib.pyplot as plt
    plt.switch_backend("agg")
    fig = plt.figure(figsize=(4 * len(nodecolor), 4), dpi=dpi)

    node_colors = defaultdict(list)

    titles = {
        'tag': ' ',
        'attn': ' '}

    for i in graph.nodes():
        for key in nodecolor:
            node_colors[key].append(graph.nodes[i][key])

    vmax = {}
    cmap = {}
    for key in nodecolor:
        vmax[key] = 19
        cmap[key] = 'tab20'
        if 'attn' in key:
            # vmax[key] = max(node_colors[key])
            vmax[key] = 0.4
            cmap[key] = 'Reds'

    pos_layout = nx.kamada_kawai_layout(graph, weight=None)

    for i, key in enumerate(nodecolor):
        ax = fig.add_subplot(1, len(nodecolor), i + 1)
        ax.set_title(titles[key], fontweight='bold')
        nx.draw(
            graph,
            pos=pos_layout,
            with_labels=False,
            font_size=4,
            node_color=node_colors[key],
            # vmin=0,
            # vmax=vmax[key],
            cmap=cmap[key],
            width=1.3,
            node_size=100,
            alpha=1.0,
        )
        if 'attn' in key:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            sm = plt.cm.ScalarMappable(cmap=cmap[key], norm=plt.Normalize(vmin=0, vmax=vmax[key]))
            sm._A = []
            plt.colorbar(sm, cax=cax)

    fig.axes[0].xaxis.set_visible(False)
    fig.canvas.draw()

    save_path = os.path.join(outdir, filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)


def main():
    all_args = load_args()
    dataset = MolTestDataset("data/test/Top_5.csv", 'pKa')
    dataset = GraphDataset(dataset, degree=True, k_hop=1, se="khopgnn", use_subgraph_edge_attr=True)

    if all_args.graph_idx < 0:
        sizes = [graph.num_nodes for graph in dataset]
        indices = np.argsort(sizes)[::-1]
        indices = [i for i in indices if sizes[i] <= 100][:5]
    else:
        indices = [all_args.graph_idx]

    for graph_idx in indices:
        print("Graph ", graph_idx)
        graph = dataset[graph_idx]

        print("Computing attention for SAT")
        attn = compute_attn('data/model_weight/SAT/Finetune_SAT.pth', dataset, graph_idx=graph_idx)

        graph.tag = graph.x.argmax(dim=-1)
        graph.attn = attn

        print(attn)
        print(graph)

        graph = utils.to_networkx(graph, node_attrs=['tag', 'attn'], to_undirected=True)
        draw_graph_with_attn(
            graph,all_args.outpath,
            'graph{}.png'.format(graph_idx),
            nodecolor=['tag', 'attn'])


if __name__ == "__main__":
    main()


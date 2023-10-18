from torch import nn
from torch_geometric import utils
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool\



atom_dim = 40
bond_dim = 10



class GINEConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINEConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2*emb_dim), nn.ReLU(), nn.Linear(2*emb_dim, emb_dim))
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.root_emb = nn.Embedding(1, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        row, col = edge_index
        deg = utils.degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return self.mlp(aggr_out)



class GINet(nn.Module):
    def __init__(self, num_layers=4, emb_dim=512,drop_ratio=0.3, pool='add', pred_n_layer=2,pred_act='relu'):
        super(GINet, self).__init__()

        # self.mlp = nn.Sequential(
        #     nn.Linear(emb_dim, 2*emb_dim), nn.ReLU(), nn.Linear(2*emb_dim, emb_dim))

        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio

        self.x_embedding = nn.Linear(40, emb_dim)
        self.edge_embedding = nn.Linear(10, emb_dim)

        nn.init.kaiming_normal_(self.x_embedding.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.edge_embedding.weight, mode='fan_in', nonlinearity='relu')

        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layers):
            self.gnns.append(GINEConv(emb_dim))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool

        self.pred_n_layer = max(1, pred_n_layer)

        if pred_act == 'relu':
            pred_head = [
                nn.Linear(self.emb_dim, self.emb_dim//2),
                nn.ReLU(inplace=True)
            ]
            for _ in range(self.pred_n_layer - 1):
                pred_head.extend([
                    nn.Linear(self.emb_dim//2, self.emb_dim//2),
                    nn.ReLU(inplace=True),
                ])
            pred_head.append(nn.Linear(self.emb_dim//2, 1))

        elif pred_act == 'softplus':
            pred_head = [
                nn.Linear(self.emb_dim, self.emb_dim//2),
                nn.Softplus()
            ]
            for _ in range(self.pred_n_layer - 1):
                pred_head.extend([
                    nn.Linear(self.emb_dim//2, self.emb_dim//2),
                    nn.Softplus()
                ])
            pred_head.append(nn.Linear(self.emb_dim//2, 1))

        elif pred_act == 'leakyrelu':  # 添加leakyrelu选项
            pred_head = [
                nn.Linear(self.emb_dim, self.emb_dim // 2),
                nn.LeakyReLU(inplace=True)
            ]
            for _ in range(self.pred_n_layer - 1):
                pred_head.extend([
                    nn.Linear(self.emb_dim // 2, self.emb_dim // 2),
                    nn.LeakyReLU(inplace=True),
                ])
            pred_head.append(nn.Linear(self.emb_dim // 2, 1))

        else:
            raise ValueError('Undefined activation function')

        self.pred_head = nn.Sequential(*pred_head)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        h = self.x_embedding(x.float())
        edge_attr = self.edge_embedding(edge_attr.float())

        for layer in range(self.num_layers):
            h_in = h  # 当前层输入
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = h_in + h  # 加入残差连接
            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        h = self.pool(h, data.batch)

        return self.pred_head(h)


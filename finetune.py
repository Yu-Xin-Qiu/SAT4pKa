
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
import yaml
import numpy as np
import torch
from models.ginet_finetune import GINet
from models.gcn_finetune import GCN
import shutil
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import time
import math
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import random
from torch.utils.data import Subset
from descriptor import MolTestDataset
from models.sat_finetune import GraphDataset,GraphTransformer


def load_pre_trained_weights(model):
    try:
        state_dict = torch.load(config['fine_tune_from'], map_location=device)
        model.load_state_dict(state_dict)
        print("Loaded pre-trained model with success.")
    except FileNotFoundError:
        print("Pre-trained weights not found. Training from scratch.")
    return model


def train_and_test(model, loader, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()
    loss_all = 0
    mae_all = 0
    r2_all = 0
    mse_all = 0

    with torch.set_grad_enabled(is_train):
        for data in loader:

            optimizer.zero_grad()
            data = data.to(device)

            data.y = normalizer.norm(data.y)

            output = model(data)

            data.y = normalizer.denorm(data.y)
            output = normalizer.denorm(output)

            loss = F.mse_loss(output, data.y) * data.num_graphs

            if is_train:
                loss.backward()
                optimizer.step()
            loss_all += loss.item()
            mae_all += F.l1_loss(output, data.y).item() * data.num_graphs
            r2_all += r2_score(data.y.detach().cpu().numpy(), output.detach().cpu().numpy()) * data.num_graphs
            mse_all += loss.item()

    loss = loss_all / len(loader.dataset)
    mae = mae_all / len(loader.dataset)
    r2 = r2_all / len(loader.dataset)
    rmse = math.sqrt(mse_all / len(loader.dataset))

    return loss, mae, r2, rmse


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_finetune.yaml', os.path.join(model_checkpoints_folder, 'config_finetune.yaml'))


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def evaluate_dataset(dataset):
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    y, pred = [], []
    with torch.no_grad():
        for data in loader:
            data= data.to(device)

            data.y = normalizer.norm(data.y)

            output = model(data)

            data.y = normalizer.denorm(data.y)
            output = normalizer.denorm(output)

            y.extend(data.y.detach().cpu().numpy())
            pred.extend(output.detach().cpu().numpy())


    y, pred = np.array(y).flatten(), np.array(pred).flatten()
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    r2 = r2_score(y, pred)
    mse = mean_squared_error(y, pred) 
    return mae, rmse, r2, mse


if __name__ == '__main__':

    config = yaml.load(open("config_finetune.yaml", "r"), Loader=yaml.FullLoader)
    print(config)

    if config['model_type'] == 'gin':
        model = GINet(**config["gnn"]).to(device)
        model = load_pre_trained_weights(model)

    elif config['model_type'] == 'gcn':
        model = GCN(**config["gnn"]).to(device)
        model = load_pre_trained_weights(model)

    elif config['model_type'] == 'sat':
        model = GraphTransformer(**config["sat"]).to(device)
        model = load_pre_trained_weights(model)


    layer_list = []
    for name, param in model.named_parameters():
        if 'pred_head' in name:
            # print(name, param.requires_grad)
            layer_list.append(name)

    params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
    base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))

    optimizer = torch.optim.Adam(
        [{'params': base_params, 'lr': config['init_base_lr']}, {'params': params}],config['init_lr'], weight_decay=config['weight_decay'])

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=config['lr_dacay_patience'], factor=config['lr_decay_factor'], verbose=True, min_lr=config['min_lr'])



    print('---loading dataset---')

    dataset = MolTestDataset("data/finetuning/finetune_7642.csv", 'pKa')


    A_dataset = MolTestDataset('data/test/AvLiLuMoVe_123.csv','pKa')
    N_dataset = MolTestDataset('data/test/novartis_280.csv','pKa')
    S7_dataset = MolTestDataset('data/test/SAMPL7.csv','pKa')

    if config['model_type'] == 'sat':
        dataset = GraphDataset(dataset,degree=True, k_hop=3,se = "khopgnn",use_subgraph_edge_attr=True)
        A_dataset=GraphDataset(A_dataset,degree=True, k_hop=3,se = "khopgnn",use_subgraph_edge_attr=True)
        N_dataset=GraphDataset(N_dataset,degree=True, k_hop=3,se = "khopgnn",use_subgraph_edge_attr=True)
        S7_dataset = GraphDataset(S7_dataset, degree=True, k_hop=3, se="khopgnn", use_subgraph_edge_attr=True)


    random.seed(config['seed'])

    dataset_size = len(dataset)
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)


    indices = list(range(dataset_size))
    random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    
    labels = []
    for d in loader:
        labels.append(d.y)
    labels = torch.cat(labels)
    normalizer = Normalizer(labels)
    print(normalizer.mean, normalizer.std, labels.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on:',device)
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)


    dir_name = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('experiments/finetune', dir_name)
    writer = SummaryWriter(log_dir=log_dir)
    best_mae = float('inf')
    model_checkpoints_folder = os.path.join(writer.log_dir, 'finetune')

    # save config file
    _save_config_file(model_checkpoints_folder)



    for epoch in range(1, config['epochs'] + 1):

        start_time = time.time()
        # loss,mae,r2,rmse

        train_loss, train_mae, _ ,_= train_and_test(model, train_loader, is_train=True)
        valid_loss, valid_mae, valid_r2,valid_rmse = train_and_test(model, valid_loader, is_train=False)



        end_time = time.time()  
        epoch_time = end_time - start_time  
        print(f"Epoch {epoch:3d}, Time: {epoch_time:.2f}s, TrainLoss: {train_loss:.6f}, Valid_loss: {valid_loss:.6f},Valid_mae: {valid_mae:.6f},Valid_RMSE: {valid_rmse:.6f},Valid_R2: {valid_r2:.6f}")

        if valid_mae < best_mae:
            best_mae = valid_mae
            # save model
            torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model_{}.pth'.format(epoch)))

            early_stopping_count = 0
        else:
            early_stopping_count += 1

        if early_stopping_count >= config["early_stop_patience"]:
            print(f"/nEarly stopping at epoch {epoch + 1}")
            break

        lr_scheduler.step(valid_mae)

    # save model
    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model_{}.pth'.format(epoch)))
    model.to(device)
    model.eval()
    test_mae, test_rmse , test_r2,test_loss= evaluate_dataset(test_dataset)
    A_MAE, _, _ ,_= evaluate_dataset(A_dataset)
    N_MAE, _, _ ,_= evaluate_dataset(N_dataset)
    S7_MAE, _, _,_= evaluate_dataset(S7_dataset)
    print(f"Test_MAE: {test_mae:.6f},Test_RMSE: {test_rmse:.6f},Test_R2: {test_r2:.6f},A_MAE: {A_MAE:.6f},N_MAE: {N_MAE:.6f},S7_MAE: {S7_MAE:.6f}")



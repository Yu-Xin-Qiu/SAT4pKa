import yaml
import torch
from models.gcn_finetune import GCN
from models.ginet_finetune import GINet
from descriptor import graph_mol2vec
from rdkit import Chem
from models.sat_finetune import GraphTransformer,GraphDataset
from descriptor import MolTestDataset
import numpy as np
from torch_geometric.loader import DataLoader
import csv,os
import contextlib

def predict_pKa(smiles,model_name):

    config_path = "config_finetune.yaml"
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    means= 6.7641
    stds=2.5215

    if model_name == 'GCN':
        model_path = 'data\model_weight\GCN\Finetune_GCN.pth'
        best_model = GCN(**config["gnn"])

        best_model.load_state_dict(torch.load(model_path))
        best_model.eval()

        mol = Chem.MolFromSmiles(smiles)
        data = graph_mol2vec(mol, y=0)
        data.y = (data.y - means) / stds
        output = best_model(data)
        output = output * stds + means

    if model_name == 'GIN':
        model_path = 'data\model_weight\GIN\Finetune_GIN.pth'
        best_model = GINet(**config["gnn"])

        best_model.load_state_dict(torch.load(model_path))
        best_model.eval()

        mol = Chem.MolFromSmiles(smiles)
        data = graph_mol2vec(mol, y=0)
        data.y = (data.y - means) / stds
        output = best_model(data)
        output = output * stds + means

    if model_name == 'SAT':
        model_path = 'data\model_weight\SAT\Finetune_SAT.pth'
        best_model = GraphTransformer(**config["sat"])
        data_file='waste_smiles.csv'
        columns = ['smiles', 'pKa']
        data = [smiles, 0]
        with open(data_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(columns)
            writer.writerow(data)
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            test_dataset = GraphDataset(MolTestDataset(data_file, 'pKa'), degree=True, k_hop=3, se="khopgnn", use_subgraph_edge_attr=True)
        best_model.load_state_dict(torch.load(model_path))
        best_model.eval()
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        test_pred = []
        with torch.no_grad():
            for data in test_loader:
                data = data
                data.y=(data.y-means)/stds
                output = best_model(data)
                output = output*stds+means
                test_pred.extend(output.detach().cpu().numpy())
        output = np.array(test_pred).flatten()
        if os.path.exists(data_file):
            os.remove(data_file)

    return output.item()  

def SMILE2pKa(smiles):
    pred_pKa1 = predict_pKa(smiles, 'GCN')
    pred_pKa2 = predict_pKa(smiles, 'GIN')
    pred_pKa3 = predict_pKa(smiles, 'SAT')
    pred_pKa = (pred_pKa1 + pred_pKa2 + pred_pKa3) / 3
    return pred_pKa
    
# # example
# model_name='SAT'  #SAT,GIN,GCN

smiles = 'C(C(=O)O)N'
pred_pKa= SMILE2pKa(smiles)
print(pred_pKa)





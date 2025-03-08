import time
import torch
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import torch.nn as nn
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Data
from sklearn.model_selection import KFold
from torch_geometric.data import DataLoader as GeometricDataLoader
from torch_geometric.nn import SAGEConv, global_max_pool as gmp, global_mean_pool as gep
from sklearn.metrics import roc_auc_score, precision_score, recall_score,precision_recall_curve, auc,accuracy_score,f1_score
warnings.filterwarnings("ignore")
# mol atom feature for mol graph


def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


# one ont encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


# mol smile to mol graph edge index
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
    return c_size, features, edge_index


def smile_embedding(smile_file_name):
    smile_pd = pd.read_csv(smile_file_name)
    smile_dict = pd.Series(smile_pd.iloc[:, 1].values, index=smile_pd.iloc[:, 0].values).to_dict()
    smile_graph = {}
    smile_key = []

    for key in smile_dict.keys():
        sml_init = Chem.MolToSmiles(Chem.MolFromSmiles(smile_dict[key]), isomericSmiles=True)
        grh = smile_to_graph(sml_init)
        smile_graph[key] = grh
        smile_key.append(key)
    return smile_key, smile_graph



class DTADataset(Dataset):
    def __init__(self, smile_keys=None, y=None, smile_graphs=None):
        self.smile_keys = smile_keys
        self.y = y
        self.smile_graphs = smile_graphs
        self.data_mol = self.process(smile_keys, smile_graphs, y)
        
    def process(self, smile_keys, smile_graphs, y):
        data_list_mol = []
        data_len = len(smile_keys)
        for i in range(data_len):
            smile_key = smile_keys[i]
            labels = y[i]
            c_size, features, edge_index = smile_graphs[smile_key]
            GCNData_mol = Data(x=torch.Tensor(features),
                               edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                               y=torch.FloatTensor([labels]),
                               c_size=torch.LongTensor([c_size]))
            data_list_mol.append(GCNData_mol)
        return data_list_mol

    def __len__(self):
        return len(self.data_mol)

    def __getitem__(self, idx):
        return self.data_mol[idx]



class SMILE_Classification(torch.nn.Module):
    def __init__(self, device, n_output=2, num_features_mol=78, hidden_state=128, dropout=None):
        super(SMILE_Classification, self).__init__()
        print('CPI_classification model Loaded..')
        self.device = device
        self.n_output = n_output
        # compounds network
        self.mol_conv1 = SAGEConv(num_features_mol, num_features_mol*2,'mean')
        self.mol_conv2 = SAGEConv(num_features_mol*2,num_features_mol*2,'mean')
        self.mol_conv3 = SAGEConv(num_features_mol*2,num_features_mol*4,'mean')
        self.mol_fc_g1 = nn.Linear(num_features_mol*4, hidden_state)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        # combined layers
        self.fc = nn.Sequential(nn.Linear(hidden_state, 128),nn.LeakyReLU(),nn.Dropout(dropout),
                                nn.Linear(128, self.n_output))
    def forward(self, data_mol):
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch
        x = self.mol_conv1(mol_x, mol_edge_index)
        x = self.relu(x)
        x = self.mol_conv2(x, mol_edge_index)
        x = self.relu(x)
        x = self.mol_conv3(x, mol_edge_index)
        x = self.relu(x)
        x = gmp(x, mol_batch)  # global max pooling
        x = self.mol_fc_g1(x)
        x = self.dropout(x)
        output = self.fc(x)
        return output



def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = []
    loss_fn = torch.nn.CrossEntropyLoss()
    for batch_idx, data in enumerate(train_loader):
        data_mol = data.to(device)
        optimizer.zero_grad()
        output = model(data_mol)
        loss = loss_fn(output, data_mol.y.view(-1, 1).long().squeeze().to(device))
        loss.backward()
        optimizer.step()
    train_loss.append(loss.item())
    train_loss = np.average(train_loss)
    return train_loss


def evaluate(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    loss_fn = torch.nn.CrossEntropyLoss()
    eval_loss = []
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data_mol = data.to(device)
            output = model(data_mol)
            
            loss = loss_fn(output, data_mol.y.view(-1, 1).long().squeeze().to(device))
            output = torch.nn.functional.softmax(output, dim=-1)
            # save predicted results
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
            eval_loss.append(loss.item())
            # sys.stdout.write(str(format(100. *batch_idx / len(loader),'.2f'))+"%\r")
        eval_loss.append(loss.item())
    eval_loss = np.average(eval_loss)
    return total_labels.numpy().flatten(), total_preds.numpy(),eval_loss


########### main train eval ###############
def predicted(pred_file,output_file):
    smile_key, feature_dict = smile_embedding(pred_file)
    pred_batch = 256
    dropout = 0
    hidden_state = 64
    device_name = 'cuda:0'
    device_name = 'cpu'
    labels = [0]*len(smile_key)
    n_fold_score = []

    print('data processing...')
    stime = time.time()
    pred_dataset = DTADataset(smile_keys=smile_key, y=labels, smile_graphs=feature_dict)
    pred_loader = GeometricDataLoader(pred_dataset, batch_size=pred_batch, shuffle=False)

    print(f'done {time.time()-stime}s')
    USE_CUDA = torch.cuda.is_available()
    device = torch.device(device_name if USE_CUDA else 'cpu')
    for fold in range(5):
        print("*"*10,f"fold{fold}","*"*10)
        model_name = f'./dataset/parm_default_fold_{fold+1}.pt'
        model = SMILE_Classification(device,hidden_state=hidden_state,dropout=dropout)
        model.load_state_dict(torch.load(model_name, map_location=device), strict=False)
        model.to(device)
        _, S, _ = evaluate(model, device, pred_loader)
        S = S[:, 1]
        n_fold_score.append(S)
        # P = (S > 0.5).astype(int)
    score_df = pd.DataFrame(n_fold_score).transpose()
    score_df.index = smile_key
    score_df['avg_score'] = score_df.mean(axis=1)
    new_cols = [f'fold{i+1}' for i in range(5)] + ['avg_score']
    score_df.columns = new_cols
    score_df.to_csv(output_file)
   

predicted(pred_file='./dataset/test2.csv',output_file ='./test2_predicted_score.csv')


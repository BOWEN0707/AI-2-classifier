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
def train_eval():
    pos_key, pos_smile = smile_embedding('./dataset/pos.csv')
    neg_key, neg_smile = smile_embedding('./dataset/neg.csv')
    train_batch = 128
    val_batch = 128
    NUM_EPOCHS = 80
    early_stop = 10
    dropout = 0.5
    LR = 0.001
    hidden_state = 64
    device_name = 'cuda:0'
    device_name = 'cpu'
    


    kf = KFold(n_splits=5, shuffle=True)
    pos_smile.update(neg_smile)
    feature_dict = pos_smile
    keys = pos_key + neg_key
    labels = [1]*len(pos_key) + [0]*len(neg_key)
    fold = 0
    n_fold_valid_results = [['Fold','AUC','PRC','ACC','Recall','f1'],]
    for train_index, test_index in kf.split(keys):
        train_key_list = [keys[i] for i in train_index]
        train_Y = [labels[i] for i in train_index]
        valid_key_list = [keys[i] for i in test_index]
        valid_Y = [labels[i] for i in test_index]
        fold = fold+1
        model_name = f'./dataset/parm_default_fold_{fold}.pt'
        print("*"*10,f"fold{fold}","*"*10)
        print('data processing...')
        stime = time.time()
        train_dataset = DTADataset(smile_keys=train_key_list, y=train_Y, smile_graphs=feature_dict)
        train_loader = GeometricDataLoader(train_dataset, batch_size=train_batch, shuffle=True)
        valid_dataset = DTADataset(smile_keys=valid_key_list, y=valid_Y, smile_graphs=feature_dict)
        valid_loader = GeometricDataLoader(valid_dataset, batch_size=val_batch, shuffle=False)
        print(f'done {time.time()-stime}s')
        USE_CUDA = torch.cuda.is_available()
        device = torch.device(device_name if USE_CUDA else 'cpu')
        model = SMILE_Classification(device,hidden_state=hidden_state,dropout=dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        model.to(device)

        best_auc = 0.5
        print('epoch\ttime\tt_loss\tv_loss\tv_auc')
        for epoch in range(NUM_EPOCHS):
            time_start = time.time()
            train_loss = train(model, device, train_loader, optimizer, epoch + 1)
            T, S, val_loss = evaluate(model, device, valid_loader)
            S = S[:, 1]
            P = (S > 0.5).astype(int)
            AUROC = roc_auc_score(T, S)
            
            AUCS = [str(epoch+1),str(format(time.time()-time_start, '.1f')),str(format(train_loss, '.4f')),str(format(val_loss, '.4f')),str(format(AUROC, '.4f'))]
            print('\t'.join(map(str, AUCS)))
            if AUROC >= best_auc:
                stop_epoch = 0
                best_auc = AUROC
                best_epoch = epoch + 1
                torch.save(model.state_dict(), model_name)
            else:
                stop_epoch += 1
            if stop_epoch == early_stop:
                print('(EARLY STOP) No improvement since epoch ', best_epoch, '; best_test_AUC', best_auc)
                break
       
        ## evaluate models
        model = SMILE_Classification(device,hidden_state=hidden_state,dropout=0)
        model.load_state_dict(torch.load(model_name, map_location=device), strict=False)
        model.to(device)
        T, S, val_loss = evaluate(model, device, valid_loader)
        S = S[:, 1]
        P = (S > 0.5).astype(int)
        AUROC = roc_auc_score(T, S)
        tpr, fpr, _ = precision_recall_curve(T, S)
        AUPR = auc(fpr, tpr)
        ACC = accuracy_score(T,P)
        REC = recall_score(T,P)
        f1 = f1_score(T, P)
        result_fold = [fold,AUROC,AUPR,ACC,REC,f1]
        n_fold_valid_results.append(result_fold)
    return n_fold_valid_results
n_fold_valid_results = train_eval()

with open('./5-fold-results.txt','w') as f:
    print(n_fold_valid_results,file=f)

for i in n_fold_valid_results:
    print(i)
import os
import pickle
import random
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader
import torch
from torch.nn import Sequential, Linear, ReLU
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, GINConv, global_mean_pool
import pytorch_lightning as pl
from torch.nn import BatchNorm1d

"""
    Implementation of the Static Graph Isomorphism Network (GIN) for graph-based property prediction.
"""

random.seed(123)

class GIN(torch.nn.Module):

    def __init__(self, dim_features, dim_target, config):
        super(GIN, self).__init__()

        self.config = config
        self.dropout = config['dropout'][0]
        self.embeddings_dim = [config['hidden_units'][0][0]] + config['hidden_units'][0]
        self.no_layers = len(self.embeddings_dim)
        self.first_h = []
        self.nns = []
        self.convs = []
        self.linears = []

        train_eps = config['train_eps'][0]
        if config['aggregation'][0] == 'sum':
            self.pooling = global_add_pool
        elif config['aggregation'][0] == 'mean':
            self.pooling = global_mean_pool

        for layer, out_emb_dim in enumerate(self.embeddings_dim):

            if layer == 0:
                self.first_h = Sequential(Linear(dim_features, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                          Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU())
                self.linears.append(Linear(out_emb_dim, dim_target))
            else:
                input_emb_dim = self.embeddings_dim[layer - 1]
                self.nns.append(Sequential(Linear(input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                           Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU()))
                self.convs.append(GINConv(self.nns[-1], train_eps=train_eps))  # Eq. 4.2

                self.linears.append(Linear(out_emb_dim, dim_target))

        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        self.linears = torch.nn.ModuleList(self.linears)  # has got one more for initial input

    def forward(self, x, edge_index, batch):
        out = 0

        for layer in range(self.no_layers):
            if layer == 0:
                x = self.first_h(x)
                out += F.dropout(self.pooling(self.linears[layer](x), batch), p=self.dropout)
            else:
                # Layer l ("convolution" layer)
                x = self.convs[layer - 1](x, edge_index)
                out += F.dropout(self.linears[layer](self.pooling(x, batch)), p=self.dropout, training=self.training)

        return out

def read_data():
    data = pd.read_csv("final_data" + ".csv", header=0)
    date_data = pd.read_csv("GnnResults/final_data_date.csv", header=0)
    avg_trans_data = pd.read_csv("GnnResults/average_transaction.csv", header=0)
    date_data.columns = ['id', 'network', 'data_duration']
    avg_trans_data.columns = ['id', 'network', 'timeframe', 'avg_daily_trans']
    data.columns = ['id', 'network', 'timeframe', 'start_date', 'node_count',
                    'edge_count', 'density', 'diameter',
                    'avg_shortest_path_length', 'max_degree_centrality',
                    'min_degree_centrality',
                    'max_closeness_centrality', 'min_closeness_centrality',
                    'max_betweenness_centrality',
                    'min_betweenness_centrality',
                    'assortativity', 'clique_number', 'motifs', "peak", "last_dates_trans",
                    "label_factor_percentage",
                    "label"]

    data = data.drop('id', axis=1)
    date_data = date_data.drop('id', axis=1)
    avg_trans_data = avg_trans_data.drop('id', axis=1)
    avg_trans_data = avg_trans_data.drop('timeframe', axis=1)
    data = pd.merge(data, date_data, on="network", how="left")
    data = pd.merge(data, avg_trans_data, on="network", how="left")
    data = data.drop_duplicates(subset=["network"], keep='first')
    data['label'] = data.pop('label')
    data.to_csv('final_data_with_header.csv', header=True)
    return data

def read_torch_data():
    file_path = "PygGraphs/"
    inx = 1
    GraphDataList = []
    files = os.listdir(file_path)
    for file in files:
        if file.endswith(".txt"):
            with open(file_path + file, 'rb') as f:
                print("\n Reading Torch Data {} / {}".format(inx, len(files)))
                data = pickle.load(f)
                GraphDataList.append(data)
                inx += 1
    return GraphDataList

def read_torch_time_series_data(network, variable=None):
    file_path = "PygGraphs/TimeSeries/{}/".format(network)
    file_path_raw_graph = "PygGraphs/TimeSeries/{}/RawGraph/".format(network)
    file_path_TDA = "PygGraphs/TimeSeries/{}/TDA/".format(network)
    file_path_different_TDA = "PygGraphs/TimeSeries/{}/TDA/{}/".format(network, variable)
    file_path_temporal_TDA = "PygGraphs/TimeSeries/{}/TemporalVectorizedGraph/".format(network)

    file_path_different_TDA_Tuned = "PygGraphs/TimeSeries/{}/TDA_Tuned/{}/".format(network, variable)
    file_path_temporal_TDA_Tuned = "PygGraphs/TimeSeries/{}/TemporalVectorizedGraph_Tuned/".format(network)
    inx = 1
    GraphDataList = []
    files = os.listdir(file_path_different_TDA_Tuned)
    for file in files:
        with open(file_path_different_TDA_Tuned + file, 'rb') as f:
            # print("\n Reading Torch Data {} / {}".format(inx, len(files)))
            data = pickle.load(f)
            GraphDataList.append(data)
            inx += 1
    return GraphDataList

def GIN_classifier(data, network):
    """
      Train and evaluate a Graph Isomorphism Network (GIN) classifier on the given data.

      Args:
          data (list): List of dictionaries containing data samples and labels.
          network (str): Name of the network associated with the data.

      Returns:
          None
    """

    count_one_labels = sum(1 for item in data if item['y'] == 1)
    count_zero_labels = sum(1 for item in data if item['y'] == 0)
    with open("config_GIN.yml", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    for duplication in range(0, 1):
        train_size = int(0.8 * len(data))
        test_size = len(data) - train_size
        # chronological order
        train_dataset = data[:train_size]
        test_dataset = data[train_size:]
        # train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
        train_loader = DataLoader(train_dataset)
        test_loader = DataLoader(test_dataset)
        # for temporal static GNN the input dim is 18 (2x7 temporal + 4 ta static)
        model = GIN(dim_features=1, dim_target=2, config=config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(0, 101):
            train(train_loader, model, criterion, optimizer)
            scores_tr = test(train_loader, model)
            train_acc = scores_tr[0]
            train_auc = scores_tr[1]
            scores_te = test(test_loader, model)
            test_acc = scores_te[0]
            test_auc = scores_te[1]
            scores_unseen = test(test_loader, model)
            unseen_acc = scores_unseen[0]
            unseen_auc = scores_unseen[1]
            if epoch % 10 == 0:
                print(
                    f"Network\t{network} Duplicate\t{duplication}\tEpoch\t {epoch}\t Train Accuracy\t {train_acc:.4f}\t Train AUC Score\t {train_auc:.4f}\t Test Accuracy: {test_acc:.4f}\t test AUC Score\t {test_auc:.4f}\t unseen AUC Score\t {unseen_auc:.4f}")

            if epoch % 100 == 0:
                with open('GnnResults/GIN_TimeSeries_Result.txt', 'a+') as file:
                    file.write(
                        f"\nNetwork\t{network}\tDuplicate\t{duplication}\tEpoch\t{epoch}\tTrain Accuracy\t{train_acc:.4f}\tTrain AUC Score\t{train_auc:.4f}\tTest Accuracy:{test_acc:.4f}\tTest AUC Score\t{test_auc:.4f}\tunseen AUC Score\t{unseen_auc:.4f}\tNumber of Zero labels\t{count_zero_labels}\tNumber of one labels\t{count_one_labels}")
                    file.close()

def train(train_loader, model, criterion, optimizer):
    """
    Train the GIN using the provided training data.

    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader for iterating through training data batches.
        model (torch.nn.Module): The graph neural network model to be trained.
        criterion (torch.nn.Module): The loss function used for training.
        optimizer (torch.optim.Optimizer): The optimizer responsible for updating model parameters.

    Returns:
        None
    """

    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
        try:
            out = model(data.x.type(torch.float32), data.edge_index, data.batch)  # Perform a single forward pass.
            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
        except Exception as e:
            print(str(e) + "In train")
            continue


def test(test_loader, model):
    """
       Evaluate the GIN on the provided test data.

       Args:
           test_loader (torch.utils.data.DataLoader): DataLoader for iterating through test data batches.
           model (torch.nn.Module): The graph neural network model to be evaluated.

       Returns:
           float: Accuracy of the model on the test data.
           float: AUC (Area Under the ROC Curve) score of the model on the test data.
       """

    model.eval()
    correct = 0
    auc_score = 0
    total_samples = 0
    y_true_list = []
    y_score_list = []
    for data in test_loader:  # Iterate in batches over the training/test dataset.
        try:
            out = model(data.x.type(torch.float32), data.edge_index, data.batch)
            pred = out.argmax(dim=1)  # Use the class with the highest probability.
            correct += int((pred == data.y).sum().item())  # Check against ground-truth labels.
            total_samples += data.y.size(0)
            arr2 = out[:, 1].detach().numpy()
            y_score_list.append(arr2[0])
            arr1 = data.y.detach().numpy()
            y_true_list.append(arr1[0])
        except Exception as e:
            print(str(e) + "In Test")
            continue

    try:
        auc_score += roc_auc_score(y_true=y_true_list, y_score=y_score_list, multi_class='ovr', average='weighted')
    except Exception as e:
        print(e)
        pass

    accuracy = correct / total_samples
    return accuracy, auc_score


if __name__ == "__main__":
    networkList = ["networkaeternity.txt", "networkaion.txt", "networkaragon.txt", "networkbancor.txt",
                   "networkcentra.txt", "networkcindicator.txt", "networkcoindash.txt", "networkdgd.txt",
                   "networkiconomi.txt"]
    for network in networkList:
        print("Working on {}\n".format(network))
        # ** should select the right data folder based on the data **
        data = read_torch_time_series_data(network, "Overlap_xx_Ncube_x")
        for i in range(1, 6):
            print(f"RUN {i}")
            GIN_classifier(data, network)


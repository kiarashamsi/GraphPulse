import os
import numpy as np
import pandas as pd
import networkx as nx
import time
import torch
from torch_geometric.utils import train_test_split_edges
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data
import pickle
from script.utils.make_edges_orign import mask_edges_det, mask_edges_prd, mask_edges_prd_new_by_marlin
from script.utils.make_edges_new import get_edges, get_prediction_edges, get_prediction_edges_modified, get_new_prediction_edges, get_new_prediction_edges_modified


def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def prepare_dir(output_folder):
    mkdirs(output_folder)
    log_folder = mkdirs(output_folder)
    return log_folder


def load_vgrnn_dataset(dataset):
    assert dataset in ['enron10', 'dblp']  # using vgrnn dataset
    print('>> loading on vgrnn dataset')
    with open('../data/input/raw/{}/adj_time_list.pickle'.format(dataset), 'rb') as handle:
        adj_time_list = pickle.load(handle, encoding='iso-8859-1')
    print('>> generating edges, negative edges and new edges, wait for a while ...')
    data = {}
    edges, biedges = mask_edges_det(adj_time_list)  # list
    pedges, nedges = mask_edges_prd(adj_time_list)  # list
    new_pedges, new_nedges = mask_edges_prd_new_by_marlin(adj_time_list)  # list
    print('>> processing finished!')
    assert len(edges) == len(biedges) == len(pedges) == len(nedges) == len(new_nedges) == len(new_pedges)
    edge_index_list, pedges_list, nedges_list, new_nedges_list, new_pedges_list = [], [], [], [], []
    for t in range(len(biedges)):
        edge_index_list.append(torch.tensor(np.transpose(biedges[t]), dtype=torch.long))
        pedges_list.append(torch.tensor(np.transpose(pedges[t]), dtype=torch.long))
        nedges_list.append(torch.tensor(np.transpose(nedges[t]), dtype=torch.long))
        new_pedges_list.append(torch.tensor(np.transpose(new_pedges[t]), dtype=torch.long))
        new_nedges_list.append(torch.tensor(np.transpose(new_nedges[t]), dtype=torch.long))

    data['edge_index_list'] = edge_index_list
    data['pedges'], data['nedges'] = pedges_list, nedges_list
    data['new_pedges'], data['new_nedges'] = new_pedges_list, new_nedges_list  # list
    data['num_nodes'] = int(np.max(np.vstack(edges))) + 1

    data['time_length'] = len(edge_index_list)
    data['weights'] = None
    print('>> data: {}'.format(dataset))
    print('>> total length:{}'.format(len(edge_index_list)))
    print('>> number nodes: {}'.format(data['num_nodes']))
    return data


def load_new_dataset(dataset):
    print('>> loading on new dataset')
    data = {}
    rawfile = '../data/input/processed/{}/{}.pt'.format(dataset, dataset)
    edge_index_list = torch.load(rawfile)  # format: list:[[[1,2],[2,3],[3,4]]]
    undirected_edges = get_edges(edge_index_list)
    num_nodes = int(np.max(np.hstack(undirected_edges))) + 1
    pedges, nedges = get_prediction_edges(undirected_edges)  # list
    new_pedges, new_nedges = get_new_prediction_edges(undirected_edges, num_nodes)

    data['edge_index_list'] = undirected_edges
    data['pedges'], data['nedges'] = pedges, nedges
    data['new_pedges'], data['new_nedges'] = new_pedges, new_nedges  # list
    data['num_nodes'] = num_nodes
    data['time_length'] = len(edge_index_list)
    data['weights'] = None
    print('>> INFO: data: {}'.format(dataset))
    print('>> INFO: total length: {}'.format(len(edge_index_list)))
    print('>> INFO: number nodes: {}'.format(data['num_nodes']))
    return data


def load_vgrnn_dataset_det(dataset):
    assert dataset in ['enron10', 'dblp']  # using vgrnn dataset
    print('>> loading on vgrnn dataset')
    with open('../data/input/raw/{}/adj_time_list.pickle'.format(dataset), 'rb') as handle:
        adj_time_list = pickle.load(handle, encoding='iso-8859-1')
    print('>> generating edges, negative edges and new edges, wait for a while ...')
    data = {}
    edges, biedges = mask_edges_det(adj_time_list)  # list
    pedges, nedges = mask_edges_prd(adj_time_list)  # list
    new_pedges, new_nedges = mask_edges_prd_new_by_marlin(adj_time_list)  # list
    print('>> processing finished!')
    assert len(edges) == len(biedges) == len(pedges) == len(nedges) == len(new_nedges) == len(new_pedges)
    edge_index_list, pedges_list, nedges_list, new_nedges_list, new_pedges_list = [], [], [], [], []
    for t in range(len(biedges)):
        edge_index_list.append(torch.tensor(np.transpose(biedges[t]), dtype=torch.long))
        pedges_list.append(torch.tensor(np.transpose(pedges[t]), dtype=torch.long))
        nedges_list.append(torch.tensor(np.transpose(nedges[t]), dtype=torch.long))
        new_pedges_list.append(torch.tensor(np.transpose(new_pedges[t]), dtype=torch.long))
        new_nedges_list.append(torch.tensor(np.transpose(new_nedges[t]), dtype=torch.long))

    data['edge_index_list'] = edge_index_list
    data['pedges'], data['nedges'] = pedges_list, nedges_list
    data['new_pedges'], data['new_nedges'] = new_pedges_list, new_nedges_list  # list
    data['num_nodes'] = int(np.max(np.vstack(edges))) + 1

    data['time_length'] = len(edge_index_list)
    data['weights'] = None
    print('>> data: {}'.format(dataset))
    print('>> total length:{}'.format(len(edge_index_list)))
    print('>> number nodes: {}'.format(data['num_nodes']))
    return data


def load_new_dataset_det(dataset):
    print('>> INFO: loading on new dataset')
    data = {}
    rawfile = '../data/input/processed/{}/{}.pt'.format(dataset, dataset)
    edge_index_list = torch.load(rawfile)  # format: list:[[[1,2],[2,3],[3,4]]]
    undirected_edges = get_edges(edge_index_list)
    num_nodes = int(np.max(np.hstack(undirected_edges))) + 1

    gdata_list = []
    for edge_index in undirected_edges:
        gdata = Data(x=None, edge_index=edge_index, num_nodes=num_nodes)
        gdata_list.append(train_test_split_edges(gdata, 0.1, 0.4))

    data['gdata'] = gdata_list
    data['num_nodes'] = num_nodes
    data['time_length'] = len(edge_index_list)
    data['weights'] = None
    print('>> INFO: data: {}'.format(dataset))
    print('>> INFO: total length: {}'.format(len(edge_index_list)))
    print('>> INFO: number nodes: {}'.format(data['num_nodes']))
    return data


def load_continuous_time_dataset(dataset, neg_sample):
    print("INFO: Loading a continuous-time dataset: {}".format(dataset))
    data = {}
    p_rawfile = '../data/input/continuous_time/{}_pedges_{}.pt'.format(dataset, neg_sample)  # positive edges
    n_rawfile = '../data/input/continuous_time/{}_nedges_{}.pt'.format(dataset, neg_sample)  # negative edges

    # positive edges
    pedge_index_list = torch.load(p_rawfile)  # format: list:[[[1,2],[2,3],[3,4]]]
    p_undirected_edges = get_edges(pedge_index_list)
    # negative edges
    nedge_index_list = torch.load(n_rawfile)  # format: list:[[[1,2],[2,3],[3,4]]]
    n_undirected_edges = get_edges(nedge_index_list)

    num_nodes = int(np.max(np.hstack(p_undirected_edges))) + 1  # only care about positive number of nodes

    pedges = get_prediction_edges_modified(p_undirected_edges)  # list
    nedges = get_prediction_edges_modified(n_undirected_edges)  # list

    new_pedges = get_new_prediction_edges_modified(p_undirected_edges, num_nodes)
    new_nedges = get_new_prediction_edges_modified(n_undirected_edges, num_nodes)

    data['edge_index_list'] = p_undirected_edges
    data['pedges'], data['nedges'] = pedges, nedges
    data['new_pedges'], data['new_nedges'] = new_pedges, new_nedges  # list
    data['num_nodes'] = num_nodes
    data['time_length'] = len(pedge_index_list)
    data['weights'] = None
    print('>> INFO: Data: {}'.format(dataset))
    print('>> INFO: Total length: {}'.format(len(pedge_index_list)))
    print('>> INFO: Number nodes: {}'.format(data['num_nodes']))
    return data


def load_TGC_dataset(dataset):
    print("INFO: Loading a Graph from `Temporal Graph Classification (TGC)` Category: {}".format(dataset))
    data = {}
    edgelist_rawfile = '../data/input/raw/{}/{}_edgelist.txt'.format(dataset, dataset)
    edgelist_df = pd.read_csv(edgelist_rawfile)
    uniq_ts_list = np.unique(edgelist_df['snapshot'])
    print("INFO: Number of unique snapshots: {}".format(len(uniq_ts_list)))
    adj_time_list = []
    for ts in uniq_ts_list:
        # NOTE: this code does not use any node or edge features
        ts_edges = edgelist_df.loc[edgelist_df['snapshot'] == ts, ['source', 'destination']]
        ts_G = nx.from_pandas_edgelist(ts_edges, 'source', 'destination')
        ts_A = nx.to_scipy_sparse_array(ts_G)
        adj_time_list.append(ts_A)

    # Now, exactly like "load_vgrnn_dataset_det"
    print('INFO: Generating edges, negative edges and new edges, wait for a while ...')
    edge_proc_start = time.time()
    data = {}
    edges, biedges = mask_edges_det(adj_time_list)  # list
    pedges, nedges = mask_edges_prd(adj_time_list)  # list
    new_pedges, new_nedges = mask_edges_prd_new_by_marlin(adj_time_list)  # list
    print('INFO: Processing finished! Elapsed time (sec.): {:.4}'.format(time.time() - edge_proc_start))
    assert len(edges) == len(biedges) == len(pedges) == len(nedges) == len(new_nedges) == len(new_pedges)
    edge_index_list, pedges_list, nedges_list, new_nedges_list, new_pedges_list = [], [], [], [], []
    for t in range(len(biedges)):
        edge_index_list.append(torch.tensor(np.transpose(biedges[t]), dtype=torch.long))
        pedges_list.append(torch.tensor(np.transpose(pedges[t]), dtype=torch.long))
        nedges_list.append(torch.tensor(np.transpose(nedges[t]), dtype=torch.long))
        new_pedges_list.append(torch.tensor(np.transpose(new_pedges[t]), dtype=torch.long))
        new_nedges_list.append(torch.tensor(np.transpose(new_nedges[t]), dtype=torch.long))

    data['edge_index_list'] = edge_index_list
    data['pedges'], data['nedges'] = pedges_list, nedges_list
    data['new_pedges'], data['new_nedges'] = new_pedges_list, new_nedges_list  # list
    data['num_nodes'] = int(np.max(np.vstack(edges))) + 1

    data['time_length'] = len(edge_index_list)
    data['weights'] = None
    print('INFO: Data: {}'.format(dataset))
    print('INFO: Total length:{}'.format(len(edge_index_list)))
    print('INFO: Number nodes: {}'.format(data['num_nodes']))
    return data


def loader(dataset='enron10', neg_sample=''):
    # if cached, load directly
    data_root = '../data/input/cached/{}/'.format(dataset)
    filepath = mkdirs(data_root) + '{}.data'.format(dataset)  # the data will be saved here after generation.
    print("INFO: Dataset: {}".format(dataset))
    print("DEBUG: Look for data at {}.".format(filepath))
    if os.path.isfile(filepath):
        print('INFO: Loading {} directly.'.format(dataset))
        return torch.load(filepath)
    
    # if not cached, to process and cached
    print('INFO: data does not exits, processing ...')
    if dataset in ['enron10', 'dblp']:
        data = load_vgrnn_dataset(dataset)
    elif dataset in ['as733', 'fbw', 'HepPh30', 'disease']:
        data = load_new_dataset(dataset)
    elif dataset in ['canVote', 'LegisEdgelist', 'wikipedia', 'UNtrade']:
        print("INFO: Loading a continuous-time dynamic graph dataset: {}".format(dataset))
        data = load_continuous_time_dataset(dataset, neg_sample)
    elif dataset in ['adex', 'aeternity', 'aion', 'aragon', 'bancor', 'centra', 'cindicator', 
                     'coindash', 'dgd', 'iconomi',  'mathoverflow', 'RedditB', 'CollegeMsg']:
        print("INFO: Loading a dynamic graph datasets for TG-Classification: {}".format(dataset))
        data = load_TGC_dataset(dataset)
    else:
        raise ValueError("ERROR: Undefined dataset!")
    torch.save(data, filepath)
    print('INFO: Dataset is saved!')
    return data

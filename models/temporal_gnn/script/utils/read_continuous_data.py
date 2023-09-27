"""
Read the edge list of the continuous-time datasets and process them to be in correct format for the discrete-time models
Date:
    - Aug. 12, 2022
"""

import datetime
import pickle
import torch
import networkx as nx
import pandas as pd
import numpy as np
import argparse



def save_nx_graph(nx_graph, path='dyn_graphs_dataset.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(nx_graph, f, protocol=pickle.HIGHEST_PROTOCOL)  # the higher the protocol, the smaller the file

def save_edges(edges, dataset_name, partial_path, suffix=''):
    path = '{}/{}{}'.format(partial_path, dataset_name, suffix)
    torch.save(edges, path + '.pt')
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(edges, f, protocol=pickle.HIGHEST_PROTOCOL)  # the higher the protocol, the smaller the file
    print("INFO: {}{} edge list is saved!".format(dataset_name, suffix))

def getID(node_id, nodes_dict):
    if node_id not in nodes_dict.keys():
        idx = len(nodes_dict)
        nodes_dict[node_id] = idx
    else:
        idx = nodes_dict[node_id]
    return idx, nodes_dict

def CT_time_analysis(edgelist_filename, val_ratio=0.15, test_ratio=0.15):
    """
    analyze the temporal information of the continuous-time edge list
    """
    ct_edgelist = pd.read_csv(edgelist_filename)
    val_time, test_time = list(np.quantile(ct_edgelist['ts'], [(1 - val_ratio - test_ratio), (1 - test_ratio)]))
    print("INFO: Validation timestamp: {}, Test timestamp: {}".format(val_time, test_time))
    print("INFO: Number of unique timestamps: ", len(pd.unique(ct_edgelist['ts'])))
    # print("DEBUG: List of unique timestamps: ", pd.unique(ct_edgelist['ts']))
    return val_time, test_time

def gen_DT_graph_from_CT_edgelist(partial_path, dataset_name, suffix, gap, val_time, test_time,
                                  no_graphs_to_save, all_snapshots=True):
    f"""
    generate a Discrete-Time graph from its Continuous-Time edgelist
    NOTE: 
        I will used the "ml_{dataset_name}.csv" file which is pre-processed and only contains the edges without features
        Also, timestamps in these datasets are in cardinal format, and the edges are sorted based on their timestamps
    """
    path = '{}/ml_{}{}.csv'.format(partial_path, dataset_name, suffix)
    print("INFO: Input file path:", path)
    df = pd.read_csv(path)  # ,u,i,ts,label,idx
    all_timestamps = len(pd.unique(df['ts']))
    if all_snapshots:
        ignore_init_snapshots = 0
    else:
        ignore_init_snapshots = 20  # ignore the first 20 snapshots and consider the more recent history; "20" is an arbitrary parameter

    print("INFO: Number of all edges: {}".format(len(df)))
    print("INFO: Number of unique timestamps: {}".format(all_timestamps))

    # --- check the time order, if not ascending, resort it ---
    tmp = df['ts'][0]
    for i in range(len(df['ts'])):
        if df['ts'][i] > tmp:
            tmp = df['ts'][i]
        elif df['ts'][i] == tmp:
            pass
        else:
            print("INFO: not in ascending order: should sort it!")
            print("INFO: ", df[i-2: i+2])
            df.sort_values(by='ts', ascending=True, inplace=True)
            df.reset_index(inplace=True)
            print("INFO: ", df[i-2: i+2])
            break
        if i == len(df['ts']) - 1:
            print("INFO: All checked: ascending!")

    # --- generate a graph and dynamic graph snapshots ---
    cnt_graphs = 0
    val_graph_idx, test_graph_idx = 0, 0
    val_graph_idx_set, test_graph_idx_set = False, False
    graphs = []
    g = nx.Graph()
    tmp = df['ts'][0]  # time is in ascending order
    for i in range(len(df['ts'])):
        # fining the graph snapshot index of val_time or test_time
        if tmp > val_time and not val_graph_idx_set:
            val_graph_idx = cnt_graphs
            val_graph_idx_set = True
        if tmp > test_time and not test_graph_idx_set:
            test_graph_idx = cnt_graphs
            test_graph_idx_set = True
        # graph processing...
        if tmp == df['ts'][i]:  # if tmp is in current timestamp
            g.add_edge(str(df['u'][i]), str(df['i'][i]))
            if i == len(df['ts']) - 1:  # EOF ---
                cnt_graphs += 1
                print('INFO: processed graphs: ', cnt_graphs, '/', all_timestamps, '; ALL done...')
        elif tmp < df['ts'][i]:  # if goes to next day
            cnt_graphs += 1
            if (cnt_graphs // gap) >= ignore_init_snapshots and cnt_graphs % gap == 0:
                g.remove_edges_from(nx.selfloop_edges(g, data=True))
                g.remove_nodes_from(list(nx.isolates(g)))
                graphs.append(g.copy())  # append previous g; for a part of graphs to reduce ROM
                g = nx.Graph()  # reset graph, based on the real-world application
            if cnt_graphs % 50 == 0:
                print('INFO: processed graphs: ', cnt_graphs, '/', all_timestamps)
            tmp = df['ts'][i]
            g.add_edge(str(df['u'][i]), str(df['i'][i]))
        else:
            print('INFO: ERROR -- EXIT -- please double check if time is in ascending order!')
            exit(0)

    # --- take out and save part of graphs ----
    print('INFO: total graphs: ', len(graphs))
    print("INFO: validation graphs start index: {}, Test graphs start index: {}".format(val_graph_idx, test_graph_idx))
    # print('INFO: we take out and save the last {} graphs...'.format(no_graphs_to_save))

    if not all_snapshots:
        print("INFO: NOTE that only {} graph snapshots have been saved, not all!!!".format(no_graphs_to_save))
        raw_graphs = graphs[-no_graphs_to_save:]
    else:
        print("INFO: ALL graph snapshots are saved!")
        raw_graphs = graphs
    print("INFO: Number of graphs that have been selected to be saved: ", len(raw_graphs))

    # remap node index:
    G = nx.Graph()  # whole graph, to count number of nodes and edges
    graphs = []  # graph list, to save remapped graphs
    nodes_dict = {}  # node re-id index, to save mapped index
    edges_list = []  # edge_index lsit, sparse matrix
    for i, raw_graph in enumerate(raw_graphs):
        g = nx.Graph()
        for edge in raw_graph.edges:
            idx_i, nodes_dict = getID(edge[0], nodes_dict)
            idx_j, nodes_dict = getID(edge[1], nodes_dict)
            g.add_edge(idx_i, idx_j)
        graphs.append(g)  # append to graph list
        edges_list.append(list(g.edges))  # append to edge list
        G.add_edges_from(g.edges)  # append to the whole graphs
        print('INFO: @ graph', i, '; # of nodes', len(graphs[i].nodes()), '; # of edges', len(graphs[i].edges()))
    print('INFO: time gap is {}'.format(gap))
    print('INFO: total edges: {}'.format(G.number_of_edges()))
    print('INFO: total nodes: {}'.format(G.number_of_nodes()))
    save_edges(edges_list, dataset_name, partial_path, suffix)
    # print("DEBUG: nodes_dict:", nodes_dict)
    print("INFO: max node idx:", max(nodes_dict.values()) + 1)



def main():
    """
    process a continuous-time dataset to have the correct format
    """
    parser = argparse.ArgumentParser(description='Pre-process CTDG data.')
    parser.add_argument('--dataset', type=str, default='canVote', help='datasets')
    parser.add_argument('--neg_sample', type=str, default='rnd', help='negative edge sampling')
    parser.add_argument('--e_type', type=str, default='p', help='positive or negative edge list')
    args = parser.parse_args()

    gap = 1  # minimum is 1; canVote=1, UNtrade=1
    partial_path = '../data/input/continuous_time/'
    dataset_name = args.dataset
    suffix = f"_{args.e_type}edges_{args.neg_sample}"
    no_graphs_to_save = 30

    print("INFO: dataset: ", dataset_name)
    val_time, test_time = CT_time_analysis(f'{partial_path}/ml_{dataset_name}.csv', val_ratio=0.15, test_ratio=0.15)
    gen_DT_graph_from_CT_edgelist(partial_path, dataset_name, suffix, gap, val_time, test_time,
                                  no_graphs_to_save, all_snapshots=True)


if __name__ == '__main__':
    main()
"""
make a timestamped edge list consisting of the negative edges used during training of CTDG methods
The negative edge generation depends on the negative sampling strategy

Date:
    - Aug. 14, 2022
"""

import numpy as np
import pandas as pd
import random
import torch
import math
import argparse
from negative_sampling_strategy import *


class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)


def get_data(dataset_name, val_ratio, test_ratio, different_new_nodes_between_val_and_test=False):
    ### Load data and train val test split
    graph_df = pd.read_csv('../data/input/continuous_time/ml_{}.csv'.format(dataset_name))

    val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    random.seed(2020)

    node_set = set(sources) | set(destinations)
    n_total_unique_nodes = len(node_set)

    # Compute nodes which appear at test time
    test_node_set = set(sources[timestamps > val_time]).union(
        set(destinations[timestamps > val_time]))
    # Sample nodes which we keep as new nodes (to test inductiveness), so than we have to remove all
    # their edges from training
    new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))

    # Mask saying for each source and destination whether they are new test nodes
    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

    # Mask which is true for edges with both destination and source not being new test nodes (because
    # we want to remove all edges involving any new test node)
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

    # For train we keep edges happening before the validation time which do not involve any new node
    # used for inductiveness
    train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)

    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask])

    # define the new nodes sets for testing inductiveness of the model
    train_node_set = set(train_data.sources).union(train_data.destinations)
    assert len(train_node_set & new_test_node_set) == 0
    new_node_set = node_set - train_node_set

    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
    test_mask = timestamps > test_time

    if different_new_nodes_between_val_and_test:
        n_new_nodes = len(new_test_node_set) // 2
        val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
        test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])

        edge_contains_new_val_node_mask = np.array(
            [(a in val_new_node_set or b in val_new_node_set) for a, b in zip(sources, destinations)])
        edge_contains_new_test_node_mask = np.array(
            [(a in test_new_node_set or b in test_new_node_set) for a, b in zip(sources, destinations)])
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_val_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)
    else:
        edge_contains_new_node_mask = np.array(
            [(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    # validation and test with all edges
    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                    edge_idxs[val_mask], labels[val_mask])

    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                     edge_idxs[test_mask], labels[test_mask])

    # validation and test with edges that at least has one new node (not in training set)
    new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                             timestamps[new_node_val_mask],
                             edge_idxs[new_node_val_mask], labels[new_node_val_mask])

    new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                              timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
                              labels[new_node_test_mask])

    print("INFO: The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                                 full_data.n_unique_nodes))
    print("INFO: The training dataset has {} interactions, involving {} different nodes".format(
        train_data.n_interactions, train_data.n_unique_nodes))
    print("INFO: The validation dataset has {} interactions, involving {} different nodes".format(
        val_data.n_interactions, val_data.n_unique_nodes))
    print("INFO:The test dataset has {} interactions, involving {} different nodes".format(
        test_data.n_interactions, test_data.n_unique_nodes))
    print("INFO: The new node validation dataset has {} interactions, involving {} different nodes".format(
        new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
    print("INFO: The new node test dataset has {} interactions, involving {} different nodes".format(
        new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))
    print("INFO: {} nodes were used for the inductive testing, i.e. are never seen during training".format(
        len(new_test_node_set)))

    return full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data


def get_edges_of_split(random_sampler, data, batch_size, neg_sample):
    num_instance = len(data.sources)
    num_batch = math.ceil(num_instance / batch_size)

    p_source, p_dest, p_ts = [], [], []
    n_source, n_dest, n_ts = [], [], []
    for k in range(0, num_batch):
        start_idx = k * batch_size
        end_idx = min(num_instance, start_idx + batch_size)
        source_batch, destination_batch = data.sources[start_idx: end_idx], \
                                          data.destinations[start_idx: end_idx]
        timestamp_batch = data.timestamps[start_idx: end_idx]
        size = len(source_batch)
        if neg_sample == 'rnd':
            negative_source_batch, negative_destination_batch = random_sampler.sample(size)
            negative_source_batch = source_batch

        p_source  = p_source + list(source_batch)
        p_dest = p_dest + list(destination_batch)
        p_ts = p_ts + list(timestamp_batch)

        n_source = n_source + list(negative_source_batch)
        n_dest = n_dest + list(negative_destination_batch)
        n_ts = n_ts + list(timestamp_batch)

    pos_edge_list_df = pd.DataFrame(zip(p_source, p_dest, p_ts), columns=['u', 'i', 'ts'])
    neg_edge_list_df = pd.DataFrame(zip(n_source, n_dest, n_ts), columns=['u', 'i', 'ts'])

    return pos_edge_list_df, neg_edge_list_df


def gen_pos_and_neg_CT_edgelist(dataset_name, neg_sample='rnd'):
    """
    generate positive and negative edge samples for training and testing
    """
    BATCH_SIZE = 200
    val_ratio, test_ratio = 0.15, 0.15
    pedges_path = f'../data/input/continuous_time/ml_{dataset_name}_pedges_{neg_sample}.csv'
    nedges_path = f'../data/input/continuous_time/ml_{dataset_name}_nedges_{neg_sample}.csv'

    # load the original data
    print("INFO: First, load the data.")
    full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = get_data(dataset_name,
                                                                                                 val_ratio,
                                                                                                 test_ratio,True)
    print("INFO: define negative samplers.")
    # define negative edge samplers
    train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
    val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
    nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations, seed=1)
    test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
    nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations, seed=3)

    # TRAINING
    print("INFO: start generating training edges...")
    tr_pos_edge_list, tr_neg_edge_list = get_edges_of_split(train_rand_sampler, train_data, BATCH_SIZE, neg_sample)

    # VALIDATION
    print("INFO: start generating validation edges...")
    val_pos_edge_list, val_neg_edge_list = get_edges_of_split(nn_val_rand_sampler, val_data, BATCH_SIZE, neg_sample)

    # UNSEEN VALIDATION
    print("INFO: start generating unseen validation edges...")
    nn_val_pos_edge_list, nn_val_neg_edge_list = get_edges_of_split(val_rand_sampler, new_node_val_data, BATCH_SIZE,
                                                                    neg_sample)
    # TEST
    print("INFO: start generating test edges...")
    ts_pos_edge_list, ts_neg_edge_list = get_edges_of_split(test_rand_sampler, test_data, BATCH_SIZE, neg_sample)

    # UNSEEN TEST
    print("INFO: start generating unseen test edges...")
    nn_ts_pos_list, nn_ts_neg_list = get_edges_of_split(nn_test_rand_sampler, new_node_test_data, BATCH_SIZE, neg_sample)

    pos_edgelist_df = pd.concat([tr_pos_edge_list, val_pos_edge_list, nn_val_pos_edge_list, ts_pos_edge_list, nn_ts_pos_list])
    pos_edgelist_df = pos_edgelist_df.sort_values(by=['ts'])
    pos_edgelist_df.to_csv(pedges_path, index=False)
    print("INFO: positive edges are saved.")

    neg_edgelist_df = pd.concat([tr_neg_edge_list, val_neg_edge_list, nn_val_neg_edge_list, ts_neg_edge_list, nn_ts_neg_list])
    neg_edgelist_df = neg_edgelist_df.sort_values(by=['ts'])
    neg_edgelist_df.to_csv(nedges_path, index=False)
    print("INFO: negative edges are saved.")


def main():
    """
    generate positive and negative edges; continuous edge list
    python utils/preprocess_ct_data.py
    """
    parser = argparse.ArgumentParser(description='Pre-process CTDG data.')
    parser.add_argument('--dataset', type=str, default='canVote', help='datasets')
    parser.add_argument('--neg_sample', type=str, default='rnd', help='negative edge sampling')
    args = parser.parse_args()

    dataset_name = args.dataset
    neg_sample = args.neg_sample
    gen_pos_and_neg_CT_edgelist(dataset_name, neg_sample)


if __name__ == '__main__':
    main()


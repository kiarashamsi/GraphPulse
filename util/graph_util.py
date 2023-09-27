import os
import shutil
from collections import defaultdict
from multiprocessing import Process
import config
import networkx as nx
import pandas as pd
import datetime as dt
import numpy as np
import multiprocessing as mp
import kmapper as km
import sklearn
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from typing import Any, Iterable, List, Optional, Tuple, Union
import torch
from torch import Tensor
import pickle
import matplotlib.ticker as mticker
from config import *

def read_and_merge_node_network_count(file_path):
    """
    Reads and merges node network counts from multiple files into a single dictionary.

    Args:
    file_path (str): The path to the directory containing the files to be processed.

    Returns:
    None
    """
    # Create an empty dictionary to store the merged result
    merged_dict = {}

    # Loop through all files with a .txt extension in the specified directory
    indx = 0
    files = os.listdir(file_path + "NodeExistenceMatrix")
    for file in files:
        if file.endswith(".txt"):
            indx += 1
            print("Processing {}/{}".format(indx, len(files)))
            with open(file_path + "NodeExistenceMatrix/" + file, 'r') as f:
                file_dict = eval(f.read())
            # Merge the dictionary with the merged result
            for key in file_dict:
                if key in merged_dict:
                    merged_dict[key] += file_dict[key]
                else:
                    merged_dict[key] = file_dict[key]

    # Print the merged dictionary to a file
    with open(file_path + "FInal_NodeTokenNetworkHashMap.txt", 'w+') as data:
        data.write(str(merged_dict))

def process_motifs(file):
    """
    Process motifs from a network data file.

    Args:
    file (str): The name of the network data file to process.

    Returns:
    None
    """
    # Print a message to indicate processing has started
    print("Processing {}".format(file))

    # Read the network data from the specified file
    selected_network = pd.read_csv((file_path + file), sep=' ', names=["from", "to", "date", "value"])
    selected_network['date'] = pd.to_datetime(selected_network['date'], unit='s').dt.date

    # Iterate through specified time windows
    for time_frame in time_window:
        print("\nProcessing Timeframe {} ".format(time_frame))

        # Create an empty directed graph for the transaction data
        transaction_graphs = nx.DiGraph()

        # Calculate start and end dates
        start_date = selected_network['date'].min()
        last_date_of_data = selected_network['date'].max()

        # Check if the network has enough data for validation
        if ((last_date_of_data - start_date).days < network_validation_duration):
            print(file + " is not a valid network")
            shutil.move(file_path + file, file_path + "Invalid/" + file)
            continue

        # Select only the rows that fall within the current time frame
        end_date = start_date + dt.timedelta(days=time_frame)
        selected_network_in_time_frame = selected_network[
            (selected_network['date'] >= start_date) & (selected_network['date'] < end_date)]

        # Populate the graph with edges based on the selected network data
        for item in selected_network_in_time_frame.to_dict(orient="records"):
            transaction_graphs.add_edge(item["from"], item["to"], value=item["value"])

        # Initialize motifs and statistics
        motifs = None
        stats = {'network': file, 'motifs': motifs}

        # Create a DataFrame to store statistics
        stat_data = pd.DataFrame(columns=['network', 'motifs'])
        stat_data = stat_data.append(stats, ignore_index=True)

        # Save statistics to a CSV file
        stat_data.to_csv('final_data_motifs.csv', mode='a', header=False)

        # Move the processed file to the appropriate directory
        shutil.move(file_path + file, file_path + "Processed/" + file)

        # Increment the processing index
        config.processing_index += 1

        # Print a message to indicate processing has finished
        print("\nFinishing processing {} \n".format(file + "   " + str(time_frame)))

def from_networkx(
            G: Any, label: int,
            group_node_attrs: Optional[Union[List[str], all]] = None,
            group_edge_attrs: Optional[Union[List[str], all]] = None,
    ) -> 'torch_geometric.data.Data':
        r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
        :class:`torch_geometric.data.Data` instance.

        Args:
            G (networkx.Graph or networkx.DiGraph): A networkx graph.
            group_node_attrs (List[str] or all, optional): The node attributes to
                be concatenated and added to :obj:`data.x`. (default: :obj:`None`)
            group_edge_attrs (List[str] or all, optional): The edge attributes to
                be concatenated and added to :obj:`data.edge_attr`.
                (default: :obj:`None`)

        .. note::

            All :attr:`group_node_attrs` and :attr:`group_edge_attrs` values must
            be numeric.

        Examples:

            # >>> edge_index = torch.tensor([
            # ...     [0, 1, 1, 2, 2, 3],
            # ...     [1, 0, 2, 1, 3, 2],
            # ... ])
            # >>> data = Data(edge_index=edge_index, num_nodes=4)
            # >>> g = to_networkx(data)
            # >>> # A `Data` object is returned
            # >>> from_networkx(g)
            Data(edge_index=[2, 6], num_nodes=4)
        """
        import networkx as nx

        from torch_geometric.data import Data

        G = nx.convert_node_labels_to_integers(G)
        G = G.to_directed() if not nx.is_directed(G) else G

        if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
            edges = list(G.edges(keys=False))
        else:
            edges = list(G.edges)

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        data = defaultdict(list)

        if G.number_of_nodes() > 0:
            node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
        else:
            node_attrs = {}

        if G.number_of_edges() > 0:
            edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
        else:
            edge_attrs = {}

        for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
            if set(feat_dict.keys()) != set(node_attrs):
                raise ValueError('Not all nodes contain the same attributes')
            for key, value in feat_dict.items():
                data[str(key)].append(value)

        for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
            if set(feat_dict.keys()) != set(edge_attrs):
                raise ValueError('Not all edges contain the same attributes')
            for key, value in feat_dict.items():
                key = f'edge_{key}' if key in node_attrs else key
                data[str(key)].append(value)

        for key, value in G.graph.items():
            key = f'graph_{key}' if key in node_attrs else key
            data[str(key)] = value

        for key, value in data.items():
            if isinstance(value, (tuple, list)) and isinstance(value[0], Tensor):
                data[key] = torch.stack(value, dim=0)
            else:
                try:
                    data[key] = torch.tensor(value)
                except (ValueError, TypeError):
                    pass

        data['edge_index'] = edge_index.view(2, -1)
        data = Data.from_dict(data)
        data['y'] = label

        if group_node_attrs is all:
            group_node_attrs = list(node_attrs)
        if group_node_attrs is not None:
            xs = []
            for key in group_node_attrs:
                x = data[key]
                x = x.view(-1, 1) if x.dim() <= 1 else x
                xs.append(x)
                del data[key]
            data.x = torch.cat(xs, dim=-1)

        if group_edge_attrs is all:
            group_edge_attrs = list(edge_attrs)
        if group_edge_attrs is not None:
            xs = []
            for key in group_edge_attrs:
                key = f'edge_{key}' if key in node_attrs else key
                x = data[key]
                x = x.view(-1, 1) if x.dim() <= 1 else x
                xs.append(x)
                del data[key]
            data.edge_attr = torch.cat(xs, dim=-1)

        if data.x is None and data.pos is None:
            data.num_nodes = G.number_of_nodes()

        return data

def get_daily_node_avg(file):
        selectedNetwork = pd.read_csv((timeseries_file_path + file), sep=' ',
                                      names=["from", "to", "date", "value"])
        selectedNetwork['date'] = pd.to_datetime(selectedNetwork['date'], unit='s').dt.date
        selectedNetwork['value'] = selectedNetwork['value'].astype(float)
        selectedNetwork = selectedNetwork.sort_values(by='date')
        window_start_date = selectedNetwork['date'].min()
        data_last_date = selectedNetwork['date'].max()
        end_date = window_start_date + dt.timedelta(days=7)
        selectedNetworkInTimeFrame = selectedNetwork[
            (selectedNetwork['date'] >= window_start_date) & (selectedNetwork['date'] < end_date)]
        print("Daily node avg of {} is {} \n".format(file, (
            (len(set(selectedNetworkInTimeFrame['from'].unique() + selectedNetworkInTimeFrame['to'].unique())) / 7))))

def get_daily_avg(file):
    timeseries_file_path = "../data/all_network/TimeSeries/"
    timeseries_file_path_other = "../data/all_network/TimeSeries/Other/"
    selectedNetwork = pd.read_csv((timeseries_file_path_other + file), sep=' ', names=["from", "to", "date"])
    selectedNetwork['value'] = 1
    selectedNetwork['date'] = pd.to_datetime(selectedNetwork['date'], unit='s').dt.date
    selectedNetwork['value'] = selectedNetwork['value'].astype(float)
    selectedNetwork = selectedNetwork.sort_values(by='date')
    window_start_date = selectedNetwork['date'].min()
    data_last_date = selectedNetwork['date'].max()
    days_of_data = (data_last_date - window_start_date).days
    avg_daily_trans = len(selectedNetwork) / days_of_data
    # Concatenate the two columns
    combined = pd.concat([selectedNetwork['from'], selectedNetwork['to']], ignore_index=True)
    # Get the number of unique items
    num_unique = combined.nunique()
    avg_daily_nodes = num_unique / days_of_data
    print(
        f"AVG daily stat for {file} -> nodes = {avg_daily_nodes} , edges = {avg_daily_trans} , days = {days_of_data} , total trans = {len(selectedNetwork)}")

def get_daily_avg_reddit(file):
    timeseries_file_path = "../data/all_network/TimeSeries/"
    timeseries_file_path_other = "../data/all_network/TimeSeries/Other/"

    selectedNetwork = pd.read_csv((timeseries_file_path_other + file), sep='\t')
    selectedNetwork = selectedNetwork[["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "TIMESTAMP", "LINK_SENTIMENT"]]
    column_mapping = {
        'SOURCE_SUBREDDIT': 'from',
        'TARGET_SUBREDDIT': 'to',
        'TIMESTAMP': 'date',
        'LINK_SENTIMENT': 'value'
    }
    selectedNetwork.rename(columns=column_mapping, inplace=True)
    selectedNetwork['date'] = pd.to_datetime(selectedNetwork['date']).dt.date
    selectedNetwork['value'] = selectedNetwork['value'].astype(float)
    selectedNetwork = selectedNetwork.sort_values(by='date')
    # reddit 800
    window_start_date = selectedNetwork['date'].min() + dt.timedelta(days=800)
    data_last_date = selectedNetwork['date'].max()
    days_of_data = (data_last_date - window_start_date).days
    avg_daily_trans = len(selectedNetwork) / days_of_data
    # Concatenate the two columns
    combined = pd.concat([selectedNetwork['from'], selectedNetwork['to']], ignore_index=True)
    # Get the number of unique items
    num_unique = combined.nunique()
    avg_daily_nodes = num_unique / days_of_data
    print(
        f"AVG daily stat for {file} -> nodes = {avg_daily_nodes} , edges = {avg_daily_trans} , days = {days_of_data} , total trans = {len(selectedNetwork)}, Range {window_start_date} {data_last_date}")

def process_data_duration(self, file):
    # load each network file
    # Timer(2, functools.partial(self.exitfunc, file_path, file)).start()
    print("Processing {}".format(file))
    selectedNetwork = pd.read_csv((self.file_path + file), sep=' ', names=["from", "to", "date", "value"])

    start_date = selectedNetwork['date'].min()
    last_date_of_data = selectedNetwork['date'].max()
    days = (last_date_of_data - start_date).days

    stats = {'network': file, "data_duration": days}

    stat_date = self.stat_date.iloc[0:0]
    stat_date = stat_date.append(stats, ignore_index=True)
    stat_date.to_csv('final_data_date.csv', mode='a', header=False)
    shutil.move(self.file_path + file, self.file_path + "Pr/" + file)

def create_node_token_network_count(file, bucket):
    print("Processing {}".format(file))
    nodeHashMap = {}

    eachNetworkNodeHashMap = []
    selectedNetwork = pd.read_csv((file_path + bucket + "/" + file), sep=' ',
                                  names=["from", "to", "date", "value"])
    selectedNetwork = selectedNetwork.drop('value', axis=1)
    selectedNetwork = selectedNetwork.drop('date', axis=1)
    unique_node_ids = np.unique(selectedNetwork[['from', 'to']].values)
    id = 0
    for nodeID in unique_node_ids:
        # calculate the number of token participation
        id += 1
        # print("{} / {} inside file ".format(id , len(unique_node_ids)))
        if nodeID not in eachNetworkNodeHashMap:
            # first time we see this node ID in the current network
            if nodeID not in nodeHashMap:
                nodeHashMap[nodeID] = 1
            else:
                nodeHashMap[nodeID] += 1
            eachNetworkNodeHashMap.append(nodeID)
    with open(file_path + bucket + "/NodeCount/" + file.split(".")[0] + '_NodeTokenNetworkHashMap.txt',
              'w+') as data:
        data.write(str(nodeHashMap))
        data.close()

def multiprocess_node_network_count():
    print("Process Started\n")
    config.processing_index = 0

    master_list = ['bucket_1', 'bucket_2', 'bucket_3', 'bucket_4', 'bucket_5']
    with mp.Pool() as pool:
        results = pool.map(process_bucket_node_network_count, master_list, chunksize=1)

def process_bucket_node_network_count(bucket):
    print("in bucket" + bucket + "\n")
    files = os.listdir(file_path + bucket)
    for file in files:
        if file.endswith(".txt"):
            process_bucket_node_network_count(file, bucket)
            shutil.move(file_path + bucket + "/" + file, file_path + bucket + "/Pr/" + file)
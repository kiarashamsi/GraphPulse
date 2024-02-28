import ast
import csv
import multiprocessing
import os
import shutil
from collections import defaultdict
from multiprocessing import Process
import networkx as nx
import pandas as pd
import datetime as dt
import numpy as np
import multiprocessing as mp
import kmapper as km
import sklearn
from sklearn.preprocessing import MinMaxScaler
import pickle
from util.graph_util import from_networkx

'''
    The following script will parse each transaction network and provide detailed stats about them. 

'''


class NetworkParser:
    # Path of the dataset folder
    file_path = "../data/all_network/"
    timeseries_file_path = "../data/all_network/TimeSeries/"
    timeseries_file_path_other = "../data/all_network/TimeSeries/Other/"
    timeWindow = [7]
    # Validation duration condition
    networkValidationDuration = 20
    finalDataDuration = 5
    labelTreshholdPercentage = 10

    # Retrieve dataset by call to dataloader
    # Ethereum Stable Coin ERC20
    stat_data = pd.DataFrame(columns=['network', 'timeframe', 'start_date', 'num_nodes',
                                      'num_edges', 'density', 'diameter',
                                      'avg_shortest_path_length', 'max_degree_centrality',
                                      'min_degree_centrality',
                                      'max_closeness_centrality', 'min_closeness_centrality',
                                      'max_betweenness_centrality',
                                      'min_betweenness_centrality',
                                      'assortativity', 'clique_number', 'motifs', "peak", "last_dates_trans",
                                      "label_factor_percentage",
                                      "label"])

    stat_date = pd.DataFrame(columns=['network', "data_duration"])
    processingIndx = 1

    def create_graph_features(self, file):
        """
          Process a network file to create graph features and save them for further analysis.

          Args:
              file (str): The filename of the network data file to be processed.

          Returns:
              None
          """

        # load each network file
        print("Processing {}".format(file))
        selectedNetwork = pd.read_csv((self.file_path + file), sep=' ', names=["from", "to", "date", "value"])
        selectedNetwork['date'] = pd.to_datetime(selectedNetwork['date'], unit='s').dt.date

        # generate the label for this network
        date_counts = selectedNetwork.groupby(['date']).count()
        peak_count = date_counts['value'].max()

        # Calculate the sum of the value column for the last two dates
        last_date_sum = date_counts['value'].tail(self.finalDataDuration).sum()
        if (last_date_sum / peak_count) * 100 > self.labelTreshholdPercentage:
            label = "live"
        else:
            label = 'dead'

        for timeFrame in self.timeWindow:
            print("\nProcessing Timeframe {} ".format(timeFrame))
            transactionGraph = nx.MultiDiGraph()
            start_date = selectedNetwork['date'].min()
            last_date_of_data = selectedNetwork['date'].max()
            # check if the network has more than 20 days of data
            if ((last_date_of_data - start_date).days < self.networkValidationDuration):
                print(file + "Is not a valid network")
                shutil.move(self.file_path + file, self.file_path + "Invalid/" + file)
                continue

            # select only the rows that fall within the first 7 days
            end_date = start_date + dt.timedelta(days=timeFrame)
            selectedNetworkInTimeFrame = selectedNetwork[
                (selectedNetwork['date'] >= start_date) & (selectedNetwork['date'] < end_date)]

            # Populate graph with edges
            for item in selectedNetworkInTimeFrame.to_dict(orient="records"):
                transactionGraph.add_edge(item["from"], item["to"], value=item["value"])

            with open('NetworkxGraphs/' + file, 'wb') as f:
                pickle.dump(transactionGraph, f)

            num_nodes = len(transactionGraph.nodes())
            num_edges = len(transactionGraph.edges())
            density = nx.density(transactionGraph)
            transactionGraph = transactionGraph.to_undirected()
            if (nx.is_connected(transactionGraph)):
                diameter = nx.diameter(transactionGraph)
                avg_shortest_path_length = nx.average_shortest_path_length(transactionGraph)
                clique_number = nx.graph_clique_number(transactionGraph)
            else:
                Gcc = sorted(nx.connected_components(transactionGraph), key=len, reverse=True)
                biggestConnectedComponent = transactionGraph.subgraph(Gcc[0])
                diameter = nx.diameter(biggestConnectedComponent)
                avg_shortest_path_length = nx.average_shortest_path_length(biggestConnectedComponent)
                clique_number = nx.graph_clique_number(biggestConnectedComponent)

            max_degree_centrality = max(nx.degree_centrality(transactionGraph).values())
            min_degree_centrality = min(nx.degree_centrality(transactionGraph).values())
            max_closeness_centrality = max(nx.closeness_centrality(transactionGraph).values())
            min_closeness_centrality = min(nx.closeness_centrality(transactionGraph).values())
            max_betweenness_centrality = max(nx.betweenness_centrality(transactionGraph).values())
            min_betweenness_centrality = min(nx.betweenness_centrality(transactionGraph).values())
            # max_eigenvector_centrality = max(nx.eigenvector_centrality(transactionGraph).values())
            # min_eigenvector_centrality = min(nx.eigenvector_centrality(transactionGraph).values())
            # pagerank = nx.pagerank(transactionGraph)
            assortativity = nx.degree_assortativity_coefficient(transactionGraph)
            # try:
            #     clique_number = nx.graph_clique_number(transactionGraph)
            # except Exception as e:
            #     clique_number = "NC"
            motifs = ""
            stats = {'network': file, 'timeframe': timeFrame, 'start_date': start_date, 'num_nodes': num_nodes,
                     'num_edges': num_edges, 'density': density, 'diameter': diameter,
                     'avg_shortest_path_length': avg_shortest_path_length,
                     'max_degree_centrality': max_degree_centrality,
                     'min_degree_centrality': min_degree_centrality,
                     'max_closeness_centrality': max_closeness_centrality,
                     'min_closeness_centrality': min_closeness_centrality,
                     'max_betweenness_centrality': max_betweenness_centrality,
                     'min_betweenness_centrality': min_betweenness_centrality,
                     'assortativity': assortativity, 'clique_number': clique_number, 'motifs': motifs,
                     "peak": peak_count, "last_dates_trans": last_date_sum,
                     "label_factor_percentage": (last_date_sum / peak_count),
                     "label": label}

            stat_data = self.stat_data.iloc[0:0]
            stat_data = stat_data.append(stats, ignore_index=True)
            stat_data.to_csv('final_data.csv', mode='a', header=False)
            shutil.move(self.file_path + file, self.file_path + "Processed/" + file)
            self.processingIndx += 1
            # nx.draw(transactionGraph, node_size=10)
            # plt.savefig("images/" + file + "_" + str(timeFrame) + "_" + ".png")

            print("\nFinisheng processing {} \n".format(file + "   " + str(timeFrame)))

    def create_time_series_graphs(self, file):
        """
        Process a time series network data file for transaction network format files to create temporal graph
        snapshots in geometric tensorflow format and save them for further analysis.

          Args:
              file (str): The filename of the time series network data file to be processed.

          Returns:
              None
        """
        print("Processing {}".format(file))
        windowSize = 7  # Day
        gap = 3
        lableWindowSize = 7  # Day
        maxDuration = 180  # Day
        indx = 0
        maxIndx = 2

        selectedNetwork = pd.read_csv((self.timeseries_file_path + file), sep=' ',
                                      names=["from", "to", "date", "value"])
        selectedNetwork['date'] = pd.to_datetime(selectedNetwork['date'], unit='s').dt.date
        selectedNetwork['value'] = selectedNetwork['value'].astype(float)
        selectedNetwork = selectedNetwork.sort_values(by='date')
        window_start_date = selectedNetwork['date'].min()
        data_last_date = selectedNetwork['date'].max()

        # print("\n {} Days OF Data -> {} ".format(file, (data_last_date - window_start_date).days ))
        # check if the network has more than 20 days of data
        if ((data_last_date - window_start_date).days < maxDuration):
            print(file + "Is not a valid network")
            shutil.move(self.file_path + file, self.file_path + "Invalid/" + file)
            return

        # normalize the edge weights for the graph network {0-9}
        max_transfer = float(selectedNetwork['value'].max())
        min_transfer = float(selectedNetwork['value'].min())

        selectedNetwork['value'] = selectedNetwork['value'].apply(
            lambda x: 1 + (9 * ((float(x) - min_transfer) / (max_transfer - min_transfer))))

        # Graph Generation Process and Labeling
        while (data_last_date - window_start_date).days > (windowSize + gap + lableWindowSize):
            print("\nRemaining Process {} ".format(
                (data_last_date - window_start_date).days / (windowSize + gap + lableWindowSize)))
            indx += 1
            # if (indx == maxIndx):
            #     break
            transactionGraph = nx.MultiDiGraph()

            # select window data
            window_end_date = window_start_date + dt.timedelta(days=windowSize)
            selectedNetworkInGraphDataWindow = selectedNetwork[
                (selectedNetwork['date'] >= window_start_date) & (selectedNetwork['date'] < window_end_date)]

            # select labeling data
            label_end_date = window_start_date + dt.timedelta(days=windowSize) + dt.timedelta(days=gap) + dt.timedelta(
                days=lableWindowSize)
            label_start_date = window_start_date + dt.timedelta(days=windowSize) + dt.timedelta(days=gap)
            selectedNetworkInLbelingWindow = selectedNetwork[
                (selectedNetwork['date'] >= label_start_date) & (selectedNetwork['date'] < label_end_date)]

            # generating the label for this window
            # 1 -> Increading Transactions 0 -> Decreasing Transactions
            label = 1 if (len(selectedNetworkInLbelingWindow) - len(selectedNetworkInGraphDataWindow)) > 0 else 0

            # group by for extracting node features
            outgoing_weight_sum = (selectedNetwork.groupby(by=['from'])['value'].sum())
            incoming_weight_sum = (selectedNetwork.groupby(by=['to'])['value'].sum())
            outgoing_count = (selectedNetwork.groupby(by=['from'])['value'].count())
            incoming_count = (selectedNetwork.groupby(by=['to'])['value'].count())

            # Node Features Dictionary for TDA mapper usage
            node_features = pd.DataFrame()

            # Populate graph with edges
            for item in selectedNetworkInGraphDataWindow.to_dict(orient="records"):
                from_node_features = {}
                to_node_features = {}
                # calculating node features for each edge
                # feature 1 -> sum of outgoing edge weights
                from_node_features["outgoing_edge_weight_sum"] = outgoing_weight_sum[item['from']]

                try:
                    to_node_features["outgoing_edge_weight_sum"] = outgoing_weight_sum[item['to']]
                except Exception as e:
                    to_node_features["outgoing_edge_weight_sum"] = 0

                # feature 2 -> sum of incoming edge weights
                to_node_features["incoming_edge_weight_sum"] = incoming_weight_sum[item['to']]
                try:
                    from_node_features["incoming_edge_weight_sum"] = incoming_weight_sum[item['from']]
                except Exception as e:
                    from_node_features["incoming_edge_weight_sum"] = 0
                # feature 3 -> number of outgoing edges
                from_node_features["outgoing_edge_count"] = outgoing_count[item['from']]
                try:
                    to_node_features["outgoing_edge_count"] = outgoing_count[item['to']]
                except Exception as e:
                    to_node_features["outgoing_edge_count"] = 0

                # feature 4 -> number of incoming edges
                to_node_features["incoming_edge_count"] = incoming_count[item['to']]
                try:
                    from_node_features["incoming_edge_count"] = incoming_count[item['from']]
                except Exception as e:
                    from_node_features["incoming_edge_count"] = 0

                # add temporal vector to all nodes, populated with -1

                from_node_features_with_daily_temporal_vector = dict(from_node_features)
                from_node_features_with_daily_temporal_vector["dailyClusterID"] = [-1] * windowSize
                from_node_features_with_daily_temporal_vector["dailyClusterSize"] = [-1] * windowSize

                to_node_features_with_daily_temporal_vector = dict(to_node_features)
                to_node_features_with_daily_temporal_vector["dailyClusterID"] = [-1] * windowSize
                to_node_features_with_daily_temporal_vector["dailyClusterSize"] = [-1] * windowSize

                # Temporal version
                transactionGraph.add_nodes_from([(item["from"], from_node_features_with_daily_temporal_vector)])
                transactionGraph.add_nodes_from([(item["to"], to_node_features_with_daily_temporal_vector)])
                transactionGraph.add_edge(item["from"], item["to"], value=item["value"])

                new_row = pd.DataFrame(({**{"nodeID": item["from"]}, **from_node_features}), index=[0])
                node_features = pd.concat([node_features, new_row], ignore_index=True)

                new_row = pd.DataFrame(({**{"nodeID": item["to"]}, **to_node_features}), index=[0])
                node_features = pd.concat([node_features, new_row], ignore_index=True)

                node_features = node_features.drop_duplicates(subset=['nodeID'])

            directory = 'PygGraphs/TimeSeries/' + file

            # Extracting TDA temporal features and adding to the graph
            transactionGraph = self.process_TDA_extracted_temporal_features(selectedNetworkInGraphDataWindow,
                                                                            transactionGraph, node_features)

            # Generating TDA graphs
            self.create_TDA_graph(node_features, label, directory, network=file, timeWindow=indx)

            featureNames = ["outgoing_edge_weight_sum", "incoming_edge_weight_sum", "outgoing_edge_count",
                            "incoming_edge_count", "dailyClusterID", "dailyClusterSize"]
            window_start_date = window_start_date + dt.timedelta(days=1)

            # Generating PyGraphs for timeseries data
            if not os.path.exists(directory):
                os.makedirs(directory)
            pygData = from_networkx(transactionGraph, label=label, group_node_attrs=featureNames)
            with open(directory + "/TemporalVectorizedGraph_Tuned/" + file + "_" + "graph_" + str(indx), 'wb') as f:
                pickle.dump(pygData, f)

    def create_time_series_other_graphs(self, file):
        """
        Process a time series network data for non transaction format networks file to create other temporal graph
        features and save them for analysis.

           Args:
               file (str): The filename of the time series network data file to be processed.

           Returns:
               None
           """

        print("Processing {}".format(file))
        windowSize = 7  # Day
        gap = 3
        lableWindowSize = 7  # Day
        maxDuration = 180  # Day
        indx = 0
        maxIndx = 2

        selectedNetwork = pd.read_csv((self.timeseries_file_path_other + file), sep=' ', names=["from", "to", "date"])
        selectedNetwork['value'] = 1
        selectedNetwork['date'] = pd.to_datetime(selectedNetwork['date'], unit='s').dt.date
        selectedNetwork['value'] = selectedNetwork['value'].astype(float)
        selectedNetwork = selectedNetwork.sort_values(by='date')
        window_start_date = selectedNetwork['date'].min() + dt.timedelta(days=2150)
        data_last_date = selectedNetwork['date'].max()

        # print("\n {} Days OF Data -> {} ".format(file, (data_last_date - window_start_date).days ))
        # check if the network has more than 20 days of data
        if ((data_last_date - window_start_date).days < maxDuration):
            print(file + "Is not a valid network")
            shutil.move(self.file_path + file, self.file_path + "Invalid/" + file)
            return

        # normalize the edge weights for the graph network {0-9}
        max_transfer = float(selectedNetwork['value'].max())
        min_transfer = float(selectedNetwork['value'].min())
        if max_transfer == min_transfer:
            max_transfer = min_transfer + 1

        selectedNetwork['value'] = selectedNetwork['value'].apply(
            lambda x: 1 + (9 * ((float(x) - min_transfer) / (max_transfer - min_transfer))))

        # Graph Generation Process and Labeling
        while (data_last_date - window_start_date).days > (windowSize + gap + lableWindowSize):
            print("\nRemaining Process {} ".format(
                (data_last_date - window_start_date).days / (windowSize + gap + lableWindowSize)))
            indx += 1
            # if (indx == maxIndx):
            #     break
            transactionGraph = nx.MultiDiGraph()

            # select window data
            window_end_date = window_start_date + dt.timedelta(days=windowSize)
            selectedNetworkInGraphDataWindow = selectedNetwork[
                (selectedNetwork['date'] >= window_start_date) & (selectedNetwork['date'] < window_end_date)]

            # select labeling data
            label_end_date = window_start_date + dt.timedelta(days=windowSize) + dt.timedelta(days=gap) + dt.timedelta(
                days=lableWindowSize)
            label_start_date = window_start_date + dt.timedelta(days=windowSize) + dt.timedelta(days=gap)
            selectedNetworkInLbelingWindow = selectedNetwork[
                (selectedNetwork['date'] >= label_start_date) & (selectedNetwork['date'] < label_end_date)]

            # generating the label for this window
            # 1 -> Increading Transactions 0 -> Decreasing Transactions
            label = 1 if (len(selectedNetworkInLbelingWindow) - len(selectedNetworkInGraphDataWindow)) > 0 else 0

            # group by for extracting node features
            outgoing_weight_sum = (selectedNetwork.groupby(by=['from'])['value'].sum())
            incoming_weight_sum = (selectedNetwork.groupby(by=['to'])['value'].sum())
            outgoing_count = (selectedNetwork.groupby(by=['from'])['value'].count())
            incoming_count = (selectedNetwork.groupby(by=['to'])['value'].count())

            # Node Features Dictionary for TDA mapper usage
            node_features = pd.DataFrame()

            # Populate graph with edges
            for item in selectedNetworkInGraphDataWindow.to_dict(orient="records"):
                from_node_features = {}
                to_node_features = {}
                # calculating node features for each edge
                # feature 1 -> sum of outgoing edge weights
                from_node_features["outgoing_edge_weight_sum"] = outgoing_weight_sum[item['from']]

                try:
                    to_node_features["outgoing_edge_weight_sum"] = outgoing_weight_sum[item['to']]
                except Exception as e:
                    to_node_features["outgoing_edge_weight_sum"] = 0

                # feature 2 -> sum of incoming edge weights
                to_node_features["incoming_edge_weight_sum"] = incoming_weight_sum[item['to']]
                try:
                    from_node_features["incoming_edge_weight_sum"] = incoming_weight_sum[item['from']]
                except Exception as e:
                    from_node_features["incoming_edge_weight_sum"] = 0
                # feature 3 -> number of outgoing edges
                from_node_features["outgoing_edge_count"] = outgoing_count[item['from']]
                try:
                    to_node_features["outgoing_edge_count"] = outgoing_count[item['to']]
                except Exception as e:
                    to_node_features["outgoing_edge_count"] = 0

                # feature 4 -> number of incoming edges
                to_node_features["incoming_edge_count"] = incoming_count[item['to']]
                try:
                    from_node_features["incoming_edge_count"] = incoming_count[item['from']]
                except Exception as e:
                    from_node_features["incoming_edge_count"] = 0

                # add temporal vector to all nodes, populated with -1

                from_node_features_with_daily_temporal_vector = dict(from_node_features)
                from_node_features_with_daily_temporal_vector["dailyClusterID"] = [-1] * windowSize
                from_node_features_with_daily_temporal_vector["dailyClusterSize"] = [-1] * windowSize

                to_node_features_with_daily_temporal_vector = dict(to_node_features)
                to_node_features_with_daily_temporal_vector["dailyClusterID"] = [-1] * windowSize
                to_node_features_with_daily_temporal_vector["dailyClusterSize"] = [-1] * windowSize

                # Temporal version
                transactionGraph.add_nodes_from([(item["from"], from_node_features_with_daily_temporal_vector)])
                transactionGraph.add_nodes_from([(item["to"], to_node_features_with_daily_temporal_vector)])
                transactionGraph.add_edge(item["from"], item["to"], value=item["value"])

                new_row = pd.DataFrame(({**{"nodeID": item["from"]}, **from_node_features}), index=[0])
                node_features = pd.concat([node_features, new_row], ignore_index=True)

                new_row = pd.DataFrame(({**{"nodeID": item["to"]}, **to_node_features}), index=[0])
                node_features = pd.concat([node_features, new_row], ignore_index=True)

                node_features = node_features.drop_duplicates(subset=['nodeID'])

            directory = 'PygGraphs/TimeSeries/' + file

            # Extracting TDA temporal features and adding to the graph
            print("Generating TDA temporal graph \n")
            transactionGraph = self.process_TDA_extracted_temporal_features(selectedNetworkInGraphDataWindow,transactionGraph, node_features)

            # Generating TDA graphs
            print("Generating TDA raw graph \n")
            self.createTDAGraph(node_features, label, directory, network=file, timeWindow=indx)

            featureNames = ["outgoing_edge_weight_sum", "incoming_edge_weight_sum", "outgoing_edge_count",
                             "incoming_edge_count", "dailyClusterID", "dailyClusterSize"]

            featureNames = ["outgoing_edge_weight_sum", "incoming_edge_weight_sum", "outgoing_edge_count",
                            "incoming_edge_count"]
            window_start_date = window_start_date + dt.timedelta(days=1)
            #
            # Generating PyGraphs for timeseries data
            print("Generating raw graph \n")
            if not os.path.exists(directory):
                os.makedirs(directory)
            pygData = self.from_networkx(transactionGraph, label=label, group_node_attrs=featureNames)
            with open(directory + "/RawGraph/" + file + "_" + "graph_" + str(indx), 'wb') as f:
                pickle.dump(pygData, f)

    def create_time_series_reddit_graphs(self, file):
        print("Processing {}".format(file))
        windowSize = 7  # Day
        gap = 3
        lableWindowSize = 7  # Day
        maxDuration = 180  # Day
        indx = 0
        maxIndx = 2

        selectedNetwork = pd.read_csv((self.timeseries_file_path_other + file), sep='\t')
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

        # print("\n {} Days OF Data -> {} ".format(file, (data_last_date - window_start_date).days ))
        # check if the network has more than 20 days of data
        if ((data_last_date - window_start_date).days < maxDuration):
            print(file + "Is not a valid network")
            shutil.move(self.file_path + file, self.file_path + "Invalid/" + file)
            return

        # normalize the edge weights for the graph network {0-9}
        max_transfer = float(selectedNetwork['value'].max())
        min_transfer = float(selectedNetwork['value'].min())
        if max_transfer == min_transfer:
            max_transfer = min_transfer + 1
        # Calculate mean and standard deviation
        # mean = np.mean(selectedNetwork['value'])
        # std = np.std(selectedNetwork['value'])

        # selectedNetwork['value'] = selectedNetwork['value'].apply(lambda x: (x - mean) / std)

        selectedNetwork['value'] = selectedNetwork['value'].apply(
            lambda x: 1 + (9 * ((float(x) - min_transfer) / (max_transfer - min_transfer))))

        # Graph Generation Process and Labeling
        while (data_last_date - window_start_date).days > (windowSize + gap + lableWindowSize):
            print("\nRemaining Process {} ".format(
                (data_last_date - window_start_date).days / (windowSize + gap + lableWindowSize)))
            indx += 1
            # if (indx == maxIndx):
            #     break
            transactionGraph = nx.MultiDiGraph()

            # select window data
            window_end_date = window_start_date + dt.timedelta(days=windowSize)
            selectedNetworkInGraphDataWindow = selectedNetwork[
                (selectedNetwork['date'] >= window_start_date) & (selectedNetwork['date'] < window_end_date)]

            # select labeling data
            label_end_date = window_start_date + dt.timedelta(days=windowSize) + dt.timedelta(days=gap) + dt.timedelta(
                days=lableWindowSize)
            label_start_date = window_start_date + dt.timedelta(days=windowSize) + dt.timedelta(days=gap)
            selectedNetworkInLbelingWindow = selectedNetwork[
                (selectedNetwork['date'] >= label_start_date) & (selectedNetwork['date'] < label_end_date)]

            # generating the label for this window
            # 1 -> Increading Transactions 0 -> Decreasing Transactions
            label = 1 if (len(selectedNetworkInLbelingWindow) - len(selectedNetworkInGraphDataWindow)) > 0 else 0

            # group by for extracting node features
            outgoing_weight_sum = (selectedNetwork.groupby(by=['from'])['value'].sum())
            incoming_weight_sum = (selectedNetwork.groupby(by=['to'])['value'].sum())
            outgoing_count = (selectedNetwork.groupby(by=['from'])['value'].count())
            incoming_count = (selectedNetwork.groupby(by=['to'])['value'].count())

            # Node Features Dictionary for TDA mapper usage
            node_features = pd.DataFrame()

            # Populate graph with edges
            for item in selectedNetworkInGraphDataWindow.to_dict(orient="records"):
                from_node_features = {}
                to_node_features = {}
                # calculating node features for each edge
                # feature 1 -> sum of outgoing edge weights
                from_node_features["outgoing_edge_weight_sum"] = outgoing_weight_sum[item['from']]

                try:
                    to_node_features["outgoing_edge_weight_sum"] = outgoing_weight_sum[item['to']]
                except Exception as e:
                    to_node_features["outgoing_edge_weight_sum"] = 0

                # feature 2 -> sum of incoming edge weights
                to_node_features["incoming_edge_weight_sum"] = incoming_weight_sum[item['to']]
                try:
                    from_node_features["incoming_edge_weight_sum"] = incoming_weight_sum[item['from']]
                except Exception as e:
                    from_node_features["incoming_edge_weight_sum"] = 0
                # feature 3 -> number of outgoing edges
                from_node_features["outgoing_edge_count"] = outgoing_count[item['from']]
                try:
                    to_node_features["outgoing_edge_count"] = outgoing_count[item['to']]
                except Exception as e:
                    to_node_features["outgoing_edge_count"] = 0

                # feature 4 -> number of incoming edges
                to_node_features["incoming_edge_count"] = incoming_count[item['to']]
                try:
                    from_node_features["incoming_edge_count"] = incoming_count[item['from']]
                except Exception as e:
                    from_node_features["incoming_edge_count"] = 0

                # add temporal vector to all nodes, populated with -1

                from_node_features_with_daily_temporal_vector = dict(from_node_features)
                from_node_features_with_daily_temporal_vector["dailyClusterID"] = [-1] * windowSize
                from_node_features_with_daily_temporal_vector["dailyClusterSize"] = [-1] * windowSize

                to_node_features_with_daily_temporal_vector = dict(to_node_features)
                to_node_features_with_daily_temporal_vector["dailyClusterID"] = [-1] * windowSize
                to_node_features_with_daily_temporal_vector["dailyClusterSize"] = [-1] * windowSize


                # Temporal version
                transactionGraph.add_nodes_from([(item["from"], from_node_features_with_daily_temporal_vector)])
                transactionGraph.add_nodes_from([(item["to"], to_node_features_with_daily_temporal_vector)])
                transactionGraph.add_edge(item["from"], item["to"], value=item["value"])

                new_row = pd.DataFrame(({**{"nodeID": item["from"]}, **from_node_features}), index=[0])
                node_features = pd.concat([node_features, new_row], ignore_index=True)

                new_row = pd.DataFrame(({**{"nodeID": item["to"]}, **to_node_features}), index=[0])
                node_features = pd.concat([node_features, new_row], ignore_index=True)

                node_features = node_features.drop_duplicates(subset=['nodeID'])

            directory = 'PygGraphs/TimeSeries/' + file

            # Extracting TDA temporal features and adding to the graph
            # print("Generating TDA temporal graph \n")
            # transactionGraph = self.processTDAExtractedTemporalFeatures(selectedNetworkInGraphDataWindow,transactionGraph, node_features)
            #
            # # Generating TDA graphs
            # print("Generating TDA raw graph \n")
            # self.createTDAGraph(node_features, label, directory, network=file, timeWindow=indx)

            # featureNames = ["outgoing_edge_weight_sum", "incoming_edge_weight_sum", "outgoing_edge_count",
            #                  "incoming_edge_count", "dailyClusterID", "dailyClusterSize"]

            featureNames = ["outgoing_edge_weight_sum", "incoming_edge_weight_sum", "outgoing_edge_count",
                            "incoming_edge_count"]
            window_start_date = window_start_date + dt.timedelta(days=1)
            #
            # Generating PyGraphs for timeseries data
            print("Generating raw graph \n")
            if not os.path.exists(directory):
                os.makedirs(directory)
            pygData = self.from_networkx(transactionGraph, label=label, group_node_attrs=featureNames)
            with open(directory + "/RawGraph/" + file + "_" + "graph_" + str(indx), 'wb') as f:
                pickle.dump(pygData, f)

    def create_time_series_rnn_sequence(self, file):
        totalRnnSequenceData = list()
        totalRnnLabelData = list()
        print("Processing {}".format(file))
        windowSize = 7  # Day
        gap = 3
        lableWindowSize = 7  # Day
        maxDuration = 180  # Day
        indx = 0
        maxIndx = 2

        selectedNetwork = pd.read_csv((self.timeseries_file_path + file), sep=' ',
                                      names=["from", "to", "date", "value"])
        selectedNetwork['date'] = pd.to_datetime(selectedNetwork['date'], unit='s').dt.date
        selectedNetwork['value'] = selectedNetwork['value'].astype(float)
        selectedNetwork = selectedNetwork.sort_values(by='date')
        window_start_date = selectedNetwork['date'].min()
        data_last_date = selectedNetwork['date'].max()

        print(f"{file} -- {window_start_date} -- {data_last_date}")

        print("\n {} Days OF Data -> {} ".format(file, (data_last_date - window_start_date).days))
        # check if the network has more than 20 days of data
        if ((data_last_date - window_start_date).days < maxDuration):
            print(file + "Is not a valid network")
            shutil.move(self.file_path + file, self.file_path + "Invalid/" + file)
            return

        # normalize the edge weights for the graph network {0-9}
        max_transfer = float(selectedNetwork['value'].max())
        min_transfer = float(selectedNetwork['value'].min())

        selectedNetwork['value'] = selectedNetwork['value'].apply(
            lambda x: 1 + (9 * ((float(x) - min_transfer) / (max_transfer - min_transfer))))

        # Graph Generation Process and Labeling

        while (data_last_date - window_start_date).days > (windowSize + gap + lableWindowSize):
            print("\nRemaining Process  {} ".format(

                (data_last_date - window_start_date).days / (windowSize + gap + lableWindowSize)))
            indx += 1
            # if (indx == maxIndx):
            #     break
            transactionGraph = nx.MultiDiGraph()

            # select window data
            window_end_date = window_start_date + dt.timedelta(days=windowSize)
            selectedNetworkInGraphDataWindow = selectedNetwork[
                (selectedNetwork['date'] >= window_start_date) & (
                        selectedNetwork['date'] < window_end_date)]

            # select labeling data
            label_end_date = window_start_date + dt.timedelta(days=windowSize) + dt.timedelta(
                days=gap) + dt.timedelta(
                days=lableWindowSize)
            label_start_date = window_start_date + dt.timedelta(days=windowSize) + dt.timedelta(days=gap)
            selectedNetworkInLbelingWindow = selectedNetwork[
                (selectedNetwork['date'] >= label_start_date) & (selectedNetwork['date'] < label_end_date)]

            # generating the label for this window
            # 1 -> Increading Transactions 0 -> Decreasing Transactions
            label = 1 if (len(selectedNetworkInLbelingWindow) - len(
                selectedNetworkInGraphDataWindow)) > 0 else 0

            # group by for extracting node features
            outgoing_weight_sum = (selectedNetwork.groupby(by=['from'])['value'].sum())
            incoming_weight_sum = (selectedNetwork.groupby(by=['to'])['value'].sum())
            outgoing_count = (selectedNetwork.groupby(by=['from'])['value'].count())
            incoming_count = (selectedNetwork.groupby(by=['to'])['value'].count())

            # Node Features Dictionary for TDA mapper usage
            node_features = pd.DataFrame()

            # Populate graph with edges
            for item in selectedNetworkInGraphDataWindow.to_dict(orient="records"):
                from_node_features = {}
                to_node_features = {}
                # calculating node features for each edge
                # feature 1 -> sum of outgoing edge weights
                from_node_features["outgoing_edge_weight_sum"] = outgoing_weight_sum[item['from']]

                try:
                    to_node_features["outgoing_edge_weight_sum"] = outgoing_weight_sum[item['to']]
                except Exception as e:
                    to_node_features["outgoing_edge_weight_sum"] = 0

                # feature 2 -> sum of incoming edge weights
                to_node_features["incoming_edge_weight_sum"] = incoming_weight_sum[item['to']]
                try:
                    from_node_features["incoming_edge_weight_sum"] = incoming_weight_sum[item['from']]
                except Exception as e:
                    from_node_features["incoming_edge_weight_sum"] = 0
                # feature 3 -> number of outgoing edges
                from_node_features["outgoing_edge_count"] = outgoing_count[item['from']]
                try:
                    to_node_features["outgoing_edge_count"] = outgoing_count[item['to']]
                except Exception as e:
                    to_node_features["outgoing_edge_count"] = 0

                # feature 4 -> number of incoming edges
                to_node_features["incoming_edge_count"] = incoming_count[item['to']]
                try:
                    from_node_features["incoming_edge_count"] = incoming_count[item['from']]
                except Exception as e:
                    from_node_features["incoming_edge_count"] = 0

                # add temporal vector to all nodes, populated with -1

                from_node_features_with_daily_temporal_vector = dict(from_node_features)
                from_node_features_with_daily_temporal_vector["dailyClusterID"] = [-1] * windowSize
                from_node_features_with_daily_temporal_vector["dailyClusterSize"] = [-1] * windowSize

                to_node_features_with_daily_temporal_vector = dict(to_node_features)
                to_node_features_with_daily_temporal_vector["dailyClusterID"] = [-1] * windowSize
                to_node_features_with_daily_temporal_vector["dailyClusterSize"] = [-1] * windowSize

                # Temporal version
                transactionGraph.add_nodes_from(
                    [(item["from"], from_node_features_with_daily_temporal_vector)])
                transactionGraph.add_nodes_from([(item["to"], to_node_features_with_daily_temporal_vector)])
                transactionGraph.add_edge(item["from"], item["to"], value=item["value"])

                new_row = pd.DataFrame(({**{"nodeID": item["from"]}, **from_node_features}), index=[0])
                node_features = pd.concat([node_features, new_row], ignore_index=True)

                new_row = pd.DataFrame(({**{"nodeID": item["to"]}, **to_node_features}), index=[0])
                node_features = pd.concat([node_features, new_row], ignore_index=True)

                node_features = node_features.drop_duplicates(subset=['nodeID'])

            timeWindowSequence = self.process_TDA_extracted_rnn_sequence(selectedNetworkInGraphDataWindow, node_features)

            # timeWindowSequenceRaw = self.processRawExtractedRnnSequence(selectedNetworkInGraphDataWindow, node_features)
            # result_list = []
            # first_key_tda = next(iter(timeWindowSequence))
            # tda_value = timeWindowSequence[first_key_tda]
            #
            # first_key_raw = next(iter(timeWindowSequenceRaw))
            # raw_value = timeWindowSequenceRaw[first_key_raw]
            #
            # for sublist1, sublist2 in zip(tda_value, raw_value):
            #     merged_sublist = sublist1 + sublist2
            #     result_list.append(merged_sublist)
            #
            #
            # totalRnnSequenceData.append({first_key_tda + "_" + first_key_raw : result_list})

            totalRnnSequenceData.append(timeWindowSequence)
            totalRnnLabelData.append(label)
            window_start_date = window_start_date + dt.timedelta(days=1)

        total_merged_seq = self.merge_dicts(totalRnnSequenceData)
        finalDict = {"sequence": total_merged_seq, "label": totalRnnLabelData}
        directory = 'Sequence/' + str(file)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(directory + '/seq_tda_ablation.txt',
                  'wb') as file_in:
            pickle.dump(finalDict, file_in)
            file_in.close()

    def create_time_series_for_other_dataset_rnn_sequence(self, file, raw=False):
        totalRnnSequenceData = list()
        totalRnnLabelData = list()
        print("Processing {}".format(file))
        windowSize = 7  # Day
        gap = 3
        lableWindowSize = 7  # Day
        maxDuration = 20  # Day
        indx = 0
        maxIndx = 2

        selectedNetwork = pd.read_csv((self.timeseries_file_path_other + file), sep=' ', names=["from", "to", "date"])
        selectedNetwork['value'] = 1
        selectedNetwork['date'] = pd.to_datetime(selectedNetwork['date'], unit='s').dt.date
        selectedNetwork['value'] = selectedNetwork['value'].astype(float)
        selectedNetwork = selectedNetwork.sort_values(by='date')
        # math stack 2150
        if "math" in file:
            window_start_date = selectedNetwork['date'].min() + dt.timedelta(2150)
        else:
            window_start_date = selectedNetwork['date'].min()
        data_last_date = selectedNetwork['date'].max()

        print(f"{file} -- {window_start_date} -- {data_last_date}")

        print("\n {} Days OF Data -> {} ".format(file, (data_last_date - window_start_date).days))
        # check if the network has more than 20 days of data
        if ((data_last_date - window_start_date).days < maxDuration):
            print(file + "Is not a valid network")
            shutil.move(self.file_path + file, self.file_path + "Invalid/" + file)
            return

        # normalize the edge weights for the graph network {0-9}
        max_transfer = float(selectedNetwork['value'].max())
        min_transfer = float(selectedNetwork['value'].min())
        if max_transfer == min_transfer:
            max_transfer = min_transfer + 1

        selectedNetwork['value'] = selectedNetwork['value'].apply(
            lambda x: 1 + (9 * ((float(x) - min_transfer) / (max_transfer - min_transfer))))

        # Graph Generation Process and Labeling

        while (data_last_date - window_start_date).days > (windowSize + gap + lableWindowSize):
            print("\nRemaining Process  {} ".format(

                (data_last_date - window_start_date).days / (windowSize + gap + lableWindowSize)))
            indx += 1
            # if (indx == maxIndx):
            #     break
            transactionGraph = nx.MultiDiGraph()

            # select window data
            window_end_date = window_start_date + dt.timedelta(days=windowSize)
            selectedNetworkInGraphDataWindow = selectedNetwork[
                (selectedNetwork['date'] >= window_start_date) & (
                        selectedNetwork['date'] < window_end_date)]

            # select labeling data
            label_end_date = window_start_date + dt.timedelta(days=windowSize) + dt.timedelta(
                days=gap) + dt.timedelta(
                days=lableWindowSize)
            label_start_date = window_start_date + dt.timedelta(days=windowSize) + dt.timedelta(days=gap)
            selectedNetworkInLbelingWindow = selectedNetwork[
                (selectedNetwork['date'] >= label_start_date) & (selectedNetwork['date'] < label_end_date)]

            # generating the label for this window
            # 1 -> Increading Transactions 0 -> Decreasing Transactions
            label = 1 if (len(selectedNetworkInLbelingWindow) - len(
                selectedNetworkInGraphDataWindow)) > 0 else 0

            # group by for extracting node features
            outgoing_weight_sum = (selectedNetwork.groupby(by=['from'])['value'].sum())
            incoming_weight_sum = (selectedNetwork.groupby(by=['to'])['value'].sum())
            outgoing_count = (selectedNetwork.groupby(by=['from'])['value'].count())
            incoming_count = (selectedNetwork.groupby(by=['to'])['value'].count())

            # Node Features Dictionary for TDA mapper usage
            node_features = pd.DataFrame()

            # Populate graph with edges
            for item in selectedNetworkInGraphDataWindow.to_dict(orient="records"):
                from_node_features = {}
                to_node_features = {}
                # calculating node features for each edge
                # feature 1 -> sum of outgoing edge weights
                from_node_features["outgoing_edge_weight_sum"] = outgoing_weight_sum[item['from']]

                try:
                    to_node_features["outgoing_edge_weight_sum"] = outgoing_weight_sum[item['to']]
                except Exception as e:
                    to_node_features["outgoing_edge_weight_sum"] = 0

                # feature 2 -> sum of incoming edge weights
                to_node_features["incoming_edge_weight_sum"] = incoming_weight_sum[item['to']]
                try:
                    from_node_features["incoming_edge_weight_sum"] = incoming_weight_sum[item['from']]
                except Exception as e:
                    from_node_features["incoming_edge_weight_sum"] = 0
                # feature 3 -> number of outgoing edges
                from_node_features["outgoing_edge_count"] = outgoing_count[item['from']]
                try:
                    to_node_features["outgoing_edge_count"] = outgoing_count[item['to']]
                except Exception as e:
                    to_node_features["outgoing_edge_count"] = 0

                # feature 4 -> number of incoming edges
                to_node_features["incoming_edge_count"] = incoming_count[item['to']]
                try:
                    from_node_features["incoming_edge_count"] = incoming_count[item['from']]
                except Exception as e:
                    from_node_features["incoming_edge_count"] = 0

                # add temporal vector to all nodes, populated with -1

                from_node_features_with_daily_temporal_vector = dict(from_node_features)
                from_node_features_with_daily_temporal_vector["dailyClusterID"] = [-1] * windowSize
                from_node_features_with_daily_temporal_vector["dailyClusterSize"] = [-1] * windowSize

                to_node_features_with_daily_temporal_vector = dict(to_node_features)
                to_node_features_with_daily_temporal_vector["dailyClusterID"] = [-1] * windowSize
                to_node_features_with_daily_temporal_vector["dailyClusterSize"] = [-1] * windowSize

                # Temporal version
                transactionGraph.add_nodes_from(
                    [(item["from"], from_node_features_with_daily_temporal_vector)])
                transactionGraph.add_nodes_from([(item["to"], to_node_features_with_daily_temporal_vector)])
                transactionGraph.add_edge(item["from"], item["to"], value=item["value"])

                new_row = pd.DataFrame(({**{"nodeID": item["from"]}, **from_node_features}), index=[0])
                node_features = pd.concat([node_features, new_row], ignore_index=True)

                new_row = pd.DataFrame(({**{"nodeID": item["to"]}, **to_node_features}), index=[0])
                node_features = pd.concat([node_features, new_row], ignore_index=True)

                node_features = node_features.drop_duplicates(subset=['nodeID'])

            file_name = "/seq_tda_ablation.txt"
            if (raw):
                timeWindowSequence = self.process_raw_extracted_rnn_sequence(selectedNetworkInGraphDataWindow,
                                                                             node_features)
                file_name = "/seq_raw.txt"
            else:
                timeWindowSequence = self.process_TDA_extracted_rnn_sequence(selectedNetworkInGraphDataWindow,
                                                                             node_features)

            totalRnnSequenceData.append(timeWindowSequence)
            totalRnnLabelData.append(label)
            window_start_date = window_start_date + dt.timedelta(days=1)

        total_merged_seq = self.merge_dicts(totalRnnSequenceData)
        finalDict = {"sequence": total_merged_seq, "label": totalRnnLabelData}
        directory = 'Sequence/' + str(file)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(directory + file_name,
                  'wb') as file_in:
            pickle.dump(finalDict, file_in)
            file_in.close()

    def create_time_series_for_reddit_dataset_rnn_sequence(self, file, raw=False):
        totalRnnSequenceData = list()
        totalRnnLabelData = list()
        print("Processing {}".format(file))
        windowSize = 7  # Day
        gap = 3
        lableWindowSize = 7  # Day
        maxDuration = 20  # Day
        indx = 0
        maxIndx = 2

        selectedNetwork = pd.read_csv((self.timeseries_file_path_other + file), sep='\t')
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

        print(f"{file} -- {window_start_date} -- {data_last_date}")

        print("\n {} Days OF Data -> {} ".format(file, (data_last_date - window_start_date).days))
        # check if the network has more than 20 days of data
        if ((data_last_date - window_start_date).days < maxDuration):
            print(file + "Is not a valid network")
            shutil.move(self.file_path + file, self.file_path + "Invalid/" + file)
            return

        # normalize the edge weights for the graph network {0-9}
        max_transfer = float(selectedNetwork['value'].max())
        min_transfer = float(selectedNetwork['value'].min())
        if max_transfer == min_transfer:
            max_transfer = min_transfer + 1

        selectedNetwork['value'] = selectedNetwork['value'].apply(
            lambda x: 1 + (9 * ((float(x) - min_transfer) / (max_transfer - min_transfer))))

        # Graph Generation Process and Labeling

        while (data_last_date - window_start_date).days > (windowSize + gap + lableWindowSize):
            print("\nRemaining Process  {} ".format(

                (data_last_date - window_start_date).days / (windowSize + gap + lableWindowSize)))
            indx += 1
            # if (indx == maxIndx):
            #     break
            transactionGraph = nx.MultiDiGraph()

            # select window data
            window_end_date = window_start_date + dt.timedelta(days=windowSize)
            selectedNetworkInGraphDataWindow = selectedNetwork[
                (selectedNetwork['date'] >= window_start_date) & (
                        selectedNetwork['date'] < window_end_date)]

            # select labeling data
            label_end_date = window_start_date + dt.timedelta(days=windowSize) + dt.timedelta(
                days=gap) + dt.timedelta(
                days=lableWindowSize)
            label_start_date = window_start_date + dt.timedelta(days=windowSize) + dt.timedelta(days=gap)
            selectedNetworkInLbelingWindow = selectedNetwork[
                (selectedNetwork['date'] >= label_start_date) & (selectedNetwork['date'] < label_end_date)]

            # generating the label for this window
            # 1 -> Increading Transactions 0 -> Decreasing Transactions
            label = 1 if (len(selectedNetworkInLbelingWindow) - len(
                selectedNetworkInGraphDataWindow)) > 0 else 0

            # group by for extracting node features
            outgoing_weight_sum = (selectedNetwork.groupby(by=['from'])['value'].sum())
            incoming_weight_sum = (selectedNetwork.groupby(by=['to'])['value'].sum())
            outgoing_count = (selectedNetwork.groupby(by=['from'])['value'].count())
            incoming_count = (selectedNetwork.groupby(by=['to'])['value'].count())

            # Node Features Dictionary for TDA mapper usage
            node_features = pd.DataFrame()

            # Populate graph with edges
            for item in selectedNetworkInGraphDataWindow.to_dict(orient="records"):
                from_node_features = {}
                to_node_features = {}
                # calculating node features for each edge
                # feature 1 -> sum of outgoing edge weights
                from_node_features["outgoing_edge_weight_sum"] = outgoing_weight_sum[item['from']]

                try:
                    to_node_features["outgoing_edge_weight_sum"] = outgoing_weight_sum[item['to']]
                except Exception as e:
                    to_node_features["outgoing_edge_weight_sum"] = 0

                # feature 2 -> sum of incoming edge weights
                to_node_features["incoming_edge_weight_sum"] = incoming_weight_sum[item['to']]
                try:
                    from_node_features["incoming_edge_weight_sum"] = incoming_weight_sum[item['from']]
                except Exception as e:
                    from_node_features["incoming_edge_weight_sum"] = 0
                # feature 3 -> number of outgoing edges
                from_node_features["outgoing_edge_count"] = outgoing_count[item['from']]
                try:
                    to_node_features["outgoing_edge_count"] = outgoing_count[item['to']]
                except Exception as e:
                    to_node_features["outgoing_edge_count"] = 0

                # feature 4 -> number of incoming edges
                to_node_features["incoming_edge_count"] = incoming_count[item['to']]
                try:
                    from_node_features["incoming_edge_count"] = incoming_count[item['from']]
                except Exception as e:
                    from_node_features["incoming_edge_count"] = 0

                # add temporal vector to all nodes, populated with -1

                from_node_features_with_daily_temporal_vector = dict(from_node_features)
                from_node_features_with_daily_temporal_vector["dailyClusterID"] = [-1] * windowSize
                from_node_features_with_daily_temporal_vector["dailyClusterSize"] = [-1] * windowSize

                to_node_features_with_daily_temporal_vector = dict(to_node_features)
                to_node_features_with_daily_temporal_vector["dailyClusterID"] = [-1] * windowSize
                to_node_features_with_daily_temporal_vector["dailyClusterSize"] = [-1] * windowSize

                # Temporal version
                transactionGraph.add_nodes_from(
                    [(item["from"], from_node_features_with_daily_temporal_vector)])
                transactionGraph.add_nodes_from([(item["to"], to_node_features_with_daily_temporal_vector)])
                transactionGraph.add_edge(item["from"], item["to"], value=item["value"])

                new_row = pd.DataFrame(({**{"nodeID": item["from"]}, **from_node_features}), index=[0])
                node_features = pd.concat([node_features, new_row], ignore_index=True)

                new_row = pd.DataFrame(({**{"nodeID": item["to"]}, **to_node_features}), index=[0])
                node_features = pd.concat([node_features, new_row], ignore_index=True)

                node_features = node_features.drop_duplicates(subset=['nodeID'])

            file_name = "/seq_tda_ablation.txt"
            if (raw):
                timeWindowSequence = self.process_raw_extracted_rnn_sequence(selectedNetworkInGraphDataWindow,
                                                                             node_features)
                file_name = "/seq_raw.txt"
            else:
                timeWindowSequence = self.process_TDA_extracted_rnn_sequence(selectedNetworkInGraphDataWindow,
                                                                             node_features)

            totalRnnSequenceData.append(timeWindowSequence)
            totalRnnLabelData.append(label)
            window_start_date = window_start_date + dt.timedelta(days=1)

        total_merged_seq = self.merge_dicts(totalRnnSequenceData)
        finalDict = {"sequence": total_merged_seq, "label": totalRnnLabelData}
        directory = 'Sequence/' + str(file)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(directory + file_name,
                  'wb') as file_in:
            pickle.dump(finalDict, file_in)
            file_in.close()

    def create_TDA_graph(self, data, label, htmlPath="", timeWindow=0, network=""):
        try:
            per_overlap = [0.3]
            n_cubes = [2]
            # cls = [2, 5, 10]
            Xfilt = data
            Xfilt = Xfilt.drop(columns=['nodeID'])
            mapper = km.KeplerMapper()
            scaler = MinMaxScaler(feature_range=(0, 1))

            Xfilt = scaler.fit_transform(Xfilt)
            lens = mapper.fit_transform(Xfilt, projection=sklearn.manifold.TSNE())
            cls = 2  # We use cls=5, but this parameter can be further refined.  Its impact on results seems minimal.

            for overlap in per_overlap:
                for n_cube in n_cubes:
                    graph = mapper.map(
                        lens,
                        Xfilt,
                        clusterer=sklearn.cluster.KMeans(n_clusters=cls, random_state=1618033),
                        cover=km.Cover(n_cubes=n_cube, perc_overlap=overlap))  # 0.2 0.4

                    # mapper.visualize(graph,
                    #                  path_html=htmlPath + "/mapper_output_{}_day_{}_cubes_{}_overlap_{}.html".format(
                    #                      network.split(".")[0], timeWindow, n_cube, overlap),
                    #                  title="Mapper graph for network {} in Day {}".format(network.split(".")[0],
                    #                                                                       timeWindow))


                    # removing al the nodes without any edges (Just looking at the links)
                    tdaGraph = nx.Graph()
                    for key, value in graph['links'].items():
                        tdaGraph.add_nodes_from([(key, {"cluster_size": len(graph["nodes"][key])})])
                        for to_add in value:
                            tdaGraph.add_nodes_from([(to_add, {"cluster_size": len(graph["nodes"][to_add])})])
                            tdaGraph.add_edge(key, to_add)

                    # we have the tda Graph here
                    # convert TDA graph to pytorch data
                    directory = 'PygGraphs/TimeSeries/' + network + '/TDA_Tuned/Overlap_{}_Ncube_{}/'.format(overlap,
                                                                                                             n_cube)
                    featureNames = ["cluster_size"]
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    pygData = self.from_networkx(tdaGraph, label=label, group_node_attrs=featureNames)
                    with open(directory + "/" + network + "_" + "TDA_graph(cube-{},overlap-{})_".format(n_cube,
                                                                                                        overlap) + str(
                        timeWindow), 'wb') as f:
                        pickle.dump(pygData, f)

        except Exception as e:
            print(str(e))

    def process_TDA_extracted_temporal_features(self, timeFrameData, originalGraph, nodeFeatures):

        # break the data to daily graphs
        data_first_date = timeFrameData['date'].min()
        data_last_date = timeFrameData['date'].max()
        numberOfDays = (data_last_date - data_first_date).days
        start_date = data_first_date
        # initiate the graph
        originalGraphWithTemporalVector = originalGraph
        processingDay = 0
        while (processingDay <= numberOfDays):
            print("Processing TDA Temporal Feature Extraction day {}".format(processingDay))
            daily_end_date = start_date + dt.timedelta(days=1)
            selectedDailyNetwork = timeFrameData[
                (timeFrameData['date'] >= start_date) & (timeFrameData['date'] < daily_end_date)]

            daily_node_features = pd.DataFrame()

            for item in selectedDailyNetwork.to_dict(orient="records"):
                new_row = pd.DataFrame(({**{"nodeID": item["from"]},
                                         **nodeFeatures[nodeFeatures["nodeID"] == item["to"]].drop("nodeID",
                                                                                                   axis=1).to_dict(
                                             orient='records')[0]}),
                                       index=[0])
                daily_node_features = pd.concat([daily_node_features, new_row], ignore_index=True)

                new_row = pd.DataFrame(({**{"nodeID": item["to"]},
                                         **nodeFeatures[nodeFeatures["nodeID"] == item["to"]].drop("nodeID",
                                                                                                   axis=1).to_dict(
                                             orient='records')[0]}),
                                       index=[0])
                daily_node_features = pd.concat([daily_node_features, new_row], ignore_index=True)

                daily_node_features = daily_node_features.drop_duplicates(subset=['nodeID'])

            # creat the TDA for each day
            try:
                per_overlap = 0.3
                n_cubes = 2
                cls = 2
                Xfilt = daily_node_features
                Xfilt = Xfilt.drop(columns=['nodeID'])
                mapper = km.KeplerMapper()
                scaler = MinMaxScaler(feature_range=(0, 1))

                Xfilt = scaler.fit_transform(Xfilt)
                lens = mapper.fit_transform(Xfilt, projection=sklearn.manifold.TSNE())
                # We use cls=5, but this parameter can be further refined.  Its impact on results seems minimal.

                dailyTdaGraph = mapper.map(
                    lens,
                    Xfilt,
                    clusterer=sklearn.cluster.KMeans(n_clusters=cls, random_state=1618033),
                    cover=km.Cover(n_cubes=n_cubes, perc_overlap=per_overlap))  # 0.2 0.4

                # extract the cluster size and cluster ID vector out of that
                capturedNode = []
                for cluster in dailyTdaGraph["nodes"]:
                    for nodeIndx in dailyTdaGraph["nodes"][cluster]:
                        # check if this
                        if nodeIndx not in capturedNode:
                            # the node has not been captured, we only consider one cluster for nodes in many clusters
                            # combine with the original graph
                            originalGraphWithTemporalVector.nodes[daily_node_features.iloc[nodeIndx]["nodeID"]][
                                "dailyClusterID"][processingDay] = list(dailyTdaGraph["nodes"].keys()).index(cluster)

                            originalGraphWithTemporalVector.nodes[daily_node_features.iloc[nodeIndx]["nodeID"]][
                                "dailyClusterSize"][processingDay] = len(dailyTdaGraph["nodes"][cluster])


                            capturedNode.append(nodeIndx)

            except Exception as e:
                print(str(e))
            start_date = start_date + dt.timedelta(days=1)
            processingDay += 1

        # the graph has been repopulated with daily temporal features
        return originalGraphWithTemporalVector

    def process_TDA_extracted_rnn_sequence(self, timeFrameData, nodeFeatures):

        # break the data to daily graphs
        timeWindowSequence = list()
        data_first_date = timeFrameData['date'].min()
        data_last_date = timeFrameData['date'].max()
        numberOfDays = (data_last_date - data_first_date).days
        start_date = data_first_date
        # initiate the graph
        processingDay = 0
        while (processingDay <= numberOfDays):
            # print("Processing TDA RNN sequential Extraction day {}".format(processingDay))
            daily_end_date = start_date + dt.timedelta(days=1)
            selectedDailyNetwork = timeFrameData[
                (timeFrameData['date'] >= start_date) & (timeFrameData['date'] < daily_end_date)]

            daily_node_features = pd.DataFrame()

            for item in selectedDailyNetwork.to_dict(orient="records"):
                new_row = pd.DataFrame(({**{"nodeID": item["from"]},
                                         **nodeFeatures[nodeFeatures["nodeID"] == item["to"]].drop("nodeID",
                                                                                                   axis=1).to_dict(
                                             orient='records')[0]}),
                                       index=[0])
                daily_node_features = pd.concat([daily_node_features, new_row], ignore_index=True)

                new_row = pd.DataFrame(({**{"nodeID": item["to"]},
                                         **nodeFeatures[nodeFeatures["nodeID"] == item["to"]].drop("nodeID",
                                                                                                   axis=1).to_dict(
                                             orient='records')[0]}),
                                       index=[0])
                daily_node_features = pd.concat([daily_node_features, new_row], ignore_index=True)

                daily_node_features = daily_node_features.drop_duplicates(subset=['nodeID'])

            # creat the TDA for each day
            try:

                Xfilt = daily_node_features
                Xfilt = Xfilt.drop(columns=['nodeID'])
                mapper = km.KeplerMapper()
                scaler = MinMaxScaler(feature_range=(0, 1))

                Xfilt = scaler.fit_transform(Xfilt)
                lens = mapper.fit_transform(Xfilt, projection=sklearn.manifold.TSNE())
                # We use cls=5, but this parameter can be further refined.  Its impact on results seems minimal.

                # # Create a multiprocessing Queue to store the results
                # result_queue = multiprocessing.Queue()
                #
                # # Create a list to store the processes
                # processes = []

                # Create a multiprocessing Pool with the desired number of processes
                with multiprocessing.Pool() as pool:
                    # List to store the result objects
                    results = []

                    # Iterate over the combinations and apply the process_combination function to each combination
                    # for per_overlap_indx in range(1, 11):
                    #     for n_cubes in range(2, 11):
                    per_overlap = 0.05
                    n_cubes = 10
                    cls = 5
                    for albation_index in [1]:
                        # per_overlap = round(per_overlap_indx * 0.05, 2)
                        result = pool.apply_async(self.TDA_process,
                                                  (mapper, lens, Xfilt, per_overlap, n_cubes, cls))
                        results.append(result)

                    # Retrieve the results as they become available
                    for result in results:
                        dailyFeatures = result.get()
                        timeWindowSequence.append(dailyFeatures)

                # for per_overlap in [0.1]:
                #     for n_cubes in [1, 2, 3, 4, 5, 6]:
                #         for cls in [1, 2, 3, 4, 5]:
                #             # print("Processing overlap={} , n_cubes={} , cls={}".format(per_overlap, n_cubes, cls))
                #
                #
                #             process = multiprocessing.Process(target=self.tda_function_wrapper,
                #                                               args=(mapper, lens, Xfilt, per_overlap, n_cubes, cls,
                #                                                     result_queue))
                #             process.start()
                #             processes.append(process)

                # dailyTdaGraph = mapper.map(
                #     lens,
                #     Xfilt,
                #     clusterer=sklearn.cluster.KMeans(n_clusters=cls, random_state=1618033),
                #     cover=km.Cover(n_cubes=n_cubes, perc_overlap=per_overlap))  # 0.2 0.4
                #
                # # mapper.visualize(dailyTdaGraph,
                # #                  path_html= "test-shape.HTML",
                # #                  title="Mapper graph for network")
                #
                # # extract the cluster size and cluster ID vector out of that
                #
                # numberOfNodes = len(dailyTdaGraph['nodes'])
                # numberOfEdges = len(dailyTdaGraph['links'])
                # try:
                #     maxClusterSize = len(
                #         dailyTdaGraph["nodes"][
                #             max(dailyTdaGraph["nodes"], key=lambda k: len(dailyTdaGraph["nodes"][k]))])
                # except Exception as e:
                #     maxClusterSize = 0
                #
                # dailyFeatures = {"overlap{}-cube{}-cls{}".format(per_overlap, n_cubes, cls): [numberOfNodes,
                #                                                                               numberOfEdges,
                #                                                                               maxClusterSize]}
                # timeWindowSequence.append(dailyFeatures)

                # Wait for all processes to finish
                # for process in processes:
                #     process.join()
                #
                #     # Collect the results from the queue
                # timeWindowSequence = []
                # while not result_queue.empty():
                #     result = result_queue.get()
                #     # print("RESUUUUULT --- > \n")
                #     # print(result)
                #     timeWindowSequence.append(result)








            except Exception as e:
                print(str(e))
            start_date = start_date + dt.timedelta(days=1)
            processingDay += 1

        # the graph has been repopulated with daily temporal features
        merged_dict = self.merge_dicts(timeWindowSequence)
        return merged_dict

    def process_raw_extracted_rnn_sequence(self, timeFrameData, nodeFeatures):

        # break the data to daily graphs
        timeWindowSequence = list()
        data_first_date = timeFrameData['date'].min()
        data_last_date = timeFrameData['date'].max()
        numberOfDays = (data_last_date - data_first_date).days
        start_date = data_first_date
        # initiate the graph
        processingDay = 0
        while (processingDay <= numberOfDays):
            # print("Processing TDA RNN sequential Extraction day {}".format(processingDay))
            daily_end_date = start_date + dt.timedelta(days=1)
            selectedDailyNetwork = timeFrameData[
                (timeFrameData['date'] >= start_date) & (timeFrameData['date'] < daily_end_date)]
            try:
                number_of_nodes = pd.concat([selectedDailyNetwork['from'], selectedDailyNetwork['to']]).count()
                number_of_edges = len(selectedDailyNetwork)
                avg_value = selectedDailyNetwork["value"].mean()
                dailyFeatures = {"raw": [number_of_nodes, number_of_edges, avg_value]}
                timeWindowSequence.append(dailyFeatures)
            except Exception as e:
                print(str(e))
            start_date = start_date + dt.timedelta(days=1)
            processingDay += 1

        merged_dict = self.merge_dicts(timeWindowSequence)
        return merged_dict

    def TDA_process(self, mapper, lens, Xfilt, per_overlap, n_cubes, cls):
        dailyTdaGraph = mapper.map(
            lens,
            Xfilt,
            clusterer=sklearn.cluster.KMeans(n_clusters=cls, random_state=1618033),
            cover=km.Cover(n_cubes=n_cubes, perc_overlap=per_overlap))  # 0.2 0.4

        # Visualizing the mapper
        # mapper.visualize(dailyTdaGraph,
        #                  path_html= "Adex_first_7_days.HTML",
        #                  title="Mapper graph for network")

        # extract the cluster size and cluster ID vector out of that

        maxClusterSize = 0
        average_cluster_size = 0
        average_edge_weight = 0
        # Number of nodes
        numberOfNodes = len(dailyTdaGraph['nodes'])
        # Number of edges
        numberOfEdges = sum(len(edges) for edges in dailyTdaGraph['links'].values())
        try:
            # max cluster size
            maxClusterSize = len(
                dailyTdaGraph["nodes"][
                    max(dailyTdaGraph["nodes"], key=lambda k: len(dailyTdaGraph["nodes"][k]))])

            # average clsuter size
            cluster_sizes = [len(nodes) for nodes in dailyTdaGraph["nodes"].values()]
            average_cluster_size = sum(cluster_sizes) / len(cluster_sizes)

            # average edge weight
            edge_weights = defaultdict(dict)
            for source_node, target_nodes in dailyTdaGraph['links'].items():
                for target_node in target_nodes:
                    common_indexes = len(
                        set(dailyTdaGraph['nodes'][source_node]) & set(dailyTdaGraph['nodes'][target_node]))
                    edge_weights[source_node][target_node] = common_indexes
                    total_edge_weights = sum(
                        weight for target_weights in edge_weights.values() for weight in target_weights.values())
                    total_edges = sum(len(target_weights) for target_weights in edge_weights.values())
                    average_edge_weight = total_edge_weights / total_edges


        except Exception as e:
            maxClusterSize = 0
            average_cluster_size = 0
            average_edge_weight = 0



        dailyFeatures = {"overlap{}-cube{}-cls{}".format(per_overlap, n_cubes, cls): [numberOfNodes,
                                                                                      numberOfEdges,
                                                                                      maxClusterSize,
                                                                                      average_cluster_size,
                                                                                      average_edge_weight]}
        return dailyFeatures

    def tda_function_wrapper(self, mapper, lens, Xfilt, per_overlap, n_cubes, cls, result_queue):
        result = self.TDA_process(mapper, lens, Xfilt, per_overlap, n_cubes, cls)
        result_queue.put(result)

    def merge_dicts(self, list_of_dicts):
        merged_dict = {}
        for dictionary in list_of_dicts:
            for key, value in dictionary.items():
                if key in merged_dict:
                    merged_dict[key].append(value)
                else:
                    merged_dict[key] = [value]
        return merged_dict

    def graph_creation_handler(self):
        files = os.listdir(self.file_path)
        for file in files:
            if file.endswith(".txt"):
                self.processingIndx += 1
                print("Processing {} / {} \n".format(self.processingIndx, len(files) - 3))
                p = Process(target=self.process_raw_extracted_rnn_sequence, args=(file,))  # make process
                p.start()  # start function
                p.join(timeout=240)

                # Check if the process is still running
                if p.is_alive():
                    # The process is still running, terminate it
                    p.terminate()
                    print("The file is taking infinite time - check the file ")
                    shutil.move(self.file_path + file, self.file_path + "issue/" + file)
                    self.processingIndx += 1
                    print("Function timed out and was terminated")
                else:
                    # The process has finished
                    self.processingIndx += 1
                    shutil.move(self.file_path + file, self.file_path + "Processed/" + file)
                    print("Process finished successfully")
                    p.terminate()

        # stat_data.to_csv("final_data.csv")

    def graph_creation_handler_time_series(self):
        files = os.listdir(self.timeseries_file_path)
        # ToDO : add a if clause for tsv for reddit_B
        for file in files:
            if file.endswith(".txt"):
                print("Processing {} / {} \n".format(self.processingIndx, len(files) - 4))
                p = Process(target=self.create_time_series_for_other_dataset_rnn_sequence, args=(file,))  # make process
                p.start()  # start function
                p.join(timeout=68000)

                # Check if the process is still running
                if p.is_alive():
                    # The process is still running, terminate it
                    p.terminate()
                    print("The file is taking infinite time - check the file ")
                    shutil.move(self.timeseries_file_path + file, self.timeseries_file_path + "issue/" + file)
                    self.processingIndx += 1
                    print("Function timed out and was terminated")
                else:
                    # The process has finished
                    self.processingIndx += 1
                    shutil.move(self.timeseries_file_path + file, self.timeseries_file_path + "Processed/" + file)
                    print("Process finished successfully")
                    p.terminate()

        # stat_data.to_csv("final_data.csv")

if __name__ == '__main__':
    np = NetworkParser()
    np.graph_creation_handler_time_series()

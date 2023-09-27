import os
import pickle
import shutil
import pandas as pd
import networkx as nx
import datetime as dt
import kmapper as km
import sklearn
from sklearn.preprocessing import MinMaxScaler

def creatTimeSeriesGraphs(self, file):
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
        transactionGraph = self.process_TDA_extracted_temporal_features(selectedNetworkInGraphDataWindow,
                                                                        transactionGraph, node_features)

        # Generating TDA graphs
        createTDAGraph(node_features, label, directory, network=file, timeWindow=indx)

        featureNames = ["outgoing_edge_weight_sum", "incoming_edge_weight_sum", "outgoing_edge_count",
                        "incoming_edge_count", "dailyClusterID", "dailyClusterSize"]
        window_start_date = window_start_date + dt.timedelta(days=1)

        # Generating PyGraphs for timeseries data
        if not os.path.exists(directory):
            os.makedirs(directory)
        pygData = from_networkx(transactionGraph, label=label, group_node_attrs=featureNames)
        with open(directory + "/TemporalVectorizedGraph_Tuned/" + file + "_" + "graph_" + str(indx), 'wb') as f:
            pickle.dump(pygData, f)

def createTDAGraph(self, data, label, timeWindow=0, network=""):
    try:
        per_overlap = 0.5
        n_cubes = 5
        cls = 5
        Xfilt = data
        Xfilt = Xfilt.drop(columns=['nodeID'])
        mapper = km.KeplerMapper()
        scaler = MinMaxScaler(feature_range=(0, 1))

        Xfilt = scaler.fit_transform(Xfilt)
        lens = mapper.fit_transform(Xfilt, projection=sklearn.manifold.TSNE())

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
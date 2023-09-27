# config.py

import pandas as pd

# Path of the dataset folder
file_path = "../path/to/all_network/"
timeseries_file_path = "../path/to/all_network/TimeSeries/"
timeseries_file_path_other = "../path/to/all_network/TimeSeries/Other/"
time_window = [7]

# Validation duration condition
network_validation_duration = 20
final_data_duration = 5
label_treshhold_percentage = 10

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
processing_index = 1


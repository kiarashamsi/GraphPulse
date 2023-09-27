import ast
import csv
import os
import pickle
import re
import time
from random import random
from keras.callbacks import Callback
import pandas as pd
import datetime as dt
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, GRU
from matplotlib import pyplot as plt, ticker
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
# import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Dense
import seaborn as sns


def read_seq_data(network):
    file_path = "Sequence/{}/".format(network)
    file = "seq.txt"
    seqData = dict()
    with open(file_path + file, 'rb') as f:
        # print("\n Reading Torch Data {} / {}".format(inx, len(files)))
        seqData = pickle.load(f)
    return seqData


def read_seq_data_by_file_name(network, file):
    file_path = "Sequence/{}/".format(network)
    seqData = dict()
    with open(file_path + file, 'rb') as f:
        # print("\n Reading Torch Data {} / {}".format(inx, len(files)))
        seqData = pickle.load(f)
    return seqData


def train_test_split_sequential(*arrays, train_size=None):
    if train_size is None:
        raise ValueError("train_size must be specified.")

    total_samples = len(arrays[0])
    train_samples = int(train_size * total_samples)
    train_data = [array[:train_samples] for array in arrays]
    test_data = [array[train_samples:] for array in arrays]

    return train_data[0], test_data[0]


def reset_random_seeds():
    os.environ['PYTHONHASHSEED'] = str(1)
    tf.random.set_seed(1)
    np.random.seed(1)


class AUCCallback(Callback):
    def __init__(self, validation_data):
        self.validation_data = validation_data
        self.auc_scores = []

    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.validation_data
        y_pred = self.model.predict(x_val)
        auc_score = roc_auc_score(y_val, y_pred)
        self.auc_scores.append(auc_score)
        print(f"Epoch {epoch + 1} - Validation AUC: {auc_score:.4f}")

    def get_auc_std(self):
        return np.std(self.auc_scores)

    def get_auc_avg(self):
        return np.average(self.auc_scores)


def LSTM_classifier(data, labels, spec, network):
    reset_random_seeds()
    # Set random seed for NumPy
    np.random.seed(42)

    # Set random seed for TensorFlow
    tf.random.set_seed(42)
    start_Rnn_training_time = time.time()
    # data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    data_train, data_test = train_test_split_sequential(data, train_size=0.8)
    labels_train, labels_test = train_test_split_sequential(labels, train_size=0.8)

    # Define the LSTM model
    model_LSTM = Sequential()
    if ("Combined" in spec):
        if ("TDA3" in spec):
            model_LSTM.add(LSTM(64, input_shape=(7, 6), return_sequences=True))  # Adjust the number of units as needed
        else:
            model_LSTM.add(LSTM(64, input_shape=(7, 8), return_sequences=True))  # Adjust the number of units as needed
    elif ("TDA5" in spec):
        if ("ablation" in spec) :
            model_LSTM.add(LSTM(64, input_shape=(7, 4), return_sequences=True))  # Adjust the number of units as needed
        else:
            model_LSTM.add(LSTM(64, input_shape=(7, 5), return_sequences=True))  # Adjust the number of units as needed

    else:
        model_LSTM.add(LSTM(64, input_shape=(7, 3), return_sequences=True))  # Adjust the number of units as needed

    # model_LSTM.add(LSTM(64, activation='relu', input_shape=(45, 1), return_sequences=True))
    model_LSTM.add(LSTM(32, activation='relu', return_sequences=True))
    model_LSTM.add(GRU(32, activation='relu', return_sequences=True))
    model_LSTM.add(GRU(32, activation='relu', return_sequences=False))
    model_LSTM.add((Dense(100, activation='relu')))
    model_LSTM.add(Dense(1, activation="sigmoid"))

    from tensorflow.python.keras.optimizers import adam_v2
    # Iconomi 0.00001
    # Centra 0.00004
    # bancor 0.00005
    # bancor_5_feature = 0.00003
    # Aragon 0.00001 - tda_5: 0.000025
    # UCI 0.00008 (0.00004 - epoch 250 )
    # Reddit-b 0.00005
    # Aternity 0.00008
    # Adex TDA5 = 0.0001

    learning_rate = 0.0001

    opt = adam_v2.Adam(learning_rate=learning_rate)

    # Compile the model
    model_LSTM.compile(loss='binary_crossentropy', optimizer=opt, metrics=['AUC', 'accuracy'])

    # Train the model
    # Create the AUCCallback and specify the validation data
    auc_callback = AUCCallback(validation_data=(data_test, labels_test))
    model_LSTM.fit(data_train, labels_train, epochs=100, validation_data=(data_test, labels_test),
                   callbacks=[auc_callback])  # Adjust the epochs and batch_size as needed

    # Make predictions on the test set

    start_Rnn_training_time = time.time() - start_Rnn_training_time
    y_pred_LSTM = model_LSTM.predict(data_test)
    roc_LSTM = roc_auc_score(labels_test, y_pred_LSTM)

    # Evaluate the model on the test set
    loss, auc, accuracy = model_LSTM.evaluate(data_test, labels_test)
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")
    print("AUC: {}".format(auc))
    print(f"AUC-test : {roc_LSTM}")
    print(f"STD : {auc_callback.get_auc_std()}")
    print(f"AVG AUC : {auc_callback.get_auc_avg()}")

    try:
        # Attempt to open the file in 'append' mode
        with open("RnnResults/RNN-Results.txt", 'a') as file:
            # Append a line to the existing file
            file.write(
                "{},{},{},{},{},{},{},{},{},{},{}".format(network, spec, loss, accuracy, auc, roc_LSTM,
                                                          auc_callback.get_auc_avg(), auc_callback.get_auc_std(),
                                                          start_Rnn_training_time,
                                                          len(data), learning_rate) + '\n')
    except FileNotFoundError:
        # File doesn't exist, so create a new file and write text
        with open("RnnResults/RNN-Results.txt", 'w') as file:
            file.write(
                "Network={} Spec={} Loss={} Accuracy={} AUC={} time={} data={}".format(network, spec, loss, accuracy,
                                                                                       auc,
                                                                                       start_Rnn_training_time,
                                                                                       len(data)) + '\n')
    return max(auc, roc_LSTM)


def merge_dicts_old(list_of_dicts):
    merged_dict = {}
    for dictionary in list_of_dicts:
        for key, value in dictionary.items():
            if key in merged_dict:
                merged_dict[key].append(value)
            else:
                merged_dict[key] = [value]
    return merged_dict


def merge_dicts(list_of_dicts):
    merged_dict = {}
    outliers = []
    len_features_valid = 5
    outlier_flag = False
    indx = 0
    for dictionary in list_of_dicts:
        try:
            for key, value in dictionary.items():
                outlier_flag = False
                if (type(value) == list):
                    for featues in value:
                        if (len(featues) != len_features_valid):
                            outliers.append(indx)
                            outlier_flag = True
                            continue

                    if (not outlier_flag):
                        if key in merged_dict:
                            merged_dict[key].append(value)
                        else:
                            merged_dict[key] = [value]
                else:
                    outliers.append(indx)
        except Exception as e:
            print(e)
            outliers.append(indx)
            indx += 1
            continue
        indx += 1
    return merged_dict, outliers


def outputCleaner():
    input_file = "RnnResults/AternityTest.txt"
    output_file = "RnnResults/Aternity_RNN_Results_cleaned_v2.csv"

    with open(input_file, "r") as file:
        lines = file.readlines()

    data = []
    for line in lines:
        values = line.strip().split()
        spec_values = values[1].split("-")
        row = [value.split("=")[1] for value in values[:1]] + spec_values + [value.split("=")[1] for value in
                                                                             values[2:]]
        data.append(row)

    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)
        file.close()

    with open(output_file, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            new_row = []
            for i, value in enumerate(row):
                if i in (1, 2, 3):
                    numeric_part = re.findall(r"[-+]?\d*\.?\d+", value)
                    if numeric_part:
                        new_row.append(numeric_part[0])
                    else:
                        new_row.append("")
                else:
                    new_row.append(value)
            data.append(new_row)

    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print("Conversion complete. CSV file created.")


def visualize_time_exp():
    # Specify the file path of the CSV
    file_path = "TDA_time_exp_RNN.csv"

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    x = df["Snap"]
    y = df["Time"]
    networkName = df["Net"]

    # Calculate the regression line
    regression_line = np.polyfit(x, y, 1)
    regression_y = np.polyval(regression_line, x)

    # Add labels to each point
    for i in range(len(x)):
        plt.text(x[i], y[i] - 0.5, f'{networkName[i]}', ha='center', va='bottom', fontsize=7)

    # Create the scatter plot
    plt.plot(x, y, label='Transaction Networks', marker='o')
    # plt.plot(x, regression_y, color='red', label='Regression Line')

    # Set the y-axis to logarithmic scale
    # plt.yscale('log')

    # Set axis labels and title
    plt.xlabel('# Snapshots')
    plt.ylabel('Time (Sec)')
    plt.title('RNN training costs of token networks')

    # Display the plot
    plt.legend()
    plt.tight_layout()
    plt.savefig('tda_time_exp_Rnn_plot.png', dpi=300)
    plt.show()

    # Display the DataFrame
    print(df)


def visualize_time_exp_bar():
    # Specify the file path of the CSV
    file_path = "TDA_time_exp_RNN.csv"

    # Load the data from the CSV file
    df = pd.read_csv(file_path)

    # Sort the DataFrame by time from high to low
    df = df.sort_values(by='Time', ascending=False)

    # Extract data for plotting
    x = df["Net"]
    y = df["Time"]

    plt.figure(figsize=(10, 6))  # Set the figure size

    # Create the bar plot
    plt.bar(x, y)

    # Customizing the plot
    plt.xlabel('Networks', fontsize=10)  # Set the x-axis label font size to 10
    plt.ylabel('Time (sec)')
    plt.title('RNN training cost')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Set y-axis ticks in multiples of 1
    plt.yticks(range(0, int(max(y)) + 1, 1))

    # Adding rounded data labels on top of each bar
    for i, v in enumerate(y):
        plt.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')

    # Displaying the plot
    plt.tight_layout()
    plt.savefig('tda_time_exp_Rnn_bar_f.png', dpi=300)
    plt.show()


def visualize_time_exp_bar_method():
    # Specify the file path of the CSV
    file_path = "Method_time_exp.csv"

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    x = df["Method"]
    y = df["Time"]

    # Set the seaborn style to remove the border and grid
    sns.set_style('ticks')

    # Create the bar plot
    plt.figure(figsize=(10, 8))  # Set the figure size
    bars = plt.bar(x, y)

    # Set the x-axis font size and rotation
    plt.xticks(fontsize=12, rotation=45, ha='right')

    # Show the value on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(height), ha='center', va='bottom', fontsize=10)

    # Customizing the plot
    plt.xlabel('Method', fontsize=14)
    plt.ylabel('Time (sec)', fontsize=14)
    plt.title('Training cost for Dgd network', fontsize=16)

    # Remove the top and right spines
    sns.despine()

    # Displaying the plot
    plt.tight_layout()  # Adjusts the layout to avoid cutting off labels
    plt.savefig('method_time_exp_bar_2.png', dpi=300)
    plt.show()


def visualize_time_exp_scatter():
    # Specify the file path of the CSV
    file_path = "TDA_time_exp.csv"

    # Read the CSV file into a pandas DataFrame

    df = pd.read_csv(file_path)
    # Filter the rows where the number of nodes is less than 250
    df_filtered = df[df["Node"] >= 250]
    x = df_filtered["Node"]
    y = df_filtered["Time"]

    # Calculate the regression line
    regression_line = np.polyfit(x, y, 1)
    regression_y = np.polyval(regression_line, x)

    # Add labels to each point

    # Create the scatter plot
    plt.scatter(x, y, label='Transaction Networks', s=6)
    plt.plot(x, regression_y, color='red', label='Regression Line')

    # Set the y-axis to logarithmic scale
    plt.yscale('log')
    # # Set specific values to show on the y-axis
    y_ticks = [10, 20, 30, 40, 50, 100]
    plt.yticks(y_ticks)
    # Set axis labels and title
    plt.xlabel('# Nodes')
    plt.ylabel('Time (Sec)')
    plt.title('TDA costs of daily token networks')

    # Remove the top and right spines
    sns.despine()



    # Display the plot
    plt.legend()
    plt.tight_layout()
    plt.savefig('tda_time_exp_plot_r_f-log_2.png', dpi=300)
    plt.show()

    # Display the DataFrame
    print(df)


def visualize_labels(file):
    labels_file_path = "../data/all_network/TimeSeries/Baseline/labels/"
    with open(labels_file_path + file, "r") as file_r:
        lines = file_r.readlines()

    # Convert the list of strings to a list of floats
    data = [float(line.strip()) for line in lines]
    start_date = dt.datetime(2015, 8, 19)
    end_date = start_date + dt.timedelta(days=len(data) - 1)
    dates = [start_date + dt.timedelta(days=i) for i in range(len(data))]

    # V1 without colosr
    plt.figure(figsize=(10, 6))
    plt.plot(dates, data, linestyle='-', color='blue')  # Set the line color to blue

    plt.xlabel("Time")
    plt.ylabel("Label")
    plt.title("Label distribution plot for " + file.split("_")[0].capitalize() + " network")
    plt.grid(False)
    plt.xticks(rotation=45)

    # Set y-axis ticks to [0, 1] only and reduce space between 0 and 1
    plt.yticks([0, 1], fontsize=12)

    # Customize the colors of data points
    colors = ['red' if value == 0 else 'blue' for value in data]

    plt.tight_layout()
    plt.savefig(labels_file_path + file + '_label_graph_line_wo_color.png', dpi=300)
    plt.show()

    # V2 with colors
    plt.figure(figsize=(10, 6))
    plt.plot(dates, data, marker=None, linestyle='-')  # Removed the marker 'o'
    plt.xlabel("Time")
    plt.ylabel("Label")
    plt.title("Label distribution plot for " + file.split("_")[0].capitalize() + " network")
    plt.grid(False)  # Removed the grid lines
    plt.xticks(rotation=45)

    # Set y-axis ticks to [0, 1] only and reduce space between 0 and 1
    plt.yticks([0, 1], fontsize=12)

    # Customize the colors of data points
    colors = ['red' if value == 0 else 'blue' for value in data]
    plt.plot(dates, data, color='blue', linestyle='-')  # Plot the blue line
    plt.scatter(dates, data, color=colors)  # Use scatter to show red and blue data points

    plt.tight_layout()
    plt.savefig(labels_file_path + file + '_label_graph_line_with_color.png', dpi=300)
    plt.show()


def getDailyAvg(file):
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


def getDailyAvgReddit(file):
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


def write_list_to_csv(filename, data_list):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for item in data_list:
            writer.writerow([item])


def read_dicts_from_files(directory_path):
    dict_list = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, "rb") as file:
                dictionary = pickle.load(file)
                dict_list.append(dictionary)

    with open(directory_path + "seq_tda_ablation.txt", "wb") as output_file:
        pickle.dump(merge_dicts_old(dict_list), output_file)
    return merge_dicts_old(dict_list)


def merge_dicts_with_same_keys(dict_list):
    merged_dict = {}
    for d in dict_list:
        try:
            for key, value in d.items():
                if key in merged_dict:
                    # If the key already exists, append the value to the list
                    if isinstance(merged_dict[key], list):
                        merged_dict[key].append(value)
                    else:
                        merged_dict[key] = [merged_dict[key], value]
                else:
                    merged_dict[key] = value
        except Exception as e:
            print(d)
            print(e)
            continue

    return merged_dict


def find_indices_of_items_with_incorrect_size(list_3d):
    incorrect_indices = []
    for i, sublist_2d in enumerate(list_3d):
        for j, sublist in enumerate(sublist_2d):
            if len(sublist) != 5:
                incorrect_indices.append((i, j))
    return incorrect_indices


def filter_valid_sublists(list_3d):
    valid_sublists = [
        sublist
        for sublist_2d in list_3d
        for sublist in sublist_2d
        if len(sublist) == 5
    ]
    return [valid_sublists[i:i + 2] for i in range(0, len(valid_sublists), 2)]


def save_fixed_seq(data, path):
    data["sequence"], outliers = merge_dicts(data["sequence"])
    np_labels = np.array(data["label"])
    np_labels = np.delete(np_labels, outliers)
    data["label"] = np_labels.tolist()
    with open(path + "seq_tda_ablation.txt", "wb") as output_file:
        pickle.dump(data, output_file)


if __name__ == "__main__":
    # read_dicts_from_files("Sequence/networkcindicator.txtl/")
    visualize_time_exp_bar_method()
    exit(0)
    # visualize_time_exp_bar_method()
    # visualize_time_exp_bar()
    # visualize_labels("mathoverflow_Label.csv")
    # # for generating daily stats
    # # processingIndx = 0
    # # timeseries_file_path_other = "../data/all_network/TimeSeries/Other/"
    # # files = os.listdir(timeseries_file_path_other)
    # # for file in files:
    # #     if file.endswith(".txt"):
    # #         print("Processing {} / {} \n".format(processingIndx, len(files) - 4))
    # #         getDailyAvg(file)
    # #         processingIndx += 1
    #
    # # visualize_time_exp_bar()
    # # #outputCleaner()
    # # # "networkaeternity.txt", "networkaion.txt", "networkaragon.txt", "networkbancor.txt", "networkcentra.txt", "networkcindicator.txt", "networkcoindash.txt" , "networkiconomi.txt", "networkadex.txt"
    # # # "networkdgd.txt","networkcentra.txt","networkcindicator.txt"
    #
    # # Iconomi 0.00001
    # # Centra 0.00004
    # # bancor 0.00005
    # # Aragon 0.00001
    # # UCI 0.00008 (0.00004 - epoch 250 )
    # # Reddit-b 0.00005
    # # Aternity 0.00008
    #
    networkList = ["networkbancor.txt"]
    normalizer = "all"
    # # networkList = ["mathoverflow.txt", "networkcoindash.txt", "networkiconomi.txt", "networkadex.txt", "networkdgd.txt",
    # #                "networkbancor.txt", "networkcentra.txt", "networkcindicator.txt", "networkaeternity.txt",
    # #                "networkaion.txt", "networkaragon.txt", "CollegeMsg.txt", "Reddit_B.tsv"]
    # # # tdaDifferentGraph = ["Overlap_0.1_Ncube_2", "Overlap_0.1_Ncube_5", "Overlap_0.2_Ncube_2", "Overlap_0.2_Ncube_5", "Overlap_0.3_Ncube_2", "Overlap_0.3_Ncube_5", "Overlap_0.5_Ncube_2", "Overlap_0.5_Ncube_5", "Overlap_0.6_Ncube_2", "Overlap_0.6_Ncube_5"]

    # for network in networkList:
    #     # for tdaVariable in tdaDifferentGraph:
    #     print("Working on {}\n".format(network))
    #     data = read_seq_data_by_file_name(network, "seq_tda_ablation.txt")
    #     labels = data["label"]
    #
    #
    #
    #     # write_list_to_csv(network.split(".")[0]+"_Label.csv", labels)
    #     # print(labels)
    #
    #     # files = os.listdir(f"Sequence/{network}/dailySeq/")
    #     # data = []
    #     # result = {}
    #     # for file in files:
    #     #     if file.endswith(".txt"):
    #     #         data.append(read_seq_data_by_file_name(network,  f"dailySeq/{file}"))
    #
    #     # data = merge_dicts(data)
    #     # for dictionary in data["sequence"]:
    #     #     for key, value in dictionary.items():
    #     #         if key not in result:
    #     #             result[key] = []
    #     #         result[key].append(value)
    #     #
    #     # data["sequence"] = result
    #
    #     for key, value in data["sequence"].items():
    #         if ("overlap" in key):
    #             print("Processing network ({}) - with parameters {}".format(network, key))
    #             np_labels = np.array(data["label"])
    #             # if (len(value[0]) != 7):
    #             #     while (len(value[0]) != 7):
    #             #         del value[0]
    #             #         np_labels = np.delete(np_labels, 0, axis=0)
    #             indxs = []
    #             if (network == "networkaion.txt"):
    #                 for i in range(0, len(value)):
    #                     if len(value[i]) != 7:
    #                         indxs.append(i)
    #             if (network == "networkcentra.txt"):
    #                 for i in range(0, len(value)):
    #                     if len(value[i]) != 7:
    #                         indxs.append(i)
    #
    #                 value = [item for index, item in enumerate(value) if index not in indxs]
    #                 np_labels = np.delete(np_labels, indxs)
    #
    #
    #             np_data = np.array(value)
    #
    #             # for correcting the ablation study array
    #             unique_arrays = []
    #             unique_indices = set()
    #
    #             # Iterate through arrays along the 42 dimension
    #             for idx in range(np_data.shape[1]):
    #                 current_array = tuple(np_data[:, idx, :].flatten())
    #                 if current_array not in unique_indices:
    #                     unique_arrays.append(np_data[:, idx, :])
    #                     unique_indices.add(current_array)
    #                     if len(unique_arrays) == 7:
    #                         break
    #
    #             # Convert the list of unique arrays into a NumPy array
    #             unique_arrays = np.array(unique_arrays)
    #
    #             # Transpose the unique arrays to match the desired shape (m, 7, 5)
    #             unique_arrays = np.transpose(unique_arrays, axes=(1, 0, 2))
    #             np_data = unique_arrays
    #             # ablation_data = np.delete(np_data, 3, axis=2)
    #             # ablation_data = np.delete(ablation_data, 3, axis=2)
    #             for i in range(0, 5):
    #                 LSTM_classifier(np_data, np_labels, key, network)
    #
    #         #ablation
    #         # for i in range(0, 5):
    #         #     ablation_data = np.delete(np_data, i, axis=2)
    #         #     print("removing feature {}".format(i))
    #         #     for t in range(0, 1):
    #         #         print("IT {} \n".format(t))
    #         #         LSTM_classifier(ablation_data, np_labels, key, network)
    # #
    # # for seq files
    auc_scores = []

    #
    # # for combined version
    for network in networkList:
        for run in range(1, 6):
            print(f"RUN {run}")
            # for tdaVariable in tdaDifferentGraph:
            print("Working on {}\n".format(network))
            data = read_seq_data_by_file_name(network, "seq_tda_ablation.txt")

            # data["sequence"], outliers = merge_dicts(data["sequence"])
            # np_labels = np.array(data["label"])
            # np_labels = np.delete(np_labels, outliers)
            # data["label"] = np_labels.tolist()
            # with open("Sequence/" + network + "/seq_tda_ablation.txt", "wb") as output_file:
            #     pickle.dump(data, output_file)
            data_raw = read_seq_data_by_file_name(network, "seq_raw.txt")

            indxs = []
            np_data = []
            np_data_raw = []
            np_labels = []

            # indx = 0
            for key, value in data["sequence"].items():
                # if (indx == 1):
                #     break

                if ("overlap" in key):
                    print("Processing network ({}) - with parameters {}".format(network, key))
                    np_labels = np.array(data["label"])
                    # if (len(value[0]) != 7):
                    #     while (len(value[0]) != 7):
                    #         del value[0]
                    #         np_labels = np.delete(np_labels, 0, axis=0)
                    # indxs = [163, 164, 165, 166, 167, 168, 169, 170]
                    #
                    if (network == "networkdgd.txt"):
                        for i in range(0, len(value)):
                            if len(value[i]) != 7:
                                indxs.append(i)
                    if (network == "networkcentra.txt"):
                        for i in range(0, len(value)):
                            if len(value[i]) != 7:
                                indxs.append(i)

                    value = [item for index, item in enumerate(value) if index not in indxs]
                    print(find_indices_of_items_with_incorrect_size(value))
                    np_labels = np.delete(np_labels, indxs)

                    print(find_indices_of_items_with_incorrect_size(value))
                    np_data = np.array(value)
                    # auc_scores.append(LSTM_classifier(np_data, np_labels, key, network))

            # for raw data conbination
            for key_raw, value_raw in data_raw["sequence"].items():
                # if (indx == 1):
                #     break

                if ("raw" in key_raw):
                    print("Processing network ({}) - with parameters {}".format(network, key_raw))
                    np_labels_raw = np.array(data["label"])
                    # if (len(value_raw[0]) != 7):
                    #     while (len(value_raw[0]) != 7):
                    #         del value_raw[0]
                    #         np_labels_raw = np.delete(np_labels_raw, 0, axis=0)
                    # indxs = [163, 164, 165, 166, 167, 168, 169, 170]

                    if (network == "networkdgd.txt"):
                        indxs = [163, 164, 165, 166, 167, 168]
                        for i in range(0, len(value_raw)):
                            if len(value_raw[i]) != 7:
                                indxs.append(i)

                    if (network == "networkcindicator.txt"):
                        indxs = [0, 1, 2]
                        for i in range(0, len(value_raw)):
                            if len(value_raw[i]) != 7:
                                indxs.append(i)

                    if (network == "networkcentra.txt"):
                        for i in range(0, len(value_raw)):
                            if len(value_raw[i]) != 7:
                                indxs.append(i)

                    value_raw = [item for index, item in enumerate(value_raw) if index not in indxs]
                    np_labels_raw = np.delete(np_labels_raw, indxs)
                    value = np.delete(value_raw, indxs)
                    np_data_raw = np.array(value_raw)
                    # auc_scores.append(LSTM_classifier(np_data_raw, np_labels, key_raw, network))

            # for correcting the ablation study array
            unique_arrays = []
            unique_indices = set()

            # # Iterate through arrays along the 42 dimension
            # for idx in range(np_data.shape[1]):
            #     current_array = tuple(np_data[:, idx, :].flatten())
            #     if current_array not in unique_indices:
            #         unique_arrays.append(np_data[:, idx, :])
            #         unique_indices.add(current_array)
            #         if len(unique_arrays) == 7:
            #             break
            #
            # min_max_normalization = True
            # unique_arrays = np.transpose(unique_arrays, axes=(1, 0, 2))
            # unique_arrays = np.array(unique_arrays)
            # np_data = unique_arrays

            if (normalizer == "per_column"):
                min_values = np.min(np_data, axis=(0, 1))
                max_values = np.max(np_data, axis=(0, 1))

                # Normalize the entire array using Min-Max normalization formula

                normalized_data_arr = (np_data - min_values) / (max_values - min_values)

                normalized_data_arr = np.nan_to_num(normalized_data_arr, nan=0)

                # min_value = np_data_raw.min()
                # max_value = np_data_raw.max()

                min_values = np.min(np_data_raw, axis=(0, 1))
                max_values = np.max(np_data_raw, axis=(0, 1))

                # Normalize the entire array using Min-Max normalization formula
                normalized_raw_data_arr = (np_data_raw - min_values) / (max_values - min_values)
                normalized_raw_data_arr = np.nan_to_num(normalized_raw_data_arr, nan=0)
            else:
                # normalize for all the data
                min_values = np.min(np_data)
                max_values = np.max(np_data)

                # Normalize the entire array using Min-Max normalization formula

                normalized_data_arr = (np_data - min_values) / (max_values - min_values)

                normalized_data_arr = np.nan_to_num(normalized_data_arr, nan=0)

                # min_value = np_data_raw.min()
                # max_value = np_data_raw.max()

                min_values = np.min(np_data_raw)
                max_values = np.max(np_data_raw)

                # Normalize the entire array using Min-Max normalization formula
                normalized_raw_data_arr = (np_data_raw - min_values) / (max_values - min_values)
                normalized_raw_data_arr = np.nan_to_num(normalized_raw_data_arr, nan=0)

            # min_value = np_data.min()
            # max_value = np_data.max()

            # Concatenate the two arrays along the third axis (axis=2)
            concatenated_arr_normalized = np.concatenate((normalized_data_arr, normalized_raw_data_arr), axis=2)

            concatenated_arr = np.concatenate((np_data, np_data_raw), axis=2)

            tda_3 = np.delete(normalized_data_arr, 4, axis=2)
            tda_3 = np.delete(tda_3, 3, axis=2)

            concatenated_arr_tda3 = np.concatenate((tda_3, normalized_raw_data_arr), axis=2)

            # concatenated_arr_tda3 = np.concatenate((ablation_data, normalized_raw_data_arr), axis=2)

            # run the combined version
            # ablation
            for i in range(0, 5):
                ablation_data = np.delete(normalized_data_arr, i, axis=2)
                print("removing feature {}".format(i))
                for t in range(0, 1):
                    print("IT {} \n".format(t))
                    # auc_scores.append(LSTM_classifier(concatenated_arr_normalized, np_labels, "Combined_TDA5_ablation_{}".format(i), network))
                    auc_scores.append(LSTM_classifier(ablation_data, np_labels, "TDA5_ablation_{}".format(i), network))

            # auc_scores.append(LSTM_classifier(concatenated_arr_normalized, np_labels, "Combined_TDA5", network))
            # #
            # # # run the combined version
            # auc_scores.append(LSTM_classifier(concatenated_arr, np_labels, "Combined_not_normalized", network))
            #
            # # run the combined version TDA 3 not normalized
            # auc_scores.append(LSTM_classifier(concatenated_arr_tda3, np_labels, "Combined_TDA3", network))
            #
            # # #run the tda 5
            # auc_scores.append(LSTM_classifier(normalized_data_arr, np_labels, "TDA5", network))
            #
            # # run the tda 3
            # auc_scores.append(LSTM_classifier(tda_3, np_labels, "TDA3", network))
            #
            # # TDA 5 not normalized
            # auc_scores.append(LSTM_classifier(np_data, np_labels, "TDA5_not_normalized", network))
            #
            # #Raw
            # auc_scores.append(LSTM_classifier(normalized_raw_data_arr, np_labels, "Raw", network))
            #
            # #Raw not normilized
            #auc_scores.append(LSTM_classifier(np_data_raw, np_labels, "Raw_not_normalized", network))

    #
    #
    # print(f"Test Avg_AUC= {np.average(auc_scores)}  and std={np.std(auc_scores)}")
    # # indx += 1
    # #

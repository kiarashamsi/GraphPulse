import os
import pickle
import time
from keras.callbacks import Callback
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, GRU
from sklearn.metrics import roc_auc_score
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Dense
from util import file_util

"""
    Implement the RNN (Recurrent Neural Network) methods tailored including F_mapper(TDA5), F_snapshot(Raw) and GraphPulse data. 
"""

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

def read_seq_data(network):
    file_path = "Sequence/{}/".format(network)
    file = "seq.txt"
    seqData = dict()
    with open(file_path + file, 'rb') as f:
        # print("\n Reading Torch Data {} / {}".format(inx, len(files)))
        seqData = pickle.load(f)
    return seqData

def read_seq_data_by_file_name(network, file):

    """
    Read and load sequence data from a specific file associated with a network.

    Args:
        network (str): The name of the network for which the data is being read.
        file (str): The name of the file containing the sequence data.

    Returns:
        dict: A dictionary containing the loaded sequence data.
    """

    import os

    file_path = "C:/Users/kiara/Desktop/MyProject/GraphPulse/data/Sequences/{}/".format(network)
    seqData = dict()
    with open(file_path + file, 'rb') as f:
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

def LSTM_classifier(data, labels, spec, network):
    """
      Train and evaluate an LSTM-based neural network classifier.

      Args:
          data (numpy.ndarray): Input data for training and testing.
          labels (numpy.ndarray): Labels corresponding to the input data.
          spec (str): Specification string for configuring the model architecture.
          network (str): Network name for choosing the learning rate from a predefined list.

      Returns:
          float: ROC-LSTM (ROC AUC score).
      """

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
    elif ("GraphPulse" in spec):
        if ("ablation" in spec):
            model_LSTM.add(LSTM(64, input_shape=(7, 4), return_sequences=True))  # Adjust the number of units as needed
        else:
            model_LSTM.add(LSTM(64, input_shape=(7, 8), return_sequences=True))  # Adjust the number of units as needed
    elif ("TDA5" in spec):
        if ("ablation" in spec):
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

    lr_list = {
        "networkiconomi.txt": 0.00001,
        "networkcentra.txt": 0.00004,
        "networkbancor.txt": 0.00005,
        "networkaragon.txt": 0.00001,
        "Reddit_B.tsv": 0.00005
    }

    learning_rate = lr_list.get(network, 0.0001)
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
    print("Last AUC: {}".format(auc))
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
    return roc_LSTM


if __name__ == "__main__":
    networkList = ["mathoverflow.txt", "networkcoindash.txt", "networkiconomi.txt", "networkadex.txt", "networkdgd.txt",
                   "networkbancor.txt", "networkcentra.txt", "networkcindicator.txt", "networkaeternity.txt",
                   "networkaion.txt", "networkaragon.txt", "CollegeMsg.txt", "Reddit_B.tsv"]

    # can be choose from ["all","per_column","auto"]
    normalizer = "all"
    ablation = False
    # # for seq files
    auc_scores = []

    # # for combined version
    for network in networkList:
        for run in range(1, 6):
            print(f"RUN {run}")
            # for tdaVariable in tdaDifferentGraph:
            print("Working on {}\n".format(network))
            data = read_seq_data_by_file_name(network, "seq_tda.txt")
            data_raw = read_seq_data_by_file_name(network, "seq_raw.txt")

            indxs = []
            np_data = []
            np_data_raw = []
            np_labels = []

            # indx = 0
            for key, value in data["sequence"].items():
           
                if ("overlap" in key):
                    print("Processing network ({}) - with parameters {}".format(network, key))
                    np_labels = np.array(data["label"])
                    if (network == "networkdgd.txt"):
                        for i in range(0, len(value)):
                            if len(value[i]) != 7:
                                indxs.append(i)
                    if (network == "networkcentra.txt"):
                        for i in range(0, len(value)):
                            if len(value[i]) != 7:
                                indxs.append(i)

                    value = [item for index, item in enumerate(value) if index not in indxs]
                    print(file_util.find_indices_of_items_with_incorrect_size(value))
                    np_labels = np.delete(np_labels, indxs)
                    print(file_util.find_indices_of_items_with_incorrect_size(value))
                    np_data = np.array(value)
                    # auc_scores.append(LSTM_classifier(np_data, np_labels, key, network))

            # for raw data conbination
            for key_raw, value_raw in data_raw["sequence"].items():
                # if (indx == 1):
                #     break

                if ("raw" in key_raw):
                    print("Processing network ({}) - with parameters {}".format(network, key_raw))
                    np_labels_raw = np.array(data["label"])
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

            if (normalizer == "auto"):
                if ("centra" in network):
                    normalizer = "per_column"
                elif ("Reddit_B" in network):
                    normalizer = "per_column"
                else:
                    normalizer = "all"

            if (normalizer == "per_column"):
                min_values = np.min(np_data, axis=(0, 1))
                max_values = np.max(np_data, axis=(0, 1))

                # Normalize the entire array using Min-Max normalization formula
                normalized_data_arr = (np_data - min_values) / (max_values - min_values)
                normalized_data_arr = np.nan_to_num(normalized_data_arr, nan=0)
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
                min_values = np.min(np_data_raw)
                max_values = np.max(np_data_raw)

                # Normalize the entire array using Min-Max normalization formula
                normalized_raw_data_arr = (np_data_raw - min_values) / (max_values - min_values)
                normalized_raw_data_arr = np.nan_to_num(normalized_raw_data_arr, nan=0)

            # Concatenate the two arrays along the third axis (axis=2)
            concatenated_arr_normalized = np.concatenate((normalized_data_arr, normalized_raw_data_arr), axis=2)
            concatenated_arr = np.concatenate((np_data, np_data_raw), axis=2)

            # Keeping the TDA section
            tda_3 = np.delete(normalized_data_arr, 4, axis=2)
            tda_3 = np.delete(tda_3, 3, axis=2)
            concatenated_arr_tda3 = np.concatenate((tda_3, normalized_raw_data_arr), axis=2)

            # run the combined version
            # ablation
            if ablation:
                for i in range(0, 5):
                    ablation_data = np.delete(normalized_data_arr, i, axis=2)
                    print("removing feature {}".format(i))
                    for t in range(0, 1):
                        print("IT {} \n".format(t))
                        # auc_scores.append(LSTM_classifier(concatenated_arr_normalized, np_labels, "Combined_GraphPulse_ablation_{}".format(i), network))
                        auc_scores.append(
                            LSTM_classifier(ablation_data, np_labels, "GraphPulse_ablation_{}".format(i), network))

            auc_scores.append(LSTM_classifier(concatenated_arr_normalized, np_labels, "GraphPulse", network))

            # # run the combined version
            auc_scores.append(LSTM_classifier(concatenated_arr, np_labels, "GraphPulse_not_normalized", network))

            # #run the tda 5
            auc_scores.append(LSTM_classifier(normalized_data_arr, np_labels, "TDA5", network))

            # run the tda 3
            auc_scores.append(LSTM_classifier(tda_3, np_labels, "TDA3", network))

            # TDA 5 not normalized
            auc_scores.append(LSTM_classifier(np_data, np_labels, "TDA5_not_normalized", network))

            # Raw
            auc_scores.append(LSTM_classifier(normalized_raw_data_arr, np_labels, "Raw", network))

            # Raw not normilized
            auc_scores.append(LSTM_classifier(np_data_raw, np_labels, "Raw_not_normalized", network))

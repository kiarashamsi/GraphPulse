import os
import shutil
import pandas as pd
import datetime as dt

def creatBaselineDatasets(self, file):
    print("Processing {}".format(file))
    windowSize = 7  # Day
    gap = 3
    lableWindowSize = 7  # Day
    maxDuration = 20  # Day
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
    if max_transfer == min_transfer:
        max_transfer = min_transfer + 1

    while (data_last_date - window_start_date).days > (windowSize + gap + lableWindowSize):
        print("\nRemaining Process  {} ".format(

            (data_last_date - window_start_date).days / (windowSize + gap + lableWindowSize)))
        indx += 1

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

        selectedNetworkInGraphDataWindow = selectedNetworkInGraphDataWindow.assign(Snapshot=indx)
        csv_file_path = "../data/all_network/TimeSeries/Baseline/" + file
        if os.path.exists(csv_file_path):
            existing_df = pd.read_csv(csv_file_path)
        else:
            existing_df = pd.DataFrame()

        appended_df = existing_df.append(selectedNetworkInGraphDataWindow, ignore_index=True)
        appended_df.to_csv(csv_file_path, index=False)

        window_start_date = window_start_date + dt.timedelta(days=1)

    print(f"f{file} Processing has been completed!")

def creatBaselineDatasetsOther(self, file):
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

    while (data_last_date - window_start_date).days > (windowSize + gap + lableWindowSize):
        print("\nRemaining Process  {} ".format(

            (data_last_date - window_start_date).days / (windowSize + gap + lableWindowSize)))
        indx += 1

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

        selectedNetworkInGraphDataWindow = selectedNetworkInGraphDataWindow.assign(Snapshot=indx)
        csv_file_path = "../data/all_network/TimeSeries/Baseline/" + file
        if os.path.exists(csv_file_path):
            existing_df = pd.read_csv(csv_file_path)
        else:
            existing_df = pd.DataFrame()

        appended_df = existing_df.append(selectedNetworkInGraphDataWindow, ignore_index=True)
        appended_df.to_csv(csv_file_path, index=False)

        window_start_date = window_start_date + dt.timedelta(days=1)
    print(f"f{file} Processing has been completed!")

def creatBaselineDatasetsReddit(self, file):
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

    while (data_last_date - window_start_date).days > (windowSize + gap + lableWindowSize):
        print("\nRemaining Process  {} ".format(

            (data_last_date - window_start_date).days / (windowSize + gap + lableWindowSize)))
        indx += 1

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

        selectedNetworkInGraphDataWindow = selectedNetworkInGraphDataWindow.assign(Snapshot=indx)
        csv_file_path = "../data/all_network/TimeSeries/Baseline/" + file
        if os.path.exists(csv_file_path):
            existing_df = pd.read_csv(csv_file_path)
        else:
            existing_df = pd.DataFrame()

        appended_df = existing_df.append(selectedNetworkInGraphDataWindow, ignore_index=True)
        appended_df.to_csv(csv_file_path, index=False)

        window_start_date = window_start_date + dt.timedelta(days=1)
    print(f"f{file} Processing has been completed!")
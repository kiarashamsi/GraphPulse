from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import seaborn as sns
import datetime as dt


def create_hist_for_node_network_count(file_path):
    with open(file_path + "FInal_NodeTokenNetworkHashMap.txt", 'r') as f:
        # Load the dictionary from the file
        file_dict = eval(f.read())

        # Extract values from the dictionary
        values = list(file_dict.values())

        # Remove values equal to 1
        items_without_ones = {k: v for k, v in file_dict.items() if v != 1}
        values_without_one = list(items_without_ones.values())

        # Cap values at 20 (replace values greater than 20 with 20)
        values_A_to_plot = [21 if i > 20 else i for i in values_without_one]

        # Define custom bin ranges
        bins = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, max(values_A_to_plot)]

        # Create the histogram
        plt.hist(values_A_to_plot, bins=bins, align='left')
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.xticks(rotation=45, ha='right')

        # Get the x-axis tick labels
        labels = [int(x) if x < 20 else "20+" for x in bins]

        # Set the tick labels
        plt.xticks(bins[:-1], labels[:-1])

        # Add labels and title
        plt.xlabel('Token network count')
        plt.ylabel('Number of nodes with that token network count')
        plt.title('Histogram of node counts')

        # Save the plot as an image
        plt.savefig('histogram_token_count.png', dpi=300, bbox_inches="tight")

        # Show the plot
        plt.show()

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

def data_by_edge_visualization(data):
    # Divide node_count into 10 equally sized buckets
    bins = [0, 5, 10, 50, 100, 500, 1000, 2000, 5000, 10000]
    labels = ["0 - 5", "5 - 10", "10 - 50", "50 - 100", "100 - 500", "500 - 1K", "1K - 2K", "2K - 5K", "5K - inf"]
    data['bucket'] = pd.cut(data['edge_count'], bins=bins, labels=labels)

    # Group by bucket and label and count the number of 0's and 1's in each group
    grouped = data.groupby(['bucket', 'label'])['edge_count'].count().unstack()

    # Plot the results as a bar chart
    ax = grouped.plot(kind='bar')
    ax.set_xlabel('Unique Edge Count')
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('Network Count')
    ax.legend(title='Label', loc='upper right')
    plt.xticks(rotation=45)
    plt.savefig('Edge.png', dpi=300)
    plt.show()

def data_by_node_visualization(data):
    # Divide node_count into 10 equally sized buckets
    bins = [0, 5, 10, 50, 100, 500, 1000, 2000, 5000, 10000]
    labels = ["0 - 5", "5 - 10", "10 - 50", "50 - 100", "100 - 500", "500 - 1K", "1K - 2K", "2K - 5K", "5K - inf"]
    data['bucket'] = pd.cut(data['node_count'], bins=bins, labels=labels)

    # Group by bucket and label and count the number of 0's and 1's in each group
    grouped = data.groupby(['bucket', 'label'])['node_count'].count().unstack()

    # Plot the results as a bar chart
    ax = grouped.plot(kind='bar')
    ax.set_xlabel('Node Count')
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('Network Count')
    ax.legend(title='Label', loc='upper right')
    plt.xticks(rotation=45)
    plt.savefig('Node.png', dpi=300)
    plt.show()

def data_by_density_visualization(data):
    # Divide node_count into 10 equally sized buckets
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    labels = ["0 - 0.1", "0.1 - 0.2", "0.2 - 0.3", "0.3 - 0.4", "0.4 - 0.5", "0.5 - 0.6", "0.6 - 0.7", "0.7 - 0.8",
              "0.8 - 0.9", "0.9 - 1"]
    data['bucket'] = pd.cut(data['density'], bins=bins, labels=labels)

    # Group by bucket and label and count the number of 0's and 1's in each group
    grouped = data.groupby(['bucket', 'label'])['density'].count().unstack()

    # Plot the results as a bar chart
    ax = grouped.plot(kind='bar')
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Density')
    ax.set_ylabel('Network Count')
    ax.legend(title='Label', loc='upper right')
    plt.xticks(rotation=45)
    plt.savefig('Density.png', dpi=300)
    plt.show()

def data_by_peak_visualization(data):
    # Divide node_count into 10 equally sized buckets
    bins = [0, 5, 10, 50, 100, 500, 1000, 2000, 5000, 10000]
    labels = ["0 - 5", "5 - 10", "10 - 50", "50 - 100", "100 - 500", "500 - 1K", "1K - 2K", "2K - 5K", "5K - inf"]
    data['bucket'] = pd.cut(data['peak'], bins=bins, labels=labels)

    # Group by bucket and label and count the number of 0's and 1's in each group
    grouped = data.groupby(['bucket', 'label'])['peak'].count().unstack()

    # Plot the results as a bar chart
    ax = grouped.plot(kind='bar')
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Number of Peak Transactions')
    ax.set_ylabel('Network Count')
    ax.legend(title='Label', loc='upper right')
    plt.xticks(rotation=45)
    plt.savefig('Peak.png', dpi=300)
    plt.show()

def data_by_data_duration_visualization(data):
    # Divide node_count into 10 equally sized buckets
    bins = [20, 50, 100, 150, 200, 250, 300, 350, 400]
    labels = ["20 - 50", "50 - 100", "100 - 150", "150 - 200", "200 - 250", "250 - 300", "300 - 350", "350 - inf"]
    data['bucket'] = pd.cut(data['data_duration'], bins=bins, labels=labels)

    # Group by bucket and label and count the number of 0's and 1's in each group
    grouped = data.groupby(['bucket', 'label'])['data_duration'].count().unstack()

    # Plot the results as a bar chart
    ax = grouped.plot(kind='bar')
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Days of having active transactions')
    ax.set_ylabel('Network Count')
    plt.xticks(rotation=45)
    ax.legend(title='Label', loc='upper right')
    plt.savefig('Duration.png', dpi=300)
    plt.show()

def data_by_Avg_shortest_path_length_visualization(data):
    # Divide node_count into 10 equally sized buckets
    bins = [0, 2, 5, 10, 15, 20]
    labels = ["0 - 2", "2 - 5", "5 - 10", "10 - 15", "15 - inf"]
    data['bucket'] = pd.cut(data['avg_shortest_path_length'], bins=bins, labels=labels)

    # Group by bucket and label and count the number of 0's and 1's in each group
    grouped = data.groupby(['bucket', 'label'])['avg_shortest_path_length'].count().unstack()

    # Plot the results as a bar chart
    ax = grouped.plot(kind='bar')
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Avg Shortest Path Length')
    ax.set_ylabel('Network Count')
    plt.xticks(rotation=45)
    ax.legend(title='Label', loc='upper right')
    plt.savefig('Avg_shortest_path_length.png', dpi=300)
    plt.show()

def data_by_max_degree_centrality_visualization(data):
    # Divide node_count into 10 equally sized buckets
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    labels = ["0 - 0.1", "0.1 - 0.2", "0.2 - 0.3", "0.3 - 0.4", "0.4 - 0.5", "0.5 - 0.6", "0.6 - 0.7", "0.7 - 0.8",
              "0.8 - 0.9", "0.9 - 1"]
    data['bucket'] = pd.cut(data['max_degree_centrality'], bins=bins, labels=labels)

    # Group by bucket and label and count the number of 0's and 1's in each group
    grouped = data.groupby(['bucket', 'label'])['max_degree_centrality'].count().unstack()

    # Plot the results as a bar chart
    ax = grouped.plot(kind='bar')
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Max Degree Centrality')
    ax.set_ylabel('Network Count')
    ax.legend(title='Label', loc='upper left')
    plt.xticks(rotation=45)
    plt.savefig('max_degree_centrality.png', dpi=300)
    plt.show()

def data_by_max_closeness_centrality_visualization(data):
    # Divide node_count into 10 equally sized buckets
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    labels = ["0 - 0.1", "0.1 - 0.2", "0.2 - 0.3", "0.3 - 0.4", "0.4 - 0.5", "0.5 - 0.6", "0.6 - 0.7", "0.7 - 0.8",
              "0.8 - 0.9", "0.9 - 1"]
    data['bucket'] = pd.cut(data['max_closeness_centrality'], bins=bins, labels=labels)

    # Group by bucket and label and count the number of 0's and 1's in each group
    grouped = data.groupby(['bucket', 'label'])['max_closeness_centrality'].count().unstack()

    # Plot the results as a bar chart
    ax = grouped.plot(kind='bar')
    ax.set_xlabel('Max Closeness Centrality')
    ax.set_ylabel('Network Count')
    plt.xticks(rotation=45)
    ax.xaxis.set_label_position('top')
    ax.legend(title='Label', loc='upper left')
    plt.savefig('max_closeness_centrality.png', dpi=300)
    plt.show()

def data_by_max_betweenness_centrality_visualization(data):
    # Divide node_count into 10 equally sized buckets
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    labels = ["0 - 0.1", "0.1 - 0.2", "0.2 - 0.3", "0.3 - 0.4", "0.4 - 0.5", "0.5 - 0.6", "0.6 - 0.7", "0.7 - 0.8",
              "0.8 - 0.9", "0.9 - 1"]
    data['bucket'] = pd.cut(data['max_betweenness_centrality'], bins=bins, labels=labels)

    # Group by bucket and label and count the number of 0's and 1's in each group
    grouped = data.groupby(['bucket', 'label'])['max_betweenness_centrality'].count().unstack()

    # Plot the results as a bar chart

    ax = grouped.plot(kind='bar')
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Max Betweenness Centrality')
    ax.set_ylabel('Network Count')
    plt.xticks(rotation=45)
    ax.legend(title='Label', loc='upper left')
    plt.savefig('max_betweenness_centrality.png', dpi=300)
    plt.show()

def data_by_clique_visualization(data):
    # Divide node_count into 10 equally sized buckets
    bins = [0, 1, 2, 3, 4, 5, 6, 7]
    labels = ["1", "2", "3", "4", "5", "6 - inf"]
    data['bucket'] = pd.cut(data['clique_number'], bins=bins, labels=labels)

    # Group by bucket and label and count the number of 0's and 1's in each group
    grouped = data.groupby(['bucket', 'label'])['clique_number'].count().unstack()

    # Plot the results as a bar chart

    ax = grouped.plot(kind='bar')
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Clique Count')
    ax.set_ylabel('Network Count')
    plt.xticks(rotation=45)
    ax.legend(title='Label', loc='upper left')
    plt.savefig('clique_number.png', dpi=300)
    plt.show()

def data_by_avg_daily_trans_visualization(data):
    # Divide node_count into 10 equally sized buckets
    bins = [0, 5, 10, 20, 50, 100, 150, 200]
    labels = ["5", "10", "20", "50", "100", "150", "200 - inf"]
    data['bucket'] = pd.cut(data['avg_daily_trans'], bins=bins, labels=labels)

    # Group by bucket and label and count the number of 0's and 1's in each group
    grouped = data.groupby(['bucket', 'label'])['avg_daily_trans'].count().unstack()

    # Plot the results as a bar chart

    ax = grouped.plot(kind='bar')
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Average daily transactions Count')
    ax.set_ylabel('Network Count')
    plt.xticks(rotation=45)
    ax.legend(title='Label', loc='upper right')
    plt.savefig('avg_daily_trans_number.png', dpi=300)
    plt.show()
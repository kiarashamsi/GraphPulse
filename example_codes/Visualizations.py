from matplotlib import pyplot as plt, ticker
import numpy as np
import pandas as pd
import datetime as dt

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

    plt.figure(figsize=(10, 8))  # Set the figure size
    # Create the bar plot
    plt.bar(x, y)

    # Set the x-axis font size
    plt.xticks(fontsize=8)
    plt.xticks(rotation=45)

    # Show the value on top of each bar
    for index, value in enumerate(y):
        plt.text(index, value, str(value), ha='center', va='bottom')
    # # Adding labels to each bar
    # for i in range(len(x)):
    #     plt.text(i, y[i], str(y[i]), ha='center', va='bottom')
    #
    # # Calculating trend line
    # z = np.polyfit(range(len(x)), y, 1)
    # p = np.poly1d(z)
    # plt.plot(range(len(x)), p(range(len(x))), 'r--')
    #
    # # Adding trend line equation as a text
    # plt.text(len(x) - 1, p(len(x) - 1), f"Trend: {z[0]:.2f}x + {z[1]:.2f}", ha='right', va='center',
    #          color='red')

    # Customizing the plot
    plt.xlabel('Method')
    plt.ylabel('Time (sec)')
    plt.title('Model training cost (Dgd network)')

    # Displaying the plot
    plt.savefig('method_time_exp_bar.png', dpi=300)
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

    # Display the plot
    plt.legend()
    plt.tight_layout()
    plt.savefig('tda_time_exp_plot_r_f-log.png', dpi=300)
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
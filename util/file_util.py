import csv
import os
import pickle
import re
import numpy as np

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

def merge_dicts_old(list_of_dicts):
    merged_dict = {}
    for dictionary in list_of_dicts:
        for key, value in dictionary.items():
            if key in merged_dict:
                merged_dict[key].append(value)
            else:
                merged_dict[key] = [value]
    return merged_dict

def output_cleaner():
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

def save_fixed_seq(data, path):
    data["sequence"], outliers = merge_dicts(data["sequence"])
    np_labels = np.array(data["label"])
    np_labels = np.delete(np_labels, outliers)
    data["label"] = np_labels.tolist()
    with open(path + "seq_tda_ablation.txt", "wb") as output_file:
        pickle.dump(data, output_file)

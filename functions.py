import numpy as np
import pandas as pd
import re
import pandas as pd
import random
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Sentence Parsing
def tokenize(lines, token='word'):

    def split_by_quotes(text):
        return re.findall(r"'(.*?)'", text)

    results = []
    for line in lines:
        quoted_strings = split_by_quotes(line)
        results.extend(quoted_strings)

    return results

def read_txt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return lines

def write_to_vocab_file(file_path, vocab_items):
    with open(file_path, 'w') as f:
        for token, index in vocab_items:
            f.write(f"{token}\t{index}\n")



def random_oversample(csv_path, target_column='bug', random_state=42):
    df = pd.read_csv(csv_path)

    df[target_column] = np.where(df[target_column] > 0, 1, 0)


    class_counts = df[target_column].value_counts().to_dict()
    max_class_count = max(class_counts.values())
    minority_classes = [cls for cls, count in class_counts.items() if count < max_class_count]


    random.seed(random_state)

    for minority_class in minority_classes:

        additional_samples_needed = max_class_count - class_counts[minority_class]
        additional_samples = df[df[target_column] == minority_class].sample(n=additional_samples_needed, replace=True, random_state=random_state)
        df = pd.concat([df, additional_samples], ignore_index=True)

    return df



def calculate_metrics(y_test, pred_y,results):

    accuracy = accuracy_score(y_test, pred_y)
    precision = precision_score(y_test, pred_y)
    recall = recall_score(y_test, pred_y)
    f1 = f1_score(y_test, pred_y)

    results.append({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    })
    return results



def process_position_data(data_list):

    layers = []
    indices = []

    for data in data_list:
        tuples = re.findall(r'\((\d+),\s*(\d+)\)', data)
        layer_list = []
        index_list = []
        for layer, index in tuples:
            layer_list.append(int(layer))
            index_list.append(int(index))
        layers.append(layer_list)
        indices.append(index_list)

    return layers, indices



def positional_encoding(position, d_model):

    angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model) // 2)) / np.float32(d_model))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return angle_rads
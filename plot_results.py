import matplotlib.pyplot as plt
import pickle
import glob
import os
import pandas as pd
from collections import defaultdict


def read_data(file_names):
    with open(file_names[0], 'rb') as fp:
        window_sizes = pickle.load(fp)

    with open(file_names[1], 'rb') as fp:
        precision = pickle.load(fp)

    with open(file_names[2], 'rb') as fp:
        recall = pickle.load(fp)

    with open(file_names[3], 'rb') as fp:
        f_measure = pickle.load(fp)

    return window_sizes, precision, recall, f_measure


def plot_score_with_window_size():

    file_names = ['windowsizes', 'precision', 'recall', 'fmeasure']
    window_sizes, precision, recall, f_measure = read_data(file_names)
    print(recall)
    print(f_measure)
    plt.plot(window_sizes, precision, marker='o',
             color='b', linewidth=0.7, label='Precision')
    plt.plot(window_sizes, recall, marker='^', linewidth=0.7, label='Recall')
    plt.plot(window_sizes, f_measure, marker='D',
             color='k', linewidth=0.7, label='F1 Score')

    plt.xlabel('Window Size')
    plt.ylabel('Score')
    plt.xticks(window_sizes, window_sizes)
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.legend(loc='best')
    plt.show()


def per_day_active_events(path):

    file_name = '*.csv'
    all_files = glob.glob(os.path.join(path, file_name))

    # dict that maps day to the number of event chains active in that day
    per_day_event_chains = defaultdict(int)

    for f in all_files:
        days_in_event_chain = {}
        df = pd.read_csv(f, header=None, encoding='latin-1')
        df_list = df.values.tolist()

        for row in df_list:
            try:
                day = row[0][0:8]
                days_in_event_chain[day] = True
            except:
                continue

        for day in days_in_event_chain.keys():
            per_day_event_chains[day] += 1

    days = sorted(per_day_event_chains.keys())
    per_day_events = []

    for key in days:
        per_day_events.append(per_day_event_chains[key])

    return days, per_day_events


def plot_active_events(input_dir, output_dir):
    days, ground_truth_events = per_day_active_events(input_dir)
    days2, formed_events = per_day_active_events(output_dir)

    day_numbers = range(0, len(days))
    plt.plot(day_numbers, ground_truth_events, label='Ground Truth Chains')
    plt.plot(day_numbers, formed_events, label='Chains formed by RevDet')
    plt.xlabel('Day')
    plt.ylabel('Number of Active Events')
    plt.legend()
    plt.show()

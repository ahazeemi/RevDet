import pandas as pd
from sklearn import metrics
import numpy as np
from scipy.misc import comb
import glob
import os


'''
This script is for evaluation of event chain algorithm
'''


def myComb(a, b):
    return comb(a, b, exact=True)


vComb = np.vectorize(myComb)


def get_tp_fp_tn_fn(cooccurrence_matrix):
    tp_plus_fp = vComb(cooccurrence_matrix.sum(0, dtype=int), 2).sum()
    tp_plus_fn = vComb(cooccurrence_matrix.sum(1, dtype=int), 2).sum()
    tp = vComb(cooccurrence_matrix.astype(int), 2).sum()
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(cooccurrence_matrix.sum(), 2) - tp - fp - fn

    return [tp, fp, tn, fn]


def precision_recall_fmeasure(cooccurrence_matrix):
    tp, fp, tn, fn = get_tp_fp_tn_fn(cooccurrence_matrix)
    # print ("TP: %d, FP: %d, TN: %d, FN: %d" % (tp, fp, tn, fn))

    rand_index = (float(tp + tn) / (tp + fp + fn + tn))
    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    f1 = ((2.0 * precision * recall) / (precision + recall))

    return rand_index, precision, recall, f1


def run(input_dir, output_dir):

    original_clusters_path = input_dir
    file_name = '*.csv'
    all_files = glob.glob(os.path.join(original_clusters_path, file_name))

    gkg_id_to_index = {}
    class_labels_dict = {}
    label = 1
    index = 0

    for f in all_files:
        df = pd.read_csv(f, header=None, encoding='latin-1')
        df_list = df.values.tolist()

        for row in df_list:
            try:
                gkg_id = row[0].strip()
            except AttributeError:
                continue
            class_labels_dict[gkg_id] = label
            gkg_id_to_index[gkg_id] = index
            index += 1

        label += 1

    class_labels = [None]*len(class_labels_dict)
    for key, value in class_labels_dict.items():
        class_labels[gkg_id_to_index[key]] = value

    formed_clusters_path = output_dir
    file_name = '*.csv'
    all_files = glob.glob(os.path.join(formed_clusters_path, file_name))

    cluster_labels_dict = {}
    label = 1
    for f in all_files:
        df = pd.read_csv(f, header=None, encoding='latin-1')
        df_list = df.values.tolist()

        for row in df_list:
            gkg_id = row[0]
            cluster_labels_dict[gkg_id] = label

        label += 1

    cluster_labels = [0] * len(cluster_labels_dict)
    for key, value in cluster_labels_dict.items():
        cluster_labels[gkg_id_to_index[key]] = value

    matrix = metrics.cluster.contingency_matrix(class_labels, cluster_labels)
    rand_index, precision, recall, f1 = precision_recall_fmeasure(matrix)

    ari = metrics.cluster.adjusted_rand_score(class_labels, cluster_labels)
    nmi = metrics.normalized_mutual_info_score(class_labels, cluster_labels)

    result = [precision, recall, f1, ari, nmi]
    return result

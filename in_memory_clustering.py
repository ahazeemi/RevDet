import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import Birch
from scipy.sparse import hstack
from operator import itemgetter
import nltk
import glob
import os
from memory_profiler import profile
import evaluate_algorithm


def tokenize(text):

    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(
        text) for word in nltk.word_tokenize(sent)]
    return tokens


@profile
def algorithm():

    birch_thresh = 2.2

    path = 'per_day_data/'
    output = 'output_chains/'
    all_files = glob.glob(os.path.join(path, '*.csv'))

    df_list = []
    for file in all_files:
        df = pd.read_csv(file, header=None, encoding="latin-1")
        df_list.append(df)

    df = pd.concat(df_list, ignore_index=True)

    themes = pd.DataFrame(df[4])
    locations = pd.DataFrame(df[5])
    heading = pd.DataFrame(df[9])

    themes.columns = ['themes']
    locations.columns = ['locations']
    heading.columns = ['heading']

    for row in heading.itertuples():
        if type(row.heading) == float:
            heading.loc[row.Index, 'heading'] = ['#']
            continue

        # one hot approach
        tokenized_data = tokenize(row.heading.lower())
        heading.loc[row.Index, 'heading'] = tokenized_data

    row_dict = df.copy(deep=True)
    row_dict.fillna('', inplace=True)
    row_dict.index = range(len(row_dict))
    # dictionary that maps row number to row
    row_dict = row_dict.to_dict('index')

    locations = pd.DataFrame(
        locations['locations'].str.split(';'))  # splitting locations

    for row in locations.itertuples():
        try:
            row.locations[:] = [(row.locations[0].split('#'))[3]]
        except:
            continue

    mlb = MultiLabelBinarizer(sparse_output=False)
    sparse_heading = pd.DataFrame(mlb.fit_transform(
        heading['heading']), columns=mlb.classes_, index=heading.index)

    mlb2 = MultiLabelBinarizer(sparse_output=False)
    sparse_locations = pd.DataFrame(mlb2.fit_transform(locations['locations']), columns=mlb2.classes_,
                                    index=locations.index)

    df = hstack([sparse_heading, sparse_locations])

    brc = Birch(branching_factor=50, n_clusters=None,
                threshold=birch_thresh, compute_labels=True)
    predicted_labels = brc.fit_predict(df)

    clusters = {}
    n = 0

    for item in predicted_labels:
        if item in clusters:
            # since row_dict[n] is itself a dictionary
            clusters[item].append(list((row_dict[n]).values()))
        else:
            clusters[item] = [list((row_dict[n]).values())]
        n += 1

    file_number = 1
    for item in clusters:
        if len(clusters[item]) > 0:
            clusters[item].sort(key=itemgetter(1))
            df = pd.DataFrame(clusters[item])
            df.to_csv(output + str(file_number) + '.csv',
                      sep=',', index=0, header=None)
            file_number = file_number + 1


def main():
    algorithm()
    result = evaluate_algorithm.run(
        'redundancy_removed_chains/', 'output_chains/')
    print(
        'Window Size: {}, Birch Threshold: {}, Precision: {:.2f}, Recall: {:.2f}, F1-Score: {:.2f}, NMI: {:.2f}, ARI: {:.2f}'.format(
            8, 2.3, result[0], result[1], result[2], result[3], result[4]))


if __name__ == "__main__":
    main()

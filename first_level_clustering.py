import pandas as pd
import glob
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import Birch
from sklearn.decomposition import PCA
from scipy.sparse import hstack
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import math


'''
This script performs first level clustering for redundancy removal.
It clusters news from w2e_gdelt on themes and locations,
further clusters them on counts,
retains one news from each cluster,
and outputs per group file to filtered_groups folder
'''


def one_hot_encode(df):
    mlb = MultiLabelBinarizer(sparse_output=True)
    sparse_df = mlb.fit_transform(df)
    return sparse_df


def main():
    stop_words = set(stopwords.words('english'))

    # parameters, determined through experiments
    perform_pca = False
    birch_thresh = 3.6
    count_thresh = 0.1

    path = r'C:\Users\lenovo\PycharmProjects\FYP\w2e\w2e_gdelt_groups_heading'
    file_name = '*.csv'
    all_files = glob.glob(os.path.join(path, file_name))

    all_files = all_files[0:50]

    print(all_files)

    for f in all_files:

            file_prefix = f.split('.')[0]
            file_prefix = file_prefix.split('\\')[-1]

            df = pd.read_csv(f, header=None, encoding='latin-1')

            df.columns = ['record_id', 'date', 'url', 'counts', 'themes', 'locations', 'persons',
                          'organizations', 'tone']

            # Retaining only those news which have non-null themes and locations
            df = df[pd.notnull(df['themes'])]
            df = df[pd.notnull(df['locations'])]

            df_locations = pd.DataFrame(df['locations'])

            row_dict = df.copy(deep=True)
            row_dict.fillna('', inplace=True)
            row_dict.index = range(len(row_dict))
            row_dict = row_dict.to_dict('index')  # dictionary that maps row number to row

            df = df[df.columns[[4]]]
            df.columns = ['themes']

            try:
                df = pd.DataFrame(df['themes'].str.split(';'))  # splitting themes
            except:
                print(file_prefix)
                print("ThemesError")
                continue

            try:
                df_locations = pd.DataFrame(df_locations['locations'].str.split(';'))  # splitting locations
            except:
                print(file_prefix)
                print("Locations Error")
                continue

            for row in df_locations.itertuples():
                for i in range(0, len(row.locations)):
                    try:
                        row.locations[i] = (row.locations[i].split('#'))[3]  # for retaining only ADM1 Code
                    except:
                        continue

            df = df[pd.notnull(df['themes'])]
            for row in df.itertuples():
                row.themes[:] = [x for x in row.themes if not x.startswith(('CRISISLEX'))]
                if len(row.themes) == 1 and row.themes[0] == '':
                    row.themes.append('#')
                    row.themes.pop(0)

            sparse_themes = one_hot_encode(df['themes'])
            sparse_locations = one_hot_encode(df_locations['locations'])

            df = hstack([sparse_themes, sparse_locations])

            # Reducing dimensions through principal component analysis
            if perform_pca:
                pca = PCA(n_components=None)
                df = pd.DataFrame(pca.fit_transform(df))

            brc = Birch(branching_factor=50, n_clusters=None, threshold=birch_thresh, compute_labels=True)
            try:
                predicted_labels = brc.fit_predict(df)
            except:
                print("Birch Error")
                continue

            clusters = {}
            n = 0

            for item in predicted_labels:
                if item in clusters:
                    clusters[item].append(
                        list((row_dict[n]).values()))  # since row_dict[n] is itself a dictionary
                else:
                    clusters[item] = [list((row_dict[n]).values())]
                n += 1

            # clustering within each cluster, on counts
            count_clusters = {}  # dictionary which maps original_cluster_key to new clusters within that cluster
            for item in clusters:
                count_clusters[item] = {}
                cluster_df = pd.DataFrame(clusters[item])
                cluster_row_dict = cluster_df.copy(deep=True)
                cluster_row_dict.fillna('', inplace=True)
                cluster_row_dict.index = range(len(cluster_row_dict))
                cluster_row_dict = cluster_row_dict.to_dict('index')

                df_counts = pd.DataFrame(cluster_df[cluster_df.columns[[3]]])
                df_counts.columns = ['counts']
                df_counts = pd.DataFrame(df_counts['counts'].str.split(';'))  # splitting counts

                for row in df_counts.itertuples():

                    for i in range(0, len(row.counts)):
                        try:
                            temp_list = row.counts[i].split('#')
                            row.counts[i] = temp_list[0] + '#' + temp_list[1] + '#' + temp_list[
                                5]  # for retaining only COUNT_TYPE and QUANTITY and LOCATION ADM1 Code
                        except:
                            continue

                    row.counts[:] = [x for x in row.counts if not x.startswith(
                        'CRISISLEX')]  # Removing CRISISLEX Entries due to elevated false positive rate

                    if len(row.counts) == 1 and row.counts[0] == '':
                        row.counts.append('#')  # so that news with no counts are clustered together
                        row.counts.pop(0)

                    if row.counts[len(row.counts) - 1] == '':
                        row.counts.pop()

                mlb4 = MultiLabelBinarizer()
                df_counts = pd.DataFrame(mlb4.fit_transform(df_counts['counts']),
                                         columns=mlb4.classes_, index=df_counts.index)

                brc2 = Birch(branching_factor=50, n_clusters=None, threshold=count_thresh, compute_labels=True)
                predicted_labels2 = brc2.fit_predict(df_counts)

                n2 = 0
                for item2 in predicted_labels2:
                    if item2 in count_clusters[item]:
                        count_clusters[item][item2].append(
                            list((cluster_row_dict[
                                n2]).values()))  # since cluster_row_dict[n2] is itself a dictionary
                    else:
                        count_clusters[item][item2] = [list((cluster_row_dict[n2]).values())]
                    n2 += 1

            '''with open('filtered_one_clusters/'+file_prefix+'.csv', 'w', encoding='utf-8') as file:
                writer = csv.writer(file, delimiter=",")
                for item in count_clusters:
                    for item2 in count_clusters[item]:
                        for i in range(0, len(count_clusters[item][item2])):
                            writer.writerow(count_clusters[item][item2][i])
                        writer.writerow('#')'''

            data = []
            for item in count_clusters:
                    for item2 in count_clusters[item]:
                        data.append(count_clusters[item][item2][0])
            df = pd.DataFrame(data)
            df.sort_values(by=[0], inplace=True)

            df.columns = ['record_id', 'date', 'url', 'counts', 'themes', 'locations', 'persons', 'organizations',
                          'tone']

            for row in df.itertuples():

                if type(row.heading) == float:
                    df.loc[row.Index, 'heading'] = ['#']
                    continue

                tokenized_data = nltk.word_tokenize(row.heading)
                stopWordRemovedData = []
                for word in tokenized_data:
                    if word not in stop_words:
                        stopWordRemovedData.append(word)

                porter_stemmer = PorterStemmer()
                for i in range(0, len(stopWordRemovedData)):
                    stopWordRemovedData[i] = porter_stemmer.stem(stopWordRemovedData[i])

                df.loc[row.Index, 'heading'] = ' '.join(stopWordRemovedData)

            df.to_csv('filtered_groups_heading/'+file_prefix+'.csv', sep=',', index=0, header=None)


if __name__ == "__main__":
    main()
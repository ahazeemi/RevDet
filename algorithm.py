import os
from operator import itemgetter

import nltk
import pandas as pd
from scipy.sparse import hstack
from sklearn.cluster import Birch
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import jaccard_similarity_score

import csv
import sys
csv.field_size_limit(sys.maxsize)



def tokenize(text):

    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(
        text) for word in nltk.word_tokenize(sent)]
    return tokens


def run(input_dir, output_dir, birch_thresh, window_size):

    file_index = {}
    fIndex = 0

    path = input_dir  # use your path

    temp_path = output_dir

    days = []

    with open('days.txt') as file:
        for line in file:
            line = line.strip()
            days.append(line)



    i = 1
    progress_df = pd.DataFrame()
    for k in range(0, len(days), window_size):

        first_half = days[k: k + window_size]

        df_list = []
        for file in first_half:
            df = pd.read_csv(path + file + '.csv',
                             header=None, encoding="latin-1",engine='python')
            df_list.append(df)

        df = pd.concat(df_list, ignore_index=True)
        print("check1")
        print(df)

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

        print("check2")
        print(heading)

        row_dict = df.copy(deep=True)
        row_dict.fillna('', inplace=True)
        row_dict.index = range(len(row_dict))
        # dictionary that maps row number to row
        row_dict = row_dict.to_dict('index')

        print("check3--row_dict done")


        locations = pd.DataFrame(
            locations['locations'].str.split(';'))  # splitting locations

        for row in locations.itertuples():
            try:
                row.locations[:] = [(row.locations[0].split('#'))[3]]
            except:
                continue

        print("check4--locations done")
        print(locations)


        mlb = MultiLabelBinarizer(sparse_output=True)
        sparse_heading = mlb.fit_transform(heading['heading'])

        print("check5--heading mlb done")

        mlb2 = MultiLabelBinarizer(sparse_output=True)
        sparse_locations = mlb2.fit_transform(locations['locations'])

        print("check6--locations mlb done")

        df = hstack([sparse_heading, sparse_locations])

        print("check7--hstack done")

        brc = Birch(branching_factor=50, n_clusters=None,
                    threshold=birch_thresh, compute_labels=True)

        predicted_labels = brc.fit_predict(df)

        print("check8--birch done")
        print(predicted_labels)

        clusters = {}
        n = 0

        for item in predicted_labels:
            if item in clusters:
                # since row_dict[n] is itself a dictionary
                clusters[item].append(list((row_dict[n]).values()))
            else:
                clusters[item] = [list((row_dict[n]).values())]
            n += 1


        print("check9--cluster dictionary formed")

        for item in clusters:
            if len(clusters[item]) > 0:
                clusters[item].sort(key=itemgetter(1))
                file_path_temp = os.path.join(
                    temp_path, "f" + str(fIndex) + ".csv")

                fIndex += 1
                df = pd.DataFrame(clusters[item])

                print("check 10--data loaded into df")


                eR = df.head(1)  # eR : earliest representative

                print("check 11--eR stored")

                for index, row in progress_df.iterrows():

                    temp_df = pd.DataFrame(eR)

                    temp_df = temp_df.append(row)#temp_df is containing lR of previous cluster and eR of current cluster..

                    locations = pd.DataFrame(temp_df[5])

                    locations = locations.reset_index(drop=True)

                    print("check 12--locations stored in df")

                    locations.columns = ['locations']

                    heading = pd.DataFrame(temp_df[9])
                    heading = heading.reset_index(drop=True)
                    heading.columns = ['heading']

                    print("check 13--heading stored in df")

                    locations = pd.DataFrame(
                        locations['locations'].str.split(';'))  # splitting locations

                    for l_row in locations.itertuples():

                        for i in range(0, len(l_row.locations)):
                            try:
                                l_row.locations[i] = (l_row.locations[i].split('#'))[
                                    3]  # for retaining only ADM1 Code
                            except:
                                continue
                    print("check 14--locations splitted")


                    for h_row in heading.itertuples():
                        if type(h_row.heading) == float:
                            heading.loc[h_row.Index, 'heading'] = ['#']
                            continue

                        tokenized_data = tokenize(h_row.heading.lower())
                        heading.at[h_row.Index, 'heading'] = tokenized_data

                    mlb = MultiLabelBinarizer(sparse_output=False)
                    sparse_heading = pd.DataFrame(mlb.fit_transform(heading['heading']), columns=mlb.classes_,
                                                  index=heading.index)

                    mlb2 = MultiLabelBinarizer(sparse_output=False)
                    sparse_locations = pd.DataFrame(mlb2.fit_transform(
                        locations['locations']), columns=mlb2.classes_, index=locations.index)

                    row_list = sparse_heading.values.tolist()

                    heading_similarity = jaccard_similarity_score(
                        row_list[0], row_list[1])

                    row_list = sparse_locations.values.tolist()
                    loc_similarity = jaccard_similarity_score(
                        row_list[0], row_list[1])

                    if heading_similarity > 0.1 and loc_similarity > 0.1:
                        previous_chain_id = temp_df[0].iloc[1]
                        print(previous_chain_id)
                        file_path_temp = file_index[previous_chain_id]
                        print("check 15--file_path_temp found")
                        conDf = pd.read_csv(
                            file_path_temp,header=None, encoding="latin-1",engine='python')
                        print("check 16--file readed")
                        df = pd.concat([conDf, df], ignore_index=True)
                        break

                lR = pd.DataFrame(df.tail(1))   # latest representative
                print("check 17 --stored last representative")
                file_index[lR[0].iloc[0]] = file_path_temp

                print("check 18--file address stored")

                progress_df = pd.concat([progress_df,lR],ignore_index=True)
 
                print("check 19--concat of pd done")

                df.drop_duplicates(subset=0, keep="first", inplace=True)
                print("check 20--drop duplicates done")
                df.sort_values(by=[0], inplace=True)
                print("check 21--sorting done")

                print(file_path_temp)
                print(df)
                df.to_csv(file_path_temp, sep=',', index=0, header=None)

                print("check 22--file address stored")
                

        i += 1

import pandas as pd
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import Birch
from scipy.sparse import hstack
from operator import itemgetter
import nltk
import re
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import hdbscan
#from memory_profiler import profile
import evaluate_algorithm
import os, shutil
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
import cProfile


def tokenize(text):

    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    return tokens
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

#@profile
def algorithm(birch_thresh, window_size):

    file_index = {}
    fIndex = 0
    SuccessCount = 0

    fileMerger = open("mergedData2.txt", "w")

    path = 'per_day_data_heading/'  # use your path

    temp_path = r'C:\Users\lenovo\PycharmProjects\FYP\w2e\output_chains2'

    days = []


    with open('days_heading.txt') as file:
        for line in file:
            line = line.strip()
            days.append(line)

    i = 1
    progress_df = pd.DataFrame()
    for k in range(0, len(days), window_size):

        # sliding window size n

        first_half = days[k: k + window_size]

        df_list = []
        for file in first_half:
            df = pd.read_csv(path + file, header=None, encoding="latin-1")
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
        row_dict = row_dict.to_dict('index')  # dictionary that maps row number to row

        locations = pd.DataFrame(locations['locations'].str.split(';'))  # splitting locations

        for row in locations.itertuples():
            try:
                row.locations[:] = [(row.locations[0].split('#'))[3]]
                #row.locations.append((row.locations[1].split('#'))[3])
            except:
                continue

        mlb = MultiLabelBinarizer(sparse_output=False)
        sparse_heading = pd.DataFrame(mlb.fit_transform(heading['heading']), columns=mlb.classes_, index=heading.index)

        mlb2 = MultiLabelBinarizer(sparse_output=False)
        sparse_locations = pd.DataFrame(mlb2.fit_transform(locations['locations']), columns=mlb2.classes_, index=locations.index)

        df = hstack([sparse_heading,sparse_locations])

        brc = Birch(branching_factor=50, n_clusters=None, threshold=birch_thresh, compute_labels=True)
        predicted_labels = brc.fit_predict(df)

        clusters = {}
        n = 0

        for item in predicted_labels:
            if item in clusters:
                clusters[item].append(list((row_dict[n]).values()))  # since row_dict[n] is itself a dictionary
            else:
                clusters[item] = [list((row_dict[n]).values())]
            n += 1

        for item in clusters:
            if len(clusters[item]) > 0:
                clusters[item].sort(key=itemgetter(1))
                file_path_temp = os.path.join(temp_path, "f" + str(fIndex) + ".csv")
                fIndex += 1
                df = pd.DataFrame(clusters[item])

                eR = df.head(1) # eR : earliest representative
                # eR_locations = eR.iloc[5].split(';')
                # eR_locations = eR_locations[0].split('#')[3]

                for index, row in progress_df.iterrows():
                    temp_df = pd.DataFrame(eR)
                    #row_df = pd.DataFrame(row).transpose()
                    temp_df = temp_df.append(row)

                    locations = pd.DataFrame(temp_df[5])
                    locations = locations.reset_index(drop=True)
                    locations.columns = ['locations']

                    heading = pd.DataFrame(temp_df[9])
                    heading = heading.reset_index(drop=True)
                    heading.columns = ['heading']

                    locations = pd.DataFrame(locations['locations'].str.split(';'))  # splitting locations

                    for l_row in locations.itertuples():

                        for i in range(0, len(l_row.locations)):
                            try:
                                l_row.locations[i] = (l_row.locations[i].split('#'))[3]  # for retaining only ADM1 Code
                            except:
                                continue
                 
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
                    sparse_locations = pd.DataFrame(mlb2.fit_transform(locations['locations']), columns=mlb2.classes_,index=locations.index)

                    row_list = sparse_heading.values.tolist()
                    a = np.array(row_list[0]).reshape(1,-1)
                    b = np.array(row_list[1]).reshape(1,-1)
                    heading_similarity = cosine_similarity(a, b)

                    row_list = sparse_locations.values.tolist()
                    a = np.array(row_list[0]).reshape(1, -1)
                    b = np.array(row_list[1]).reshape(1, -1)
                    loc_similarity = cosine_similarity(a, b)

                    if heading_similarity > 0.5 and loc_similarity > 0.5:
                        previous_chain_id = temp_df[0].iloc[1]
                        file_path_temp = file_index[previous_chain_id]
                        conDf = pd.read_csv(file_path_temp, header=None, encoding="latin-1")
                        df = pd.concat([conDf, df], ignore_index=True)
                        #df.drop_duplicates(subset=0, keep="first", inplace=True)
                        #df.sort_values(by=[0], inplace=True)
                        SuccessCount += 1
                        # print("Success: " + str(SuccessCount))
                        fileMerger.write(str(SuccessCount) + ": " + file_path_temp + "\n")
                        break

                    #sparse_heading.to_csv('sph.csv')

                lR = pd.DataFrame(df.tail(1))   # latest representative
                file_index[lR[0].iloc[0]] = file_path_temp

                progress_df = lR

                df.drop_duplicates(subset=0, keep="first", inplace=True)
                df.to_csv(file_path_temp, sep=',', index=0, header=None)

        i += 1

    fileMerger.close()


def delete_files():

    folder = r'C:\Users\lenovo\PycharmProjects\FYP\w2e\output_chains2'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def main():
    result = []
    highest_f1_score = 0

    for birch_thresh in np.arange(2.3, 2.5, 0.2):
        for window_size in range(5, 20, 1):
            algorithm(birch_thresh, window_size)
            temp_result = evaluate_algorithm.run()
            print(birch_thresh,window_size,temp_result)
            if temp_result[3] > highest_f1_score:
                highest_f1_score = temp_result[3]
                result = temp_result
                result.insert(0,window_size)
                result.insert(0,birch_thresh)
            delete_files()

    print("Result")
    print(result)


if __name__ == "__main__":
    main()

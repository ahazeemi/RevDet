import pandas as pd
import glob
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import Birch
from sklearn.decomposition import PCA
from scipy.sparse import hstack
import nltk

from nltk.stem.porter import PorterStemmer
import re
import unidecode

nltk.download('stopwords')
from nltk.corpus import stopwords

'''
This script performs redundancy removal on RevDet Dataset.
It performs stopword removal and stemming on article title, clusters news on title and locations,
further clusters them on counts,
retains one news from each cluster,
and outputs per group file to output folder
'''


def tokenize_and_stem(text, porter_stemmer):#first tokenizing the words and then removing tokens not containing letters..and then replacing words with root words(going->go..)
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(
        text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)


    #filtered_tokens = [unidecode.unidecode(word) for word in filtered_tokens if word[0].isupper()]
    stems = [porter_stemmer.stem(t) for t in filtered_tokens]
    
    return stems


def one_hot_encode(df):
    mlb = MultiLabelBinarizer(sparse_output=True)
    sparse_df = mlb.fit_transform(df)
    return sparse_df


def remove_stop_words(df, stop_words):

    for row in df.itertuples():

        if type(row.heading) == float:
            df.loc[row.Index, 'heading'] = ['#']
            continue

        porter_stemmer = PorterStemmer()

        processed_data = tokenize_and_stem(
            row.heading, porter_stemmer)
        stop_word_removed_data = []
        for word in processed_data:
            if word.lower() not in stop_words:
                stop_word_removed_data.append(word)

        df.loc[row.Index, 'heading'] = stop_word_removed_data
        print(df)

    return df


def run(input_dir, output_dir):

    stop_words = set(stopwords.words('english'))

    # parameters, determined through experiments
    birch_thresh = 2.4
    count_thresh = 0.1

    perform_pca = False
    path = input_dir
    output_path = output_dir
    file_name = '*.csv'
    all_files = glob.glob(os.path.join(path, file_name))#storing the path of all input csv files..

    
    for f in all_files:

        file_prefix = f.split('.')[0]#picking the prefix of filename and removing the extensions..

        file_prefix = file_prefix.split('\\')[-1]

        df = pd.read_csv(f, header=None, encoding='latin-1')#storing the input csv files into a dataframe..

        df.columns = ['record_id', 'date', 'url', 'counts', 'themes', 'locations', 'persons',
                      'organizations', 'tone', 'heading'] #providing column names(these names will replace 0,1,2...indexing)



        # Retaining only those news which have non-null locations and heading
        df = df[pd.notnull(df['locations'])]
        df = df[pd.notnull(df['heading'])]


        # removing news with wrong scraped title e.g. bloomberg instead of article title
        #removing news with having length of article's title< 20..
        try:
            mask = (df['heading'].str.len() >= 20)
            df = df.loc[mask]
        except:
            continue



        # retaining original heading for analysis afterwards
        #adding one extra col named 'heading_original'

        df['heading_original'] = df['heading']

        locations = pd.DataFrame(df['locations'])
        heading = pd.DataFrame(df['heading'])

        locations.columns = ['locations']
        heading.columns = ['heading']


        # stop-word removal and stemming in heading
        heading = remove_stop_words(heading, stop_words)

        #storing locations and heading in a separate dataframe..
        #df_locations = pd.DataFrame(df['locations'])
        #df_heading = pd.DataFrame(df['heading'])

        # dictionary that maps row number to row, helps later in forming clusters through cluster labels

        row_dict = df.copy(deep=True)#copied df dataframe into row_dict
        row_dict.fillna('', inplace=True)

        row_dict.index = range(len(row_dict))#setting the index of the df from 0 to len-1..(because of del of some rows index might not be in order in original df)
        row_dict = row_dict.to_dict('index')#converting row_dict into a dictionary with index as keys and rows mapped with index as a dictionary with columns as key..




        try:
            locations = pd.DataFrame(
                locations['locations'].str.split(';'))  # splitting locations
        except:
            continue
    

        for row in locations.itertuples():#Converting the locations into their ADM1 code..
            for i in range(0, len(row.locations)):
                try:
                    row.locations[i] = (row.locations[i].split('#'))[
                        3]  # for retaining only ADM1 Code
                except:
                    continue


        sparse_heading = one_hot_encode(heading['heading'])
        sparse_locations = one_hot_encode(locations['locations'])

        df = hstack([sparse_heading, sparse_locations])


        # Reducing dimensions through principal component analysis
        if perform_pca:
            pca = PCA(n_components=None)
            df = pd.DataFrame(pca.fit_transform(df))

        brc = Birch(branching_factor=50, n_clusters=None,
                    threshold=birch_thresh, compute_labels=True)
        try:
            predicted_labels = brc.fit_predict(df)
        except:
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
        # dictionary which maps original_cluster_key to new clusters within that cluster
        count_clusters = {}
        for item in clusters:

            count_clusters[item] = {}
            cluster_df = pd.DataFrame(clusters[item])
            cluster_row_dict = cluster_df.copy(deep=True)
            cluster_row_dict.fillna('', inplace=True)
            cluster_row_dict.index = range(len(cluster_row_dict))
            cluster_row_dict = cluster_row_dict.to_dict('index')


            df_counts = pd.DataFrame(cluster_df[cluster_df.columns[[3]]])

            df_counts.columns = ['counts']


            df_counts = pd.DataFrame(
                df_counts['counts'].str.split(';'))  # splitting counts


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


                if len(row.counts) == 1 and row.counts[0] == '':  #empty count will only be present at beginning or end..
                    # so that news with no counts are clustered together
                    row.counts.append('#')
                    row.counts.pop(0)

                if row.counts[len(row.counts) - 1] == '':
                    row.counts.pop()

            mlb4 = MultiLabelBinarizer()
            df_counts = pd.DataFrame(mlb4.fit_transform(df_counts['counts']),
                                     columns=mlb4.classes_, index=df_counts.index)



            brc2 = Birch(branching_factor=50, n_clusters=None,
                         threshold=count_thresh, compute_labels=True)
            predicted_labels2 = brc2.fit_predict(df_counts)


            n2 = 0
            for item2 in predicted_labels2:
                if item2 in count_clusters[item]:
                    count_clusters[item][item2].append(
                        list((cluster_row_dict[
                            n2]).values()))  # since cluster_row_dict[n2] is itself a dictionary
                else:
                    count_clusters[item][item2] = [
                        list((cluster_row_dict[n2]).values())]
                n2 += 1

        data = []
        for item in count_clusters:
            for item2 in count_clusters[item]:
                data.append(count_clusters[item][item2][0])



        df = pd.DataFrame(data)
        df.sort_values(by=[0], inplace=True)

        df.to_csv(output_path+file_prefix+'.csv',
                  sep=',', index=0, header=None)

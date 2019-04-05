import pandas as pd
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import Birch
from scipy.sparse import hstack
from operator import itemgetter
import nltk
from nltk.corpus import stopwords

def algorithm():

    fileIndex = {}
    fIndex = 0
    SuccessCount = 0

    fileMerger = open("mergedData.txt", "w")

    path = 'per_day_data_heading/'  # use your path

    temp_path = r'C:\Users\lenovo\PycharmProjects\FYP\w2e\output_chains'

    birch_thresh = 2.5
    window_size = 10
    progress_df = pd.DataFrame()



    days = []

    with open('days_heading.txt') as file:
        for line in file:
            line = line.strip()
            days.append(line)

    i = 1
    for k in range(0, len(days), window_size):

        # sliding window size n
        # first_half: 0 - n
        # second_half n - 2n

        first_half = days[k: k + window_size]
        second_half = days[k + window_size: k + (2*window_size)]

        if len(second_half) == 0:
            return

        print("iteration :",i)
        print("first half ", first_half)
        print("second half ", second_half)

        df_list = []
        for file in first_half:
            df = pd.read_csv(path + file, header=None, encoding="latin-1")
            df_list.append(df)

        # first half data
        df_1 = pd.concat(df_list, ignore_index=True)

        df_list = []
        for file in second_half:
            df = pd.read_csv(path + file, header=None, encoding="latin-1")
            df_list.append(df)

        # second half data
        df_2 = pd.concat(df_list, ignore_index=True)

        df = pd.concat([df_1, df_2, progress_df], ignore_index=True)

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

            tokenized_data = nltk.word_tokenize(row.heading)
            heading.loc[row.Index, 'heading'] = tokenized_data
            # except:
            #     heading.loc[row.Index, 'heading'] = ['#']

            #print(heading.loc[row.Index, 'heading'])

        row_dict = df.copy(deep=True)
        row_dict.fillna('', inplace=True)
        row_dict.index = range(len(row_dict))
        row_dict = row_dict.to_dict('index')  # dictionary that maps row number to row


        #themes = pd.DataFrame(themes['themes'].str.split(';'))  # splitting themes
        locations = pd.DataFrame(locations['locations'].str.split(';'))  # splitting locations

        for row in locations.itertuples():
            for i in range(0, len(row.locations)):
                try:
                    row.locations[i] = (row.locations[i].split('#'))[3]  # for retaining only ADM1 Code
                except:
                    continue

        # for row in themes.itertuples():
        #     row.themes[:] = [x for x in row.themes if not x.startswith(('CRISISLEX'))]
        #     if len(row.themes) == 1 and row.themes[0] == '':
        #         row.themes.append('#')
        #         row.themes.pop(0)

        # for row in heading.itertuples():
        #     if row.heading is None:
        #
        #     if len(row.heading) == 1 and row.heading[0] == '':
        #         row.heading.append('#')
        #         row.heading.pop(0)

        # mlb = MultiLabelBinarizer(sparse_output=True)
        # sparse_themes = mlb.fit_transform(themes['themes'])

        mlb = MultiLabelBinarizer(sparse_output=False)
        sparse_heading = mlb.fit_transform(heading['heading'])

        mlb2 = MultiLabelBinarizer(sparse_output=True)
        sparse_locations = mlb2.fit_transform(locations['locations'])

        #df = hstack([sparse_themes, sparse_locations])

        df = hstack([sparse_heading, sparse_locations])

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

        progress_df = pd.DataFrame()
        for item in clusters:
            if len(clusters[item]) > 0:
                clusters[item].sort(key=itemgetter(1))
                file_path_temp = os.path.join(temp_path, "f" + str(fIndex) + ".csv")
                fIndex += 1
                df = pd.DataFrame(clusters[item])

                eR = df[0].iloc[0]  # eR : earliest representative
                if eR in second_half:
                    continue

                if eR in fileIndex:
                    file_path_temp = fileIndex[eR]
                    conDf = pd.read_csv(file_path_temp, header=None, encoding="latin-1")
                    df = pd.concat([conDf, df], ignore_index=True)
                    df.drop_duplicates(subset=0, keep="first", inplace=True)
                    df.sort_values(by=[0], inplace=True)
                    SuccessCount += 1
                    print("Success: " + str(SuccessCount))
                    fileMerger.write(str(SuccessCount) + ": " + file_path_temp + "\n")

                lR = pd.DataFrame(df.tail(1))   # latest representative
                fileIndex[lR[0].iloc[0]] = file_path_temp
                progress_df = pd.concat([lR, progress_df], ignore_index=True)
                df.to_csv(file_path_temp, sep=',', index=0, header=None)

        # progress_df.to_csv("jj77", sep=',', index=0, header=None)
        print("This iteration: " + str(i))

        i += 1

    fileMerger.close()


def main():
    algorithm()


if __name__ == "__main__":
    main()

import glob
import os
import pandas as pd

'''
This script transforms event chains to per day files
'''


def main():

    path = r'C:\Users\lenovo\PycharmProjects\FYP\w2e\ground_truth_chains'

    file_name = '*.csv'
    all_files = glob.glob(os.path.join(path, file_name))

    per_day_data = {}

    for f in all_files:
        df = pd.read_csv(f, header=None, encoding='latin-1')
        df_list = df.values.tolist()

        print(f)

        for row in df_list:
            try:
                day = row[0][0:8]
                if day not in per_day_data:
                    per_day_data[day] = []

                per_day_data[day].append(row)
            except:
                continue

    days = sorted(per_day_data.keys())
    days.sort()

    with open('days.txt', 'w') as f:
        for item in days:
            f.write("%s\n" % item)


if __name__ == "__main__":
    main()

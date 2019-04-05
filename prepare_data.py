import glob
import os
import pandas as pd

'''
This script transforms event groups from w2e dataset to per day files
'''


def main():
    path = r'C:\Users\lenovo\PycharmProjects\FYP\w2e\filtered_groups_heading'

    file_name = '*.csv'
    all_files = glob.glob(os.path.join(path, file_name))

    per_day_data = {}

    for f in all_files:
        df = pd.read_csv(f, header=None, encoding='latin-1')
        df_list = df.values.tolist()

        print(f)

        for row in df_list:
            day = row[0][0:8]
            if day not in per_day_data:
                per_day_data[day] = []

            per_day_data[day].append(row)
            if day == '20160130':
                print(row)

    days = []
    for key, value in per_day_data.items():
        days.append(key+'.csv')
        df = pd.DataFrame(value)
        df.sort_values(by=[0], inplace=True)
        df.to_csv('per_day_data_heading/' + key + '.csv', sep=',', index=0, header=None)

    days.sort()

    with open('days_heading.txt', 'w') as f:
        for item in days:
            f.write("%s\n" % item)


if __name__ == "__main__":
    main()

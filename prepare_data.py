import glob
import os
import pandas as pd
import remove_redundancy
import argparse

'''
This script transforms event chains to per day files
'''

parser = argparse.ArgumentParser()

parser.add_argument(
    '--groundtruthchains',
    default='ground_truth_chains/',
    type=str,
    help='Input directory for input event chains'
)
parser.add_argument(
    '--perdaydata',
    default='per_day_data/',
    type=str,
    help='Output directory for per day data'
)
parser.add_argument(
    '--redundancyremoveddata',
    default='redundancy_removed_chains/',
    type=str,
    help='Output directory for redundancy removed chains'
)


def main(args):

    input_dir = args.groundtruthchains
    output_dir = args.redundancyremoveddata

    print("Removing redundancy")

    remove_redundancy.run(input_dir, output_dir)

    input_dir = output_dir
    output_dir = args.perdaydata

    print("Preparing per day files")

    path = os.path.join(input_dir,"ground_truth_chains")

    file_name = "*.csv"
    all_files = glob.glob(os.path.join(path, file_name))
    

    per_day_data = {}

    for f in all_files:
        df = pd.read_csv(f, header=None, encoding='latin-1')
        df_list = df.values.tolist()

        for row in df_list:
            try:
                day = row[0][0:8]#extracting the dates..
                if day not in per_day_data:
                    per_day_data[day] = []

                per_day_data[day].append(row)
            except:
                continue

    for key in per_day_data:
        df = pd.DataFrame(per_day_data[key])
        df.sort_values(by=[0], inplace=True)
        df.to_csv(output_dir + key + '.csv', sep=',', index=0, header=None)

    days = sorted(per_day_data.keys())
    days.sort()#list of days in sorted order

    with open('days.txt', 'w') as f:
        for item in days:
            f.write("%s\n" % item)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

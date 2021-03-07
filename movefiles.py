import glob
import os
import pandas as pd

path = r'C:\Users\lenovo\PycharmProjects\FYP\revdet\events'

file_name = '*.csv'
all_files = glob.glob(os.path.join(path, file_name))

per_day_data = {}

for f in all_files:
    file_prefix = f.split('.')[0]
    file_prefix = file_prefix.split('\\')[-1]

    df = pd.read_csv('chains/'+file_prefix+'.csv',
                     header=None, encoding='latin-1')
    df.to_csv('ground_truth_chains/' + file_prefix +
              '.csv', sep=',', index=0, header=None)

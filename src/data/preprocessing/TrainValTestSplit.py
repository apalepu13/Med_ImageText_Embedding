import pandas as pd
import numpy as np

datadir = '../../../data/'
splits_csv = pd.read_csv(datadir + 'mimic-cxr-2.0.0-split.csv')
print(splits_csv.head())
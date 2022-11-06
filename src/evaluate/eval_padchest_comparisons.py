import pickle5 as pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
from functools import reduce

import numpy as np

modelpaths = ['/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/clip_regularized/exp2/',
               '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/clip_regularized/exp3/']
num_models = [5, 5]
sub = ['all', 'all']
name = ['regularized zeroshot', 'unregularized zeroshot', 'chexzero']
colors = ['r', 'b']

fullpaths = [modelpaths[i] + 'padchest_aucs_' + str(num_models[i]) + '_' + sub[i] + '.pickle' for i in range(len(sub))]
fullpaths = fullpaths + ['/n/data2/hms/dbmi/beamlab/anil/CheXzero/files/padchest_aucs_all.pickle']

dfs = []

plt.figure()
for i in range(len(fullpaths)):
    with open(fullpaths[i], 'rb') as handle:
        aucs, names = pickle.load(handle)
        print(fullpaths[i])
        print(np.mean(np.array(list(aucs.values()))))
        dfs.append(pd.DataFrame({'names':names, name[i]:[aucs[n] for n in names]}))

df = reduce(lambda x, y: pd.merge(x, y, on = 'names'), dfs)
print(df.head())
print(df['unregularized zeroshot'].values.mean())
print(df['regularized zeroshot'].values.mean())
print(df['chexzero'].values.mean())
df['unreg_diff'] = df['unregularized zeroshot'] - df['chexzero']
df['reg_diff'] = df['regularized zeroshot'] - df['chexzero']
df['reg_unreg_diff'] = df['regularized zeroshot'] - df['unregularized zeroshot']
df['maxvalue'] = (df['reg_unreg_diff'] > 0).astype(int) * 4 + (df['reg_diff'] > 0).astype(int) * 2 + (df['unreg_diff'] > 0).astype(int)

#6,7 = reg best = 29
#0,4 = chexzero best = 10
#1,3 = unreg best = 18
'''
print((df['reg_unreg_diff']>0.01).sum())
print((df['reg_unreg_diff']<-0.01).sum())
print(np.unique(df['maxvalue'].values, return_counts=True))
'''

df2 = df.loc[:, ['names', 'regularized zeroshot', 'unregularized zeroshot', 'chexzero']]
pd.set_option('display.max_rows', 60)
print(df2.sort_values(['regularized zeroshot'], ascending=False))

pd.set_option('display.max_rows', 5)
df = df.loc[:, ['names','unreg_diff', 'reg_diff']]
dfm = df.melt('names', var_name='model', value_name='aucs')

plt.figure(figsize = (30, 20))
sns.catplot(dfm, x='aucs', y='names', hue='model')
plt.savefig('/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/results/padchest.png')





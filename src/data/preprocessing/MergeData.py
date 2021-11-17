import os
import sys
import pandas as pd
import numpy as np

dataDir = '/n/scratch3/users/a/anp2971/mimic_cxr/'
studyList = pd.read_csv(dataDir + '/txt_files/cxr-study-list.csv')
reportsDir = dataDir + 'txt_files/reports/'
print(studyList.shape)
print(np.unique(studyList['study_id']).shape)
print(studyList.head())

failed = []
for i in np.arange(studyList.shape[0]):
    s_id = 's' + str(studyList['study_id'].values[i])
    p_id = 'p' + str(studyList['subject_id'].values[i]) + '/'
    p_small_id = p_id[:3] + '/'
    fromfile = reportsDir + s_id + '.txt'
    tofile = dataDir + 'jpg_files/physionet.org/files/mimic-cxr-jpg/2.0.0/files/' + p_small_id + p_id + s_id
    try:
        os.system('mv ' + fromfile + ' ' + tofile)
    except:
        failed.append(s_id)
        print(s_id)

print(len(failed))
print(failed)


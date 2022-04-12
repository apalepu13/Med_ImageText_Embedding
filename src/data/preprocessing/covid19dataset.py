import os
import pandas as pd

basepath = '/n/data2/hms/dbmi/beamlab/covid19qu/infection_dat/'
testpath = basepath + 'Test/'
trainpath = basepath + 'Train/'
valpath = basepath + 'Val/'

paths = [trainpath, valpath, testpath]
adds = ['COVID-19', 'Non-COVID', 'Normal']
imtypes = ['images', 'infection masks', 'lung masks']

groups = ['train', 'val', 'test']
labs = ['covid19', 'Pneumonia', 'No Finding']
myDf = []
for i, p in enumerate(paths):
    for j, a in enumerate(adds):
        mypath = p + a
        mypath_ims = os.path.join(mypath, 'images')
        filelist = os.listdir(mypath_ims)
        impaths = [os.path.join(mypath, 'images', f) for f in filelist]
        infpaths = [os.path.join(mypath, 'infection masks', f) for f in filelist]
        lungpaths = [os.path.join(mypath, 'lung masks', f) for f in filelist]
        for k, f in enumerate(impaths):
            myDf.append([impaths[k], infpaths[k], lungpaths[k], groups[i], labs[j]])

myDf = pd.DataFrame(myDf)
myDf.columns = ['im_path', 'inf_path', 'lung_path', 'group', 'label']
myDf.to_csv(basepath + 'data_list.csv')



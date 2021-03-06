import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
datadir = '/n/data2/hms/dbmi/beamlab/medpix/'
file = datadir + 'medpix_case_data_table.csv'

df = pd.read_csv(file)
unique, ct = np.unique(df['Diagnosis'].astype(str), return_counts=True)

print(unique)
print(ct)
print(unique.shape)
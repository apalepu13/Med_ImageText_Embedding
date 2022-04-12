import sys
sys.path.insert(0, '../../models/joint_embedding_model/')
import pandas as pd
import numpy as np
from Data_Loading import *
import torch
print("CUDA Available: " + str(torch.cuda.is_available()))
from Transformer import *
from Pretraining import *
from matplotlib import pyplot as plt

def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: http://stackoverflow.com/a/25074150/395857
    By HYRY
    '''
    pc.update_scalarmappable()
    ax = pc.axes
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)

def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: http://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels):
    '''
    Inspired by:
    - http://stackoverflow.com/a/16124677/395857
    - http://stackoverflow.com/a/25074150/395857
    '''

    # Plot it out
    fig, ax = plt.subplots()
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='Reds', vmin=0.0, vmax=1)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    # resize
    fig = plt.gcf()
    fig.set_size_inches(cm2inch(40, 20))

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

heads = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']
mimic_dat = pd.read_csv('/n/data2/hms/dbmi/beamlab/mimic_cxr/mimic-cxr-2.0.0-chexpert.csv')
mydat = mimic_dat.loc[:, heads]



for i, h in enumerate(heads):
    print(i, h)
    mydat.iloc[:, i] = ((mydat.iloc[:, i] == 1).astype(int))


label_headers = mydat.columns
label_data = mydat.astype(int).values
cooccurrence_matrix = np.dot(label_data.transpose(),label_data)
cooccurrence_matrix_diagonal = np.diagonal(cooccurrence_matrix)
print('\ncooccurrence_matrix:\n{0}'.format(cooccurrence_matrix))

with np.errstate(divide='ignore', invalid='ignore'):
    cooccurrence_matrix_percentage = np.nan_to_num(np.true_divide(cooccurrence_matrix, cooccurrence_matrix_diagonal[:, None]))
print('\ncooccurrence_matrix_percentage:\n{0}'.format(cooccurrence_matrix_percentage))

# Plotting
label_header_with_count = [ '{0} ({1})'.format(label_header, cooccurrence_matrix_diagonal[label_number]) for label_number, label_header in enumerate(label_headers)]
print('\nlabel_header_with_count: {0}'.format(label_header_with_count))

x_axis_size = cooccurrence_matrix.shape[0]
y_axis_size = cooccurrence_matrix.shape[1]
title = "Co-occurrence matrix\n"
xlabel= ''#"Labels"
ylabel= ''#"Labels"
xticklabels = label_header_with_count
yticklabels = label_header_with_count
heatmap(cooccurrence_matrix_percentage, title, xlabel, ylabel, xticklabels, yticklabels)
plt.savefig('/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/results/mimic_label_heatmap.png', dpi=300, bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures
#plt.show()

mydat_exclusive = mydat[mydat.sum(axis=1) == 1].astype(int).values
cooccur_exclusive = np.diagonal(np.dot(mydat_exclusive.transpose(), mydat_exclusive))
label_header_with_count = [ '{0} ({1})'.format(label_header, cooccur_exclusive[label_number]) for label_number, label_header in enumerate(label_headers)]
print('\nexclusive positivity label counts: {0}'.format(label_header_with_count))

'''
mydat['sum'] = mydat.values.sum(axis=1)
counts = mydat['sum'].value_counts()
plt.title("label distribution")
plt.xlabel("0-none, 1-cardio, 2-ede, 4-consoli, 8-atel, 16-pleural")
plt.bar(counts.index, counts.values)
plt.savefig('/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/results/mimic_label_dist.png', bbox_inches='tight')
'''
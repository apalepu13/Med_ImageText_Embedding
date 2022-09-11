import pickle
import numpy as np
import matplotlib.pyplot as plt
load_file = "/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/results/training/chexpert_finetune_losses.pickle"
with open(load_file, 'rb') as handle:
    all_train_losses, all_val_losses = pickle.load(handle)

plt.figure(figsize = (8,8))
mod_names = ['CNN Real', 'CNN Shortcut', 'CLIP Real', 'CLIP Shortcut']


#print(all_train_losses)
#print(all_val_losses)
trains = np.array(all_train_losses)
vals = np.array(all_val_losses)
if trains.shape[1] > 5:
    for i in np.arange(4):
        trains[:, i] = np.mean(trains[:, np.arange(trains.shape[1]) == i], axis = 1)
    trains = trains[:, :4]

print(trains.shape)
print(vals.shape)

best_real = np.min(vals[:, 0])
best_synth = np.min(vals[:, 1])

fig, ax = plt.subplots(2,2, figsize = (8,8))
for i,m in enumerate(mod_names):
    myax = ax[i//2, i%2]
    myax.plot(np.arange(trains.shape[0]), trains[:, i], 'r', label='Train')
    myax.plot(np.arange(vals.shape[0]), vals[:, i], 'b', label='Val')
    myax.set_title(mod_names[i])
    myax.set_ylim((0,1))
    myax.axhline(best_real, color = 'k', alpha = 0.5)
    myax.axhline(best_synth, color='k', alpha=0.5)
    myax.legend()

plt.savefig('/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/results/training/chexpert_finetune_losses.png', bbox_inches='tight')


from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage

baseline = 0.3

def plot_all_train_val_losses(train_losses, val_losses, lossnames, finetune=True, detail=None, resultsDir = '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/results/training/'):
    fig, ax = plt.subplots(2, 2, figsize = (12, 12), sharey=True)
    positions = [ax[0, 0], ax[1, 0], ax[0, 1], ax[1, 1]]
    for i in range(len(train_losses)):
        CLIP = 'clip' in lossnames[i] or 'CLIP' in lossnames[i]
        synth = 'synth' in lossnames[i] or 'synthetic' in lossnames[i]
        plot_train_val_loss(train_losses[i], val_losses[i], lossname = "BCE" if (finetune or not CLIP) else "CLIP", finetune=finetune, CLIP=CLIP, synth=synth, detail=detail, myax=positions[i])

    savetitle = ("Finetuned_" if finetune else "") + (detail if detail else "")
    plt.savefig(resultsDir + savetitle + ".png", bbox_inches='tight')

def plot_train_val_loss(train_losses, val_losses, lossname="BCE", finetune=True, CLIP=False, synth=False,
                        resultsDir = '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/results/training/', expNumber = -1, detail = None, myax=None):
    epoch_nums = np.arange(len(train_losses))

    train_losses = ndimage.median_filter(train_losses, size=3)
    val_losses = ndimage.median_filter(val_losses, size=3)
    myax.plot(epoch_nums, train_losses, 'r', label="finetune train")
    myax.plot(epoch_nums, val_losses, 'b', label="finetune val")
    myax.set_xlabel("Epochs", size=16)

    if lossname=="CLIP":
        myax.set_ylim(0.8, 3.5)
    else:
        myax.set_ylim(0.1, 0.8)
    myax.set_xlim(0, len(train_losses))

    origtitle = lossname + " loss "
    if finetune:
        origtitle = origtitle + "after finetuning."
    else:
        origtitle = origtitle + "while training."

    myax.set_ylabel(origtitle, size=16)

    origtitle = ("Synth trained " if synth else "Real trained ") + (("CLIP") if CLIP else ("CNN"))
    myax.set_title(origtitle, size=20)
    myax.axhline(baseline, 0, len(train_losses))
    myax.legend()
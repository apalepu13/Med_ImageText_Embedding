import argparse
import sys
sys.path.insert(0, '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/src/models/Patch_CLIP/')
import torch
import CLIP_Embedding
import MedDataHelpers
import utils
from torchmetrics import AUROC
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import metrics
import pickle

print("CUDA Available: " + str(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(args):
    rerun = True
    heads = np.array(['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'])
    heads2 = np.array(['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion', 'No Finding'])
    if args.subset == 'a' or args.subset == 'all':
        subset = ['all']
    elif args.subset == 't' or args.subset == 'test':
        subset = ['test']

    aucs, aucs_synth, aucs_adv, tprs, fprs, thresholds = {}, {}, {}, {}, {}, {}
    auroc = AUROC(pos_label=1)
    filters = MedDataHelpers.getFilters(args.je_model_path)
    modname = os.path.basename(args.je_model_path)
    dat = MedDataHelpers.getDatasets(source=args.sr, subset=subset, synthetic=False, filters=filters,
                                     heads=heads)  # Real
    DLs = MedDataHelpers.getLoaders(dat, args, shuffle=False)
    DL = DLs[subset[0]]
    #clip_models = [CLIP_Embedding.getCLIPModel(args.je_model_path)]
    clip_models = CLIP_Embedding.getCLIPModel(args.je_model_path, num_models = 5)
    tot_preds = 0
    for cnum, clip_model in enumerate(clip_models):
        if os.path.isfile(args.je_model_path + 'predictions/' + args.sr + str(cnum) + 'preds.pickle') and not rerun:
            with open(args.je_model_path + 'predictions/' + args.sr + str(cnum) + 'preds.pickle', 'rb') as handle:
                test_preds, test_targets = pickle.load(handle)
        else:
            test_preds, test_targets = utils.get_all_preds(DL, clip_model,similarity=True, heads=heads, convirt=True, normalization=True, squash_studies=False) #N c
            test_preds = test_preds[0].cpu().detach().numpy()
            test_targets = test_targets.cpu().int().detach().numpy()
            with open(args.je_model_path + 'predictions/' + args.sr + str(cnum) + 'preds.pickle', 'wb') as handle:
                pickle.dump([test_preds, test_targets], handle, protocol=pickle.HIGHEST_PROTOCOL)

        if cnum == 0:
            tot_preds = test_preds
        else:
            tot_preds = tot_preds + test_preds

    tot_preds = tot_preds / (cnum + 1)


    for i, h in enumerate(heads):
        targs = test_targets[:, i]
        tpreds = tot_preds[:, i]
        if subset != ['test']:
            twhere = np.argwhere(np.logical_or(targs == 0,targs == 1))[0]
            targs = targs[twhere]
            tpreds = tpreds[twhere]
            fprs[h], tprs[h], thresholds[h] = metrics.roc_curve(targs, tpreds)
            aucs[h] = np.round(auroc(test_preds[twhere, i], test_targets[twhere, i].int()).item(), 5)
        else:
            fprs[h], tprs[h], thresholds[h] = metrics.roc_curve(targs, tpreds)
            aucs[h] = np.round(metrics.auc(fprs[h], tprs[h]), 5)
            #aucs[h] = np.round(auroc(test_preds[:, i], test_targets[:, i].int()).item(), 5)
    aucs['Total'] = np.round(np.mean(np.array([aucs[h] for h in heads])), 5)

    print("Normal")
    print("Total AUC avg: ", aucs['Total'])
    for i, h in enumerate(heads):
        print(h, aucs[h])

    #ROC Curve
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = {'Atelectasis': 'r', 'Cardiomegaly': 'tab:orange', 'Consolidation': 'g', 'Edema': 'c',
              'Pleural Effusion': 'tab:purple'}
    for i, h in enumerate(heads):
        ax.plot(fprs[h], tprs[h], color=colors[h], label=h + ", AUC = " + str(np.round(aucs[h], 4)))
    xrange = np.linspace(0, 1, 100)
    avgTPRS = np.zeros_like(xrange)
    for i, h in enumerate(heads):
        avgTPRS = avgTPRS + np.interp(xrange, fprs[h], tprs[h])
    avgTPRS = avgTPRS / 5
    ax.plot(xrange, avgTPRS, color='k', label="Average, AUC = " + str(np.round(aucs['Total'], 4)))
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_title("ROC Curves for labels", size=30)
    ax.set_xlabel("False Positive Rate", size=24)
    ax.set_ylabel("True Positive Rate", size=24)
    ax.legend(prop={'size': 16})
    plt.savefig(args.results_dir + modname +  "roc_curves.png", bbox_inches="tight")





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--je_model_path', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/clip_regularized/exp2/', help='path for saving trained models')
    parser.add_argument('--sr', type=str, default='chextest') #c, co
    parser.add_argument('--subset', type=str, default='test')
    parser.add_argument('--embed_size', type=int, default=128, help='dimension of word embedding vectors')
    parser.add_argument('--batch_size', type=int, default=32) #32 normally
    parser.add_argument('--results_dir',type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/results/zeroshot/')
    parser.add_argument('--dat_dir', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/data/')
    args = parser.parse_args()
    print(args)
    main(args)
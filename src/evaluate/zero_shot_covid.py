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

print("CUDA Available: " + str(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_best_thresh(fprs, tprs, thresholds):
    dist = np.sqrt(np.square(fprs) + np.square(1-tprs))
    best_thresh_ind = np.argmin(dist)
    return thresholds[best_thresh_ind]

def main(args):
    heads1 = np.array(['covid19', 'Pneumonia'])
    heads2 = np.array(['covid19', 'Pneumonia', 'No Finding'])
    if args.subset == 'a' or args.subset == 'all':
        subset = ['all']
    elif args.subset == 't' or args.subset == 'test':
        subset = ['test']
    clip_models = CLIP_Embedding.getCLIPModel(args.je_model_path, num_models=1)
    clip_models = [clip_models]
    filters = MedDataHelpers.getFilters(args.je_model_path)
    #TODO: add frontal filters to coviddataset
    modname = args.je_model_path[-5:-1]

    dat = MedDataHelpers.getDatasets(source=args.sr, subset=subset, synthetic=False, filters = filters, heads = heads1) #Real
    print(dat['all'])
    print(dat['all'].__len__())
    DLs = MedDataHelpers.getLoaders(dat, args)
    DL = DLs[subset[0]]

    aucs, aucs_synth, aucs_adv, tprs, fprs, thresholds, accs = {}, {}, {}, {}, {}, {}, {}
    auroc = AUROC(pos_label=1)

    for k, clip_model in enumerate(clip_models):
        test_preds, test_targets = utils.get_all_preds(DL, clip_model,similarity=True, heads=heads2)
        test_preds = test_preds[0].cpu()
        test_targets = test_targets.cpu()
        print(test_targets.sum(dim=0))
        for i, h in enumerate(heads1):
            test_preds[:, i] = test_preds[:, i] - test_preds[:, 2]
        test_preds = test_preds[:, :2]
        test_preds = torch.nn.Softmax(dim=1)(test_preds)
        if k == 0:
            tot_preds = test_preds
        else:
            tot_preds += test_preds

    test_preds = tot_preds


    for i, h in enumerate(heads1):
        fprs[h], tprs[h], thresholds[h] = metrics.roc_curve(test_targets[:, i].int().detach().numpy(), test_preds[:, i].detach().numpy())
        best_thresh = get_best_thresh(fprs[h], tprs[h], thresholds[h])
        accs[h] = metrics.accuracy_score(test_targets[:, i].int().detach().numpy(), test_preds[:, i].detach().numpy() > best_thresh)
        print(h, "acc", accs[h])
        aucs[h] = np.round(auroc(test_preds[:, i], test_targets[:, i].int()).item(), 5)
    aucs['Total'] = np.round(np.mean(np.array([aucs[h] for h in heads1])), 5)
    print("Normal")
    print("Total AUC avg: ", aucs['Total'])
    for i, h in enumerate(heads1):
        print(h, aucs[h])

    #ROC Curve
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = {'covid19': 'r', 'Pneumonia': 'tab:orange', 'No Finding': 'g'}
    for i, h in enumerate(heads1):
        ax.plot(fprs[h], tprs[h], color=colors[h], label=h + ", AUC = " + str(np.round(aucs[h], 4)))
    xrange = np.linspace(0, 1, 100)
    avgTPRS = np.zeros_like(xrange)
    for i, h in enumerate(heads1):
        avgTPRS = avgTPRS + np.interp(xrange, fprs[h], tprs[h])
    avgTPRS = avgTPRS / len(heads1)
    ax.plot(xrange, avgTPRS, color='k', label="Average, AUC = " + str(np.round(aucs['Total'], 4)))
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_title("ROC Curves for labels", size=30)
    ax.set_xlabel("False Positive Rate", size=24)
    ax.set_ylabel("True Positive Rate", size=24)
    ax.legend(prop={'size': 16})
    plt.savefig(args.results_dir + modname +  "covid_roc_curves.png", bbox_inches="tight")





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--je_model_path', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/clip_regularized/exp2/', help='path for saving trained models')
    parser.add_argument('--sr', type=str, default='co') #c, co
    parser.add_argument('--subset', type=str, default='all')
    parser.add_argument('--embed_size', type=int, default=128, help='dimension of word embedding vectors')
    parser.add_argument('--batch_size', type=int, default=32) #32 normally
    parser.add_argument('--results_dir',type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/results/zeroshot/')
    parser.add_argument('--dat_dir', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/data/')
    args = parser.parse_args()
    print(args)
    main(args)
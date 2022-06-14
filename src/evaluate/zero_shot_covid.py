import argparse
import torch
import numpy as np
import sys
sys.path.insert(0, '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/src/models/joint_embedding_model/')
from CNN import *
from HelperFunctions import *
from Transformer import *
from torchmetrics import AUROC
import matplotlib.pyplot as plt
print("CUDA Available: " + str(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
from sklearn import metrics
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def getbinarydat(preds, targets, elim):
    userows = np.where(targets[:, elim] == 0)[0]
    targets = targets[userows, :]
    preds = preds[userows, :]
    return preds, targets

def main(args):
    all_heads = [np.array(['covid19', 'No Finding']), np.array(['Pneumonia', 'No Finding']), np.array(['covid19', 'Pneumonia'])]
    fprs = {}
    tprs = {}
    all_aucs = {}
    thresholds = {}
    for qqq, heads in enumerate(all_heads):
        dat = getDatasets(source=args.sr, subset=[args.subset], heads=heads)
        [DL] = getLoaders(dat, args, subset=[args.subset])

        je_vision_model, transformer, tokenizer = getSimilarityClassifier(args.model_path, args.model, heads=heads, avg_embedding=True, text_num=500)
        test_preds, test_targets = get_all_preds(DL, je_vision_model, heads, device = device)

        if args.use_softmax:
            test_preds = torch.nn.Softmax(dim=1)(test_preds)

        test_preds = test_preds.cpu()
        test_targets = test_targets.cpu()
        aucs = {}
        auroc = AUROC(pos_label=1)
        for i, h in enumerate(heads):
            if i == 0:
                fprs[qqq], tprs[qqq], thresholds[qqq] = metrics.roc_curve(test_targets[:, i].int(), test_preds[:, i])
                all_aucs[qqq] = np.round(auroc(test_preds[:, i], test_targets[:, i].int()).item(), 4)

            aucs[h] = auroc(test_preds[:, i], test_targets[:, i].int()).item()
            if heads.shape[0] == 3:
                aucs[h + h] = auroc(test_preds[:, ((i + 1) % 3)], test_targets[:, i].int()).item()
                aucs[h + h + h] = auroc(test_preds[:, ((i + 2) % 3)], test_targets[:, i].int()).item()

        aucs['Total'] = np.mean(np.array([aucs[h] for h in heads]))
        print("Total AUC avg: ", aucs['Total'])
        for i, h in enumerate(heads):
            print(h, aucs[h])
            if heads.shape[0] == 3:
                print(h, aucs[h], aucs[h + h], aucs[h + h + h])

        #just pneumonia experiment
        if heads.shape[0] == 3:
            co_vs_p_preds, co_vs_p_targ = getbinarydat(test_preds, test_targets, 2)
            co_vs_n_preds, co_vs_n_targ = getbinarydat(test_preds, test_targets, 1)
            p_vs_n_preds, p_vs_n_targ = getbinarydat(test_preds, test_targets, 0)
            covp_auc = auroc(co_vs_p_preds[:, 1], co_vs_p_targ[:, 0].int()).item()
            covn_auc = auroc(co_vs_n_preds[:, 1], co_vs_n_targ[:, 0].int()).item()
            pvn_auc = auroc(p_vs_n_preds[:, 1], p_vs_n_targ[:, 1].int()).item()
            print(covp_auc)
            print(covn_auc)
            print(pvn_auc)

    fig,ax = plt.subplots(figsize=(8,8))
    colors = ['r', 'b', 'g']
    titles = ['covid vs no finding, AUC=', 'pneumonia vs no finding, AUC=', 'covid vs pneumonia, AUC=']
    for i in range(3):
        ax.plot(fprs[i], tprs[i], color = colors[i], label=titles[i] + str(all_aucs[i]))
    ax.set_title("Zero shot covid ROC curves")
    ax.legend(prop={'size': 16})
    ax.set_xlabel("False Positive Rate", size=24)
    ax.set_ylabel("True Positive Rate", size=24)
    plt.savefig(args.results_dir + 'roc_curves.png', bbox_inches='tight')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/je_model/', help='path for saving trained models')
    parser.add_argument('--model', type=str, default='synth/exp7/je_model-12.pt', help='path from root to model')

    parser.add_argument('--sr', type=str, default='co')
    parser.add_argument('--subset', type=str, default='all')
    parser.add_argument('--use_softmax', type=bool, default=True)

    parser.add_argument('--embed_size', type=int, default=512, help='dimension of word embedding vectors')
    parser.add_argument('--batch_size', type=int, default=32) #32 normally
    parser.add_argument('--results_dir',type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/results/covid/')
    args = parser.parse_args()
    print(args)
    main(args)
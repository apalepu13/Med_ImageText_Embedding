import argparse
import torch
import pandas as pd

import torch.nn as nn
from CNN import *
from jointEmbedding import JointEmbeddingModel
from Pretraining import *
from Transformer import *
from torchmetrics import AUROC
import matplotlib.pyplot as plt
print("CUDA Available: " + str(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
from sklearn import metrics
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(args):
    heads = np.array(['covid19', 'Pneumonia'])
    dat = getDatasets(source=args.sr, subset=[args.subset], heads=heads)
    [DL] = getLoaders(dat, args, subset=[args.subset])

    je_vision_model, transformer, tokenizer = getSimilarityClassifier(args.model_path, args.model, heads=heads, use_covid=True)
    test_preds, test_targets = get_all_preds(DL, je_vision_model, heads, device = device)

    if args.use_softmax:
        test_preds = torch.nn.Softmax(dim=1)(test_preds)

    if args.use_diff:
        temp = test_preds[:, 0]
        test_preds[:, 0] = (test_preds[:, 0] + 1) - (test_preds[:, 1] + 1)
        test_preds[:, 1] = (test_preds[:, 1] + 1) - (temp + 1)

    test_preds = test_preds.cpu()
    test_targets = test_targets.cpu()
    fprs = {}
    tprs = {}
    thresholds = {}
    aucs = {}
    auroc = AUROC(pos_label=1)
    for i, h in enumerate(heads):
        fprs[h], tprs[h], thresholds[h] = metrics.roc_curve(test_targets[:, i].int(), test_preds[:, i])
        aucs[h] = auroc(test_preds[:, i], test_targets[:, i].int()).item()

    aucs['Total'] = np.mean(np.array([aucs[h] for h in heads]))
    print("Total AUC avg: ", aucs['Total'])
    for i, h in enumerate(heads):
        print(h, aucs[h])




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/je_model/', help='path for saving trained models')
    parser.add_argument('--model', type=str, default='synth/exp7/je_model-12.pt', help='path from root to model')

    parser.add_argument('--sr', type=str, default='co')
    parser.add_argument('--subset', type=str, default='all')
    parser.add_argument('--use_softmax', type=bool, default=False)
    parser.add_argument('--use_diff', type=bool, default=False)

    parser.add_argument('--embed_size', type=int, default=512, help='dimension of word embedding vectors')
    parser.add_argument('--batch_size', type=int, default=32) #32 normally
    parser.add_argument('--results_dir',type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/results/covid/')
    args = parser.parse_args()
    print(args)
    main(args)
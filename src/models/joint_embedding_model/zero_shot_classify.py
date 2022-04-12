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

def getLabels(df, heads, sr = 'c'):
    if sr == 'c':
        labels = None
        for i, h in enumerate(heads):
            label = df[h].float()
            if h == 'Edema' or h == 'Atelectasis':
                label[label == -1.0] = float('nan')
            else:
                label[label == -1.0] = float('nan')
            label[label == 0.0] = 0.0
            label[label == 1.0] = 1.0
            label = label.to(device)
            if labels is None:
                labels = label
                labels = labels[:, None]
            else:
                labels = torch.cat((labels, label[:, None]), axis=1)

    return labels

def getTextEmbeddings(heads, transformer, tokenizer, use_convirt = False):
    if use_convirt:
        filename = '/n/data2/hms/dbmi/beamlab/chexpert/convirt-retrieval/text-retrieval/query_custom.csv'
    else:
        filename = '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/data/mimic_label_queries.csv'

    mycsv = pd.read_csv(filename)
    if not use_convirt:
        l = []
        for h in heads:
            temp = mycsv[mycsv['Variable'] == h]
            l.append(temp.sample(n=10))
        mycsv = pd.concat(l)
    lab = mycsv['Variable']
    desc = mycsv['Text'].values
    bsize = 32
    numbatches = int(desc.shape[0]/bsize) + 1
    e = None
    for i in np.arange(numbatches):
        mydesc = desc[i*bsize:((i+1)*bsize)]
        t = tokenizer.do_encode(list(mydesc)).to(device)
        if e is None:
            e = torch.tensor(transformer(t))
        else:
            e = torch.cat((e, torch.tensor(transformer(t))), axis =0)

    head_dict = {}
    for A, B in zip(heads, np.arange(heads.shape[0])):
        head_dict[A] = B
    outlabs = torch.tensor(lab.map(head_dict).values)
    return torch.tensor(e), outlabs


#Returns an artificial probability (avg cosine sim with each label, kind of an ensemble)
def getTextSimilarities(imembeds, heads, transformer, tokenizer, textembeds=None, textlabs=None, method="mean"):

    if textembeds is None or textlabs is None:
        textembeds, textlabs = getTextEmbeddings(heads, transformer, tokenizer)
    imembeds = imembeds / imembeds.norm(dim=-1, keepdim=True)
    textembeds = textembeds / textembeds.norm(dim=-1, keepdim=True)
    cosineSimilarities = imembeds @ textembeds.t()

    head_dict = {}
    for A, B in zip(heads, np.arange(heads.shape[0])):
        head_dict[A] = B

    if method == "mean":
        p = None
        for h in heads:
            hsims = cosineSimilarities[:,  textlabs == head_dict[h]]
            hsims = torch.mean(hsims, 1, True)
            if p is None:
                p = hsims
            else:
                p = torch.cat((p, hsims), axis=1)
    elif method == "knn":
        k = 10
        distance = torch.sqrt(1-(cosineSimilarities+1)/2)
        ranksims = torch.argsort(distance, axis = 1)[:, :k]
        #print(ranksims.shape)
        #print(distance.shape)
        distance = torch.gather(distance, 1, ranksims)
        #print(distance.shape)
        ranklabs = torch.gather(textlabs.to(device).reshape(-1,).repeat(distance.shape[0], 1), 1, ranksims)
        #print(ranklabs.shape)
        weights = 1/distance
        totalDist = torch.sum(weights, axis=1).to(device)
        p = None
        for r in np.arange(ranksims.shape[0]):
            myp = torch.zeros(5).to(device)
            myrow = weights[r, :]
            myl = ranklabs[r, :]

            for i, h in enumerate(heads):
                inds = myl == head_dict[h]
                myp[i] = myrow[inds].sum()
            if p is None:
                p = (myp/totalDist[r]).reshape(1, -1)
            else:
                p = torch.cat((p, (myp/totalDist[r]).reshape(1,-1)), axis=0).to(device)

    #print(p.shape)
    return p



def main(args):
    if args.sr == 'chexpert' or args.sr == 'c':
        args.sr = 'c'
    elif args.sr == 'covid' or args.sr == 'co':
        args.sr = 'co'

    if args.subset == 'a' or args.subset == 'all':
        subset = ['all']
    elif args.subset == 't' or args.subset == 'test':
        subset = ['test']

    heads = np.array(['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'])

    dat = getDatasets(source=args.sr, subset=subset, synthetic=False)
    [DL] = getLoaders(dat, args, subset=subset)

    dat_synth = getDatasets(source=args.sr, subset=subset, synthetic=True, get_good=True, get_overwrites=True)
    DL2 = getLoaders(dat_synth, args, subset=heads)

    dat_adv = getDatasets(source=args.sr, subset=subset, synthetic=True, get_adversary=True, get_overwrites=True)
    DL3 = getLoaders(dat_adv, args, subset=heads)

    model_root_dir = args.model_path
    eval_model = args.model

    loadpath = model_root_dir + eval_model
    if device == 'cuda':
        checkpoint = torch.load(loadpath)
    else:
        checkpoint = torch.load(loadpath, map_location=torch.device('cpu'))

    je_model = JointEmbeddingModel(args.embed_size).to(device)

    je_model.load_state_dict(checkpoint['model_state_dict'], strict = False)
    if not hasattr(checkpoint['args'], 'vit') or not checkpoint['args'].vit:
        vision_model = je_model.cnn
    else:
        vision_model = je_model.vit

    transformer_model = je_model.transformer
    tokenizer = Bio_tokenizer()
    temb, tlab = getTextEmbeddings(heads, transformer_model, tokenizer)
    vision_model.eval()

    aucs = {}
    aucs_synth = {}
    aucs_adv = {}
    tprs = {}
    fprs = {}
    thresholds = {}
    auroc = AUROC(pos_label=1)

    test_preds, test_targets = None, None
    with torch.no_grad():
        for i, res in enumerate(DL):
            ims, df = res
            images = ims.to(device)
            preds = vision_model(images)
            preds = getTextSimilarities(preds, heads, transformer_model, tokenizer, temb, tlab, method=args.method)
            labels = getLabels(df, heads, args.sr)

            if test_preds is None:
                test_preds = preds
            else:
                test_preds = torch.cat((test_preds, preds), axis=0)

            if test_targets is None:
                test_targets = labels
            else:
                test_targets = torch.cat((test_targets, labels), axis=0)

    for j, h in enumerate(heads):
        test_preds_synth, test_targets_synth = None, None
        with torch.no_grad():
            for i, res in enumerate(DL2[j]):
                ims, df = res
                images = ims.to(device)
                preds = vision_model(images)
                preds = getTextSimilarities(preds, heads, transformer_model, tokenizer, temb, tlab, method=args.method)
                labels = getLabels(df, heads, args.sr)

                if test_preds_synth is None:
                    test_preds_synth = preds
                else:
                    test_preds_synth = torch.cat((test_preds_synth, preds), axis=0)

                if test_targets_synth is None:
                    test_targets_synth = labels
                else:
                    test_targets_synth = torch.cat((test_targets_synth, labels), axis=0)
        test_preds_synth = test_preds_synth.cpu()
        test_targets_synth = test_targets_synth.cpu()
        aucs_synth[h] = auroc(test_preds_synth[:, j], test_targets_synth[:, j].int()).item()

    for j, h in enumerate(heads):
        test_preds_adv, test_targets_adv = None, None
        with torch.no_grad():
            for i, res in enumerate(DL3[j]):
                ims, df = res
                images = ims.to(device)
                preds = vision_model(images)
                preds = getTextSimilarities(preds, heads, transformer_model, tokenizer, temb, tlab, method=args.method)
                labels = getLabels(df, heads, args.sr)

                if test_preds_adv is None:
                    test_preds_adv = preds
                else:
                    test_preds_adv = torch.cat((test_preds_adv, preds), axis=0)

                if test_targets_adv is None:
                    test_targets_adv = labels
                else:
                    test_targets_adv = torch.cat((test_targets_adv, labels), axis=0)
        test_preds_adv = test_preds_adv.cpu()
        test_targets_adv = test_targets_adv.cpu()
        aucs_adv[h] = auroc(test_preds_adv[:, j], test_targets_adv[:, j].int()).item()



    test_preds = test_preds.cpu()
    test_targets = test_targets.cpu()

    # print(test_targets)
    for i, h in enumerate(heads):
        fprs[h], tprs[h], thresholds[h] = metrics.roc_curve(test_targets[:, i].int(), test_preds[:, i])
        aucs[h] = auroc(test_preds[:, i], test_targets[:, i].int()).item()


    aucs['Total'] = np.mean(np.array([aucs[h] for h in heads]))
    aucs_synth['Total'] = np.mean(np.array([aucs_synth[h] for h in heads]))
    aucs_adv['Total'] = np.mean(np.array([aucs_adv[h] for h in heads]))

    print("Normal")
    print("Total AUC avg: ", aucs['Total'])
    for i, h in enumerate(heads):
        print(h, aucs[h])

    # ROC Curve
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
    plt.savefig(args.results_dir + "roc_curves.png", bbox_inches="tight")

    print("Synthetic")
    print("Total AUC avg: ", aucs_synth['Total'])
    for i, h in enumerate(heads):
        print(h, aucs_synth[h])

    print("Adversarial")
    print("Total AUC avg: ", aucs_adv['Total'])
    for i, h in enumerate(heads):
        print(h, aucs_adv[h])
    '''

    '''
    fig, ax = plt.subplots()
    x = np.arange(len(heads) + 1)
    width = .2
    ax.bar(x, aucs.values(), width, color='r', label='real test')
    ax.bar(x + width, aucs_synth.values(), width, color='b', label='synthetic test')
    ax.bar(x + 2 * width, aucs_adv.values(), width, color='g', label='adversarial test')
    ax.set_ylabel('AUC')
    ax.set_ylim(0, 1)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(aucs.keys())
    ax.set_xlabel('Class')
    ax.legend()
    if "synth" in args.model_path or "synth" in args.model:
        if args.method == 'mean':
            ax.set_title('Synthetic CLIP-Trained zero-shot AUCS')
            plt.savefig(args.results_dir + "synth_auc_zero_vtext.png", bbox_inches="tight")
        else:
            ax.set_title('Synthetic CLIP-Trained zero-shot AUCS')
            plt.savefig(args.results_dir + "synth_auc_knn_zero_vtext.png", bbox_inches="tight")
        print("Saved synth")
    else:
        if args.method == 'mean':
            ax.set_title('Real CLIP-Trained zero_shot AUCS')
            plt.savefig(args.results_dir + "real_auc_zero_vtext.png", bbox_inches="tight")
        else:
            ax.set_title('Real CLIP-Trained zero-shot AUCS')
            plt.savefig(args.results_dir + "real_auc_knn_zero_vtext.png", bbox_inches="tight")
        print("Saved real")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/je_model/', help='path for saving trained models')
    parser.add_argument('--model', type=str, default='synth/exp7/je_model-12.pt', help='path from root to model')
    parser.add_argument('--sr', type=str, default='c') #c, co
    parser.add_argument('--subset', type=str, default='test')
    parser.add_argument('--synth', type=bool, default=False, const=True, nargs='?', help='Train on synthetic dataset')
    parser.add_argument('--method', type=str, default='mean')

    #synth/exp7/je_model-12.pt
    #messed up synth exp 3
    #synth/exp2/je_model-44.pt
    #synth/exp1/je_model-76.pt

    #exp6/je_model-28.pt
    #exp5/je_model-94.pt
    #exp4/je_model-90
    #exp3/je_model-76
    #exp2/je_model-81.pt
    parser.add_argument('--embed_size', type=int, default=512, help='dimension of word embedding vectors')
    parser.add_argument('--batch_size', type=int, default=32) #32 normally
    parser.add_argument('--results_dir',type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/results/zeroshot/')
    args = parser.parse_args()
    print(args)
    main(args)
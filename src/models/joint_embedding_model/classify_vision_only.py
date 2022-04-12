import argparse
import torch
import pandas as pd
import torch.nn as nn
from CNN import *
from Vision_Transformer import *
from jointEmbedding import JointEmbeddingModel
from Pretraining import *
from Transformer import *
from torchmetrics import AUROC
import torch.nn.functional as nnf
import matplotlib.pyplot as plt
print("CUDA Available: " + str(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
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
    elif sr == 'co':
        labels = None
        for i, h in enumerate(heads):
            label = df[h].float()
            if labels is None:
                labels = label[:, None]
            else:
                labels = torch.cat((labels, label[:, None]), axis=1)

    return labels

def getTextEmbeddings(heads, transformer, tokenizer):
    filename = '/n/data2/hms/dbmi/beamlab/chexpert/convirt-retrieval/text-retrieval/query.csv'
    mycsv = pd.read_csv(filename)
    lab = mycsv['Variable']
    desc = mycsv['Text'].values
    t = tokenizer.do_encode(list(desc)).to(device)
    e = transformer(t)
    head_dict = {}
    for A, B in zip(heads, np.arange(heads.shape[0])):
        head_dict[A] = B
    outlabs = torch.tensor(lab.map(head_dict).values)
    return torch.tensor(e), outlabs


#Returns an artificial probability (avg cosine sim with each label, kind of an ensemble)
def getTextSimilarities(imembeds, heads, transformer, tokenizer, add_no_finding = False):
    if add_no_finding:
        heads = np.append(heads, np.array(['No Finding']))
        #print(heads)

    textembeds, textlabs = getTextEmbeddings(heads, transformer, tokenizer)
    imembeds = imembeds / imembeds.norm(dim=-1, keepdim=True)
    textembeds = textembeds / textembeds.norm(dim=-1, keepdim=True)
    cosineSimilarities = imembeds @ textembeds.t()

    head_dict = {}
    for A, B in zip(heads, np.arange(heads.shape[0])):
        head_dict[A] = B

    p = None
    for h in heads:
        hsims = cosineSimilarities[:,  textlabs == head_dict[h]]
        hsims = torch.mean(hsims, 1, True)
        if p is None:
            p = hsims
        else:
            p = torch.cat((p, hsims), axis=1)

    #print(p)
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

    heads = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']

    dat = getDatasets(source=args.sr, subset=subset, synthetic=False)
    [DL] = getLoaders(dat, args, subset=subset)

    dat_synth = getDatasets(source=args.sr, subset=subset, synthetic=True, get_good=True)
    [DL2] = getLoaders(dat_synth, args, subset=subset)

    dat_adv = getDatasets(source=args.sr, subset=['test'], synthetic=True, get_adversary=True, get_overwrites=True)
    DL3 = getLoaders(dat_adv, args, subset=heads)

    model_root_dir = args.model_path
    eval_model = args.model

    loadpath = model_root_dir + eval_model
    if device == 'cuda':
        checkpoint = torch.load(loadpath)
    else:
        checkpoint = torch.load(loadpath, map_location=torch.device('cpu'))


    if 'VIT' in args.model or 'VIT' in args.model_path:
        vision_model = VisionClassifier(len(heads), args.embed_size).to(device)
    else:
        cnn = CNN_Embeddings(args.embed_size).to(device)
        vision_model = CNN_Classifier(cnn, args.embed_size, num_heads = len(heads)).to(device)


    vision_model.load_state_dict(checkpoint['model_state_dict'], strict = False)

    vision_model.eval()
    aucs = {}
    aucs_synth = {}
    aucs_adv = {}
    auroc = AUROC(pos_label=1)

    test_preds, test_targets = None, None
    with torch.no_grad():
        for i, res in enumerate(DL):
            ims, df = res
            images = ims.to(device)
            preds = vision_model(images)
            #preds = nnf.softmax(preds)
            labels = getLabels(df, heads, args.sr)

            if test_preds is None:
                test_preds = preds
            else:
                test_preds = torch.cat((test_preds, preds), axis = 0)

            if test_targets is None:
                test_targets = labels
            else:
                test_targets = torch.cat((test_targets, labels), axis = 0)

    test_preds_synth, test_targets_synth = None, None
    with torch.no_grad():
        for i, res in enumerate(DL2):
            ims, df = res
            images = ims.to(device)
            preds = vision_model(images)
            labels = getLabels(df, heads, args.sr)

            if test_preds_synth is None:
                test_preds_synth = preds
            else:
                test_preds_synth = torch.cat((test_preds_synth, preds), axis=0)

            if test_targets_synth is None:
                test_targets_synth = labels
            else:
                test_targets_synth = torch.cat((test_targets_synth, labels), axis=0)

    for j, h in enumerate(heads):
        test_preds_adv, test_targets_adv = None, None
        with torch.no_grad():
            for i, res in enumerate(DL3[j]):
                ims, df = res
                images = ims.to(device)
                preds = vision_model(images)
                labels = getLabels(df, heads, args.sr)

                if test_preds_adv is None:
                    test_preds_adv = preds
                else:
                    test_preds_adv = torch.cat((test_preds_adv, preds), axis=0)

                if test_targets_adv is None:
                    test_targets_adv = labels
                else:
                    test_targets_adv = torch.cat((test_targets_adv, labels), axis=0)

        test_preds_adv = test_preds_adv.to(device)
        test_targets_adv = test_targets_adv.to(device)
        aucs_adv[h] = auroc(test_preds_adv[:, j], test_targets_adv[:, j].int()).item()

    test_preds = test_preds.to(device)
    test_targets = test_targets.to(device)
    test_preds_synth = test_preds_synth.to(device)
    test_targets_synth = test_targets_synth.to(device)

    #print(test_targets)
    for i, h in enumerate(heads):
        aucs[h] =auroc(test_preds[:, i], test_targets[:, i].int()).item()
        aucs_synth[h] = auroc(test_preds_synth[:, i], test_targets_synth[:, i].int()).item()


    aucs['Total'] = np.mean(np.array([aucs[h] for h in heads]))
    aucs_synth['Total'] = np.mean(np.array([aucs_synth[h] for h in heads]))
    aucs_adv['Total'] = np.mean(np.array([aucs_adv[h] for h in heads]))

    print("Normal")
    print("Total AUC avg: ", aucs['Total'])
    for i, h in enumerate(heads):
        print(h, aucs[h])

    print("Synthetic")
    print("Total AUC avg: ", aucs_synth['Total'])
    for i, h in enumerate(heads):
        print(h, aucs_synth[h])

    print("Adversarial")
    print("Total AUC avg: ", aucs_adv['Total'])
    for i, h in enumerate(heads):
        print(h, aucs_adv[h])


    fig, ax = plt.subplots()
    x = np.arange(len(heads) + 1)
    width = .2
    ax.bar(x, aucs.values(), width, color = 'r', label='real test')
    ax.bar(x + width, aucs_synth.values(), width, color = 'b', label='synthetic test')
    ax.bar(x + 2*width, aucs_adv.values(), width, color = 'g', label='adversarial test')
    ax.set_ylabel('AUC')
    ax.set_ylim(0, 1)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(aucs.keys())
    ax.set_xlabel('Class')
    ax.legend()
    if "synth" in args.model_path or "synth" in args.model:
        if "je_model" in args.model_path:
            if "finetuned" in args.model:
                ax.set_title('Synthetic CLIP finetuned AUCS')
                plt.savefig(args.results_dir + "synth_clip_finetuned_auc.png", bbox_inches="tight")
        else:
            if "finetuned" in args.model:
                ax.set_title('Synthetic finetuned label-trained AUCS')
                plt.savefig(args.results_dir + "synth_CNN_finetuned_auc.png", bbox_inches="tight")
            else:
                ax.set_title('Synthetic label-trained AUCS')
                plt.savefig(args.results_dir + "synth_CNN_auc.png", bbox_inches="tight")
    else:
        if "je_model" in args.model_path:
            if "finetuned" in args.model:
                ax.set_title('Real CLIP finetuned AUCS')
                plt.savefig(args.results_dir + "real_clip_finetuned_auc.png", bbox_inches="tight")
        else:
            if "finetuned" in args.model:
                ax.set_title('Real finetuned label-trained AUCS')
                plt.savefig(args.results_dir + "real_CNN_finetuned_auc.png", bbox_inches="tight")
            else:
                ax.set_title('Real label-trained AUCS')
                plt.savefig(args.results_dir + "real_CNN_auc.png", bbox_inches="tight")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='../../../../models/vision_model/vision_CNN_synthetic/', help='path for saving trained models')
    #parser.add_argument('--model_path', type=str, default='../../../../models/vision_model/vision_CNN_real/')
    #parser.add_argument('--model_path', type=str, default='../../../../models/je_model/exp6/')
    #parser.add_argument('--model_path', type=str, default='../../../../models/je_model/synth/exp7/')
    parser.add_argument('--model', type=str, default='model-14.pt', help='path from root to model')
    #parser.add_argument('--model', type=str, default='finetuned_je_model-12.pt', help='path from root to model')

    #vision_model/vision_CNN_synthetic/model-14.pt
    #vision_model/vision_VIT_real/model-50.pt
    #vision_model/vision_VIT_synthetic/model-90.pt
    #je_model/synth/exp2/finetuned_je_model-44.pt
    #je_model/exp5/finetuned_je_model-94.pt
    parser.add_argument('--sr', type=str, default='c') #c, co
    parser.add_argument('--subset', type=str, default='test')
    parser.add_argument('--synth', type=bool, default=False, const=True, nargs='?', help='Train on synthetic dataset')
    parser.add_argument('--embed_size', type=int, default=512, help='dimension of word embedding vectors')
    parser.add_argument('--batch_size', type=int, default=32) #32 normally
    parser.add_argument('--results_dir', type=str,default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/results/zeroshot/')
    args = parser.parse_args()
    print(args)
    main(args)
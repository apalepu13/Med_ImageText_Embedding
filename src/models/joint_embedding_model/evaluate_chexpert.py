import argparse
import torch
import pandas as pd
import torch.nn as nn
from CNN import *
from jointEmbedding import JointEmbeddingModel
from Pretraining import *
from Transformer import *
from torchmetrics import AUROC
print("CUDA Available: " + str(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def getTextEmbeddings(heads, transformer, tokenizer):
    filename = '/n/scratch3/users/a/anp2971/datasets/chexpert/CheXpert-v1.0-small/convirt-retrieval/text-retrieval/query.csv'
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
def getTextSimilarities(imembeds, heads, transformer, tokenizer):
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
    zeroshot = ('finetune' not in args.model)
    if zeroshot:
        subset = ['train', 'val', 'all']
    else:
        subset = ['train', 'val', 'test']

    chexpert_dat = getDatasets(source='c', subset=subset)
    train_data_loader_chexpert, val_data_loader_chexpert, test_data_loader_chexpert = getLoaders(chexpert_dat, args, subset=subset)

    model_root_dir = args.model_path
    eval_model = args.model

    loadpath = model_root_dir + eval_model
    if device == 'cuda':
        checkpoint = torch.load(loadpath)
    else:
        checkpoint = torch.load(loadpath, map_location=torch.device('cpu'))

    je_model = JointEmbeddingModel(args.embed_size).to(device)

    if not zeroshot:
        heads = ['Sex', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion',
                 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
                 'Pleural Other']
        theirheads = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']
        cnn_model = CNN_Classifier(je_model.cnn, args.embed_size, freeze=True, num_heads=len(heads)).to(device)
        cnn_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        #heads = np.array(['Atelectasis', 'Cardiomegaly', 'Edema', 'Fracture', 'Pleural Effusion', 'Pneumonia', 'Pneumothorax','No Finding'])
        heads = np.array(['Atelectasis', 'Cardiomegaly', 'Edema','Fracture', 'Pleural Effusion', 'Pneumonia', 'Pneumothorax','No Finding'])
        je_model.load_state_dict(checkpoint['model_state_dict'])
        cnn_model = je_model.cnn
        transformer_model = je_model.transformer
        tokenizer = Bio_tokenizer()

    cnn_model.eval()
    test_preds, test_targets = None, None
    with torch.no_grad():
        for i, (ims, df) in enumerate(test_data_loader_chexpert):
            images = ims.to(device)
            preds = cnn_model(images)

            if zeroshot:
                preds = getTextSimilarities(preds, heads, transformer_model, tokenizer)

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
                    labels = torch.cat((labels, label[:, None]), axis= 1)

            if test_preds is None:
                test_preds = preds
            else:
                test_preds = torch.cat((test_preds, preds), axis = 0)

            if test_targets is None:
                test_targets = labels
            else:
                test_targets = torch.cat((test_targets, labels), axis = 0)

    sums = {}
    aucs = {}
    notsums = {}
    nans = {}
    auroc = AUROC(pos_label=1)
    print(test_targets)
    for i, h in enumerate(heads):
        sums[h] = torch.sum(test_targets[:, i] == 1).int().item()
        notsums[h] = torch.sum(test_targets[:, i] == 0).int().item()
        nans[h] = torch.sum(torch.isnan(test_targets[:, i])).int().item()
        aucs[h] =auroc(test_preds[:, i], test_targets[:, i].int()).item()

    print("Total AUC avg: ", np.mean(np.array([aucs[h] for h in heads])))
    if not zeroshot:
        print("CONVIRT AUC avg: ", np.mean(np.array([aucs[h] for h in theirheads])))
    for i, h in enumerate(heads):
        print(h, aucs[h])
    for i, h in enumerate(heads):
        print(h, '1:', sums[h],'0:', notsums[h])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='../../../../models/je_model/', help='path for saving trained models')
    parser.add_argument('--model', type=str, default='exp2/je_model-81.pt', help='path from root to model')
    #exp2/je_model-81.pt
    #super_baseline, baseline, frozen, unfrozen
    parser.add_argument('--embed_size', type=int, default=512, help='dimension of word embedding vectors')
    parser.add_argument('--batch_size', type=int, default=32) #32 normally
    args = parser.parse_args()
    print(args)
    main(args)
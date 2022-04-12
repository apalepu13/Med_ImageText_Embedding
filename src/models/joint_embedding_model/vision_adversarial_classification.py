import argparse
import torch
import pandas as pd
import copy
import torch.nn as nn
from CNN import *
from Vision_Transformer import *
from jointEmbedding import JointEmbeddingModel
from Pretraining import *
from Transformer import *
from torchmetrics import AUROC
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as nnf
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


print("CUDA Available: " + str(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def getPredsTargets(myDL, v, heads, transformer_model, tokenizer, src = 'chexpert', classifier=False):
    tp, tt = None, None
    for i, res in enumerate(myDL):
        if src == 'mimic_cxr':
            im1, im2, df, study = res
            images = im1.to(device)
            images2 = im2.to(device)
            preds = v(images)
            preds = getTextSimilarities(preds, heads, transformer_model, tokenizer)
            preds2 = v(images2)
            preds2 = getTextSimilarities(preds2, heads, transformer_model, tokenizer)
            labels = getLabels(df, heads, args.sr)
            preds = torch.cat((preds, preds2), axis=0)
            labels = torch.cat((labels, labels), axis=0)
        elif src == 'chexpert':
            ims, df = res
            images = ims.to(device)
            preds = v(images)
            if classifier:
                preds = nnf.sigmoid(preds)
                preds = preds
            else:
                preds = getTextSimilarities(preds, heads, transformer_model, tokenizer)
            labels = getLabels(df, heads, args.sr)

        if tp is None:
            tp = preds
        else:
            tp = torch.cat((tp, preds), axis=0)
        if tt is None:
            tt = labels
        else:
            tt = torch.cat((tt, labels), axis=0)

    return tp, tt

def build_mim_classifier(tp, tt):
    tp = tp.cpu()
    tt = tt.cpu()
    tt = tt == 1
    clf = RandomForestClassifier()
    clf.fit(tp, tt)
    return clf

def getLabels(df, heads, sr = 'c'):
    if sr == 'c':
        labels = None
        for i, h in enumerate(heads):
            label = df[h].float()
            if h == 'Edema' or h == 'Atelectasis':
                label[label == -1.0] = 1.0 #float('nan')
            else:
                label[label == -1.0] = 0.0 #float('nan')
            label[label == 0.0] = 0.0
            label[label == 1.0] = 1.0
            label = label.to(device)
            if labels is None:
                labels = label
                labels = labels[:, None]
            else:
                labels = torch.cat((labels, label[:, None]), axis=1)

    return labels

def getTextEmbeddings(heads, transformer, tokenizer):
    filename = '/n/data2/hms/dbmi/beamlab/chexpert/convirt-retrieval/text-retrieval/query_custom.csv'
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
def getTextSimilarities(imembeds, heads, transformer, tokenizer, take_mean=True):
    textembeds, textlabs = getTextEmbeddings(heads, transformer, tokenizer)
    imembeds = imembeds / imembeds.norm(dim=-1, keepdim=True)
    textembeds = textembeds / textembeds.norm(dim=-1, keepdim=True)
    cosineSimilarities = imembeds @ textembeds.t()
    #cosineSimilarities = np.exp(np.log(1 / 0.07)) * cosineSimilarities

    head_dict = {}
    for A, B in zip(heads, np.arange(heads.shape[0])):
        head_dict[A] = B

    p = None
    for h in heads:
        hsims = cosineSimilarities[:,  textlabs == head_dict[h]]
        if take_mean:
            hsims = torch.mean(hsims, 1, True)
        if p is None:
            p = hsims
        else:
            p = torch.cat((p, hsims), axis=1)

    return p


def main(args):
    heads = np.array(['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'])
    dat = getDatasets(source=args.sr, subset=['test'], synthetic=True, get_adversary=True, heads=heads)
    dat_normal = getDatasets(source=args.sr, subset=['test'], heads=heads, synthetic=False)

    DL_synthetics = getLoaders(dat, args, subset=heads)
    [DL_real] = getLoaders(dat_normal, args, subset=['test'])

    usemimic = args.usemimic
    if usemimic:
        dat_mimic = getDatasets(source='mimic_cxr', subset=['test'], synthetic=True, heads=heads, get_text=False)
        [DLmimic] = getLoaders(dat_mimic, args, subset=['test'])

    #Vision model
    loadpath = args.model_path + args.model
    if device == 'cuda':
        checkpoint = torch.load(loadpath)
    else:
        checkpoint = torch.load(loadpath, map_location=torch.device('cpu'))

    if 'VIT' in args.model_path:
        vision_model = VisionClassifier(len(heads), args.embed_size).to(device)
    else:
        cnn = CNN_Embeddings(args.embed_size).to(device)
        vision_model = CNN_Classifier(cnn, args.embed_size, num_heads = len(heads)).to(device)
    vision_model.load_state_dict(checkpoint['model_state_dict'])
    vision_model.eval()

    #JE zero-shot model
    je_model_path = args.je_model_path + args.je_model
    if device =='cuda':
        checkpoint = torch.load(je_model_path)
    else:
        checkpoint = torch.load(je_model_path, map_location=torch.device('cpu'))

    je_model = JointEmbeddingModel(args.embed_size).to(device)
    je_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if not hasattr(checkpoint['args'], 'vit') or not checkpoint['args'].vit:
        je_vision_model = je_model.cnn
    else:
        je_vision_model = je_model.vit
    transformer_model = je_model.transformer
    tokenizer = Bio_tokenizer()

    #JE finetuned model
    finetunepath = args.je_model_path + "finetuned_" + args.je_model
    if device == 'cuda':
        checkpoint2 = torch.load(finetunepath)
    else:
        checkpoint2 = torch.load(finetunepath, map_location=torch.device('cpu'))

    finetuned = VisionClassifier(len(heads), args.embed_size).to(device)
    finetuned.load_state_dict(checkpoint2['model_state_dict'], strict=True)


    with torch.no_grad():
        if usemimic:
            mimpreds, mimtargets = getPredsTargets(DLmimic, je_vision_model, heads,
                                                   transformer_model, tokenizer, src='mimic_cxr', classifier=False)
            clf = build_mim_classifier(mimpreds, mimtargets)

        test_preds, test_targets_all = getPredsTargets(DL_real, vision_model, heads,
                                                   transformer_model, tokenizer, src='chexpert', classifier=True)
        je_test_preds, je_test_targets_all = getPredsTargets(DL_real, je_vision_model, heads,
                                               transformer_model, tokenizer, src='chexpert', classifier=False)
        f_test_preds, f_test_targets_all = getPredsTargets(DL_real, finetuned, heads,
                                                       transformer_model, tokenizer, src='chexpert', classifier=True)


    test_preds = test_preds.cpu().numpy()
    je_test_preds = je_test_preds.cpu().numpy()
    f_test_preds = f_test_preds.cpu().numpy()
    if usemimic:
        lol = [preds[:, 1][:, np.newaxis] for preds in clf.predict_proba(je_test_preds)]
        je_test_preds = np.concatenate(lol, axis = 1)
    test_targets_all = test_targets_all.cpu().numpy()
    je_test_targets_all = je_test_targets_all.cpu().numpy()
    f_test_targets_all = f_test_targets_all.cpu().numpy()

    for dnumber, dhead in enumerate(heads):
        print("HEAD", dhead)
        with torch.no_grad():
            adv_preds, adv_targets = getPredsTargets(DL_synthetics[dnumber], vision_model, heads, transformer_model, tokenizer, classifier=True)
            je_adv_preds, je_adv_targets = getPredsTargets(DL_synthetics[dnumber], je_vision_model, heads, transformer_model, tokenizer, classifier=False)
            f_adv_preds, f_adv_targets = getPredsTargets(DL_synthetics[dnumber], finetuned, heads, transformer_model, tokenizer, classifier=True)

        adv_preds = adv_preds.cpu().numpy()
        je_adv_preds = je_adv_preds.cpu().numpy()
        f_adv_preds = f_adv_preds.cpu().numpy()
        if usemimic:
            lol = [preds[:, 1][:, np.newaxis] for preds in clf.predict_proba(je_adv_preds)]
            je_adv_preds = np.concatenate(lol, axis=1)
        adv_targets = adv_targets.cpu().numpy()[:, dnumber]
        je_adv_targets = je_adv_targets.cpu().numpy()[:, dnumber]
        f_adv_targets = f_adv_targets.cpu().numpy()[:, dnumber]




        adv_yes_preds = adv_preds[adv_targets == 1, dnumber]
        adv_no_preds = adv_preds[adv_targets == 0, dnumber]
        f_adv_yes_preds = f_adv_preds[f_adv_targets==1, dnumber]
        f_adv_no_preds = f_adv_preds[f_adv_targets==0, dnumber]
        japi = je_adv_preds[:, dnumber]
        je_adv_yes_preds = japi[je_adv_targets == 1]
        je_adv_no_preds = japi[je_adv_targets == 0]

        test_targets = test_targets_all[:, dnumber]
        f_test_targets = f_test_targets_all[:, dnumber]
        je_test_targets = je_test_targets_all[:, dnumber]

        test_yes_preds = test_preds[test_targets == 1, dnumber]
        test_no_preds = test_preds[test_targets == 0,dnumber]
        f_test_yes_preds = f_test_preds[f_test_targets == 1, dnumber]
        f_test_no_preds = f_test_preds[f_test_targets == 0, dnumber]
        jtpi = je_test_preds[:, dnumber]
        je_test_yes_preds = jtpi[je_test_targets == 1]
        je_test_no_preds = jtpi[je_test_targets == 0]

        print("Vision only")
        print(np.mean(adv_yes_preds))
        print(np.mean(adv_no_preds))
        print(np.mean(test_yes_preds))
        print(np.mean(test_no_preds))
        print("JE zero shot")
        print(np.mean(je_adv_yes_preds))
        print(np.mean(je_adv_no_preds))
        print(np.mean(je_test_yes_preds))
        print(np.mean(je_test_no_preds))
        print("JE finetuned")
        print(np.mean(f_adv_yes_preds))
        print(np.mean(f_adv_no_preds))
        print(np.mean(f_test_yes_preds))
        print(np.mean(f_test_no_preds))

        plt.figure(figsize = (8, 8))
        if "synth" in args.model_path:
            plt.title("Predicted " + dhead + " probabilities synth-trained vision model")
        else:
            plt.title("Predicted " + dhead + " probabilities real-trained vision model")
        plt.boxplot([adv_yes_preds, adv_no_preds, test_yes_preds,  test_no_preds],
                    labels=["+watermark, + label", "+watermark, - label", "no watermark, + label", "no watermark, - label"])
        if "synth" in args.model_path:
            plt.savefig(args.results_dir + dhead + "_vision.png", bbox_inches='tight')
        else:
            plt.savefig(args.results_dir + dhead + "_vision_baseline.png", bbox_inches='tight')

        plt.figure(figsize=(8,8))
        if "synth" in args.je_model_path:
            plt.title("Predicted " + dhead + " zeroshot output synth-trained vis/text model")
        else:
            plt.title("Predicted " + dhead + " zeroshot output real-trained vis/text model")
        plt.boxplot([je_adv_yes_preds, je_adv_no_preds, je_test_yes_preds,  je_test_no_preds],
                    labels=["+watermark, + label", "+watermark, - label", "no watermark, + label", "no watermark, - label"])
        if "synth" in args.je_model_path:
            plt.savefig(args.results_dir + dhead + "_vtext.png", bbox_inches='tight')
        else:
            plt.savefig(args.results_dir + dhead + "_vtext_baseline.png", bbox_inches='tight')

        plt.figure(figsize=(8, 8))
        if "synth" in args.je_model_path:
            plt.title("Predicted " + dhead + " probabilities linear probe synth-trained vis/text model")
        else:
            plt.title("Predicted " + dhead + " probabilties linear probe real-trained vis/text model")
        plt.boxplot([f_adv_yes_preds, f_adv_no_preds, f_test_yes_preds, f_test_no_preds],
                    labels=["+watermark, + label", "+watermark, - label", "no watermark, + label",
                            "no watermark, - label"])
        if "synth" in args.je_model_path:
            plt.savefig(args.results_dir + dhead + "_finetuned_vtext.png", bbox_inches='tight')
        else:
            plt.savefig(args.results_dir + dhead + "_finetuned_vtext_baseline.png", bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--je_model_path', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/je_model/exp5/')
    parser.add_argument('--model_path', type=str, default='../../../../models/vision_model/vision_VIT_real/', help='path for saving trained models')
    parser.add_argument('--je_model', type=str, default='je_model-94.pt') ##44 for synth, ##94 for not
    parser.add_argument('--model', type=str, default='model-50.pt', help='path from root to model')
    parser.add_argument('--results_dir', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/results/')
    #vision_VIT_synthetic/model-90.pt
    #synth/exp2/je_model-44.pt
    parser.add_argument('--sr', type=str, default='c') #c, co
    parser.add_argument('--subset', type=str, default='test')
    parser.add_argument('--synth', type=bool, default=False, const=True, nargs='?', help='Train on synthetic dataset')
    parser.add_argument('--usemimic', type=bool, default=False, const=True, nargs='?', help='Use mimic to alter zeroshot')
    parser.add_argument('--embed_size', type=int, default=512, help='dimension of word embedding vectors')
    parser.add_argument('--batch_size', type=int, default=32) #32 normally
    args = parser.parse_args()
    print(args)
    main(args)
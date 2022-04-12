import argparse
import torch
import pandas as pd
import copy
import torch.nn as nn
from CNN import *
from Vision_Transformer import *

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from jointEmbedding import JointEmbeddingModel
from Pretraining import *
from Transformer import *
import matplotlib.pyplot as plt
import numpy as np


print("CUDA Available: " + str(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category
    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]

class EmbeddingOutputTarget:
    def __init__(self, category, text_embed, text_labs):
        self.category = category
        t = text_embed[text_labs == category, :]
        self.t = torch.mean(t, 0)
    def __call__(self, output):
        output_feat = output / output.norm(dim=-1, keepdim=True)
        output_feat = torch.squeeze(output_feat)
        tembed = self.t / self.t.norm(dim=-1, keepdim=True)
        class_score = torch.dot(output_feat, tembed)
        return class_score


def getSaliency(img, vis_model, device, args,head_choice = None, camje=None, camvision=None, text_embed = None, text_labs = None,
                labs = None, text=None, heads=None,tokenizer=None, transformer_model=None, embedding=False, mod=None, synth=False):
    img.to(device)
    img.requires_grad_()
    output = vis_model(img)
    myi = 4
    if embedding and text:
        text_tokens = tokenizer.do_encode(text)
        text_embed = transformer_model(text_tokens)
        output_feat = output/output.norm(dim=-1, keepdim=True).squeeze()
        text_embed = text_embed/text_embed.norm(dim=-1, keepdim=True).squeeze()
        class_score = torch.dot(output_feat, text_embed)
    elif embedding:
        for i, h in enumerate(heads):
            if h == head_choice:
                myi = i
                t = text_embed[text_labs == i, :]
                t = torch.mean(t, 0)

        output_feat = output/output.norm(dim=-1, keepdim=True)
        output_feat = torch.squeeze(output_feat)
        tembed = t/t.norm(dim=-1, keepdim=True)
        class_score = torch.dot(output_feat, tembed)

    else:
        for i, h in enumerate(heads):
            if h == head_choice:
                class_score = output[:, i]
                myi = i

    class_score.backward()
    if args.gradcam:
        if embedding:
            gradc = camje(input_tensor = img, targets = [EmbeddingOutputTarget(myi, text_embed, text_labs)])
            gradc = gradc[0, :]
            #gradc = (gradc < (gradc[:, None]) * 1.0).mean(axis=1)
        else:
            gradc = camvision(input_tensor = img, targets = [ClassifierOutputTarget(myi)])
            gradc = gradc[0, :]
            #gradc = (gradc < (gradc[:, None]) * 1.0).mean(axis=1)
    saliency, _ = torch.max(img.grad.data.abs(), dim=1) #1
    saliency = saliency.reshape(224, 224) * 1.0
    #saliency = ((saliency < (saliency[:, None])) *1.0).mean(axis=1)
    img.requires_grad_(False)
    img = img.reshape(-1, 224, 224)
    img[0, :, :] = (img[0, :, :] * .229) + .485
    img[1, :, :] = (img[1, :, :] * .224) + .456
    img[2, :, :] = (img[2, :, :] * .225) + .406

    if mod:
        if args.gradcam:
            fig, ax = plt.subplots(1, 3)
        else:
            fig, ax = plt.subplots(1,2)

        ax[0].imshow(img.cpu().detach().numpy().transpose(1, 2, 0))
        ax[0].axis('off')
        ax[1].imshow(saliency.cpu(), cmap='hot')
        ax[1].axis('off')
        if args.gradcam:
            ax[2].imshow(gradc, cmap='hot')
        plt.tight_layout()

        if synth:
            if args.gradcam:
                fig.suptitle(str(head_choice) + " class saliency/gradcam for " + labs + " positive synthetic image")
            else:
                fig.suptitle(str(head_choice) + " class saliency for " + labs + " positive synthetic image")
            plt.savefig(args.results_dir + mod + ".png", bbox_inches='tight')
        else:
            if args.gradcam:
                fig.suptitle(str(head_choice) + " class saliency/gradcam for " + labs + " positive real image")
            else:
                fig.suptitle(str(head_choice) + " class saliency for " + labs + " positive real image")
            plt.savefig(args.results_dir + mod + ".png", bbox_inches='tight')

    return saliency

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
            l.append(temp.sample(n=50))
        mycsv = pd.concat(l)
    temp_list = [mycsv[mycsv['Variable'] == h] for h in heads]
    temp_list_lim = [t.iloc[:20, :] for t in temp_list]
    mycsv = pd.concat(temp_list_lim)

    lab = mycsv['Variable']
    desc = mycsv['Text'].values
    bsize = 32
    numbatches = int(desc.shape[0] / bsize) + 1
    e = None
    for i in np.arange(numbatches):
        mydesc = desc[i * bsize:((i + 1) * bsize)]
        t = tokenizer.do_encode(list(mydesc)).to(device)
        if e is None:
            e = torch.tensor(transformer(t))
        else:
            e = torch.cat((e, torch.tensor(transformer(t))), axis=0)

    head_dict = {}
    for A, B in zip(heads, np.arange(heads.shape[0])):
        head_dict[A] = B
    outlabs = torch.tensor(lab.map(head_dict).values)
    return torch.tensor(e), outlabs


def main(args):
    heads = np.array(['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'])
    dat_good = getDatasets(source=args.sr, subset=['test'], heads=heads, synthetic=True, get_good=True)
    dat_bad = getDatasets(source=args.sr, subset=['test'], heads=heads, synthetic=True, get_adversary=True)
    dat_normal = getDatasets(source=args.sr, subset=['test'], heads=heads, synthetic=False)
    [DL_synthetics] = getLoaders(dat_good, args, subset=['test'], shuffle=False)
    [DL_adversarial] = getLoaders(dat_bad, args, subset=['test'], shuffle=False)
    [DL_real] = getLoaders(dat_normal, args, subset=['test'], shuffle=False)


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
    camje = None
    camvision=None
    if args.gradcam and 'VIT' not in args.model_path:
        target_layers = [vision_model.cnn_model.resnet.layer4[-1]]
        camvision = GradCAM(model=vision_model, target_layers=target_layers, use_cuda=torch.cuda.is_available())

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
    je_vision_model.eval()#synth/exp2/je_model-44.pt
    text_embed, text_labs = getTextEmbeddings(heads, transformer_model, tokenizer)

    if args.gradcam and not checkpoint['args'].vit:
        target_layers = [je_vision_model.resnet.layer4[-1]]
        camje = GradCAM(model=je_vision_model, target_layers=target_layers, use_cuda=torch.cuda.is_available())

    if args.gradcam:
        typ = "sal"
    else:
        typ = "sal"

    for dnumber, dhead in enumerate(heads):
        for i, res in enumerate(DL_real):
            if i < 100:
                continue
            myims, df = res
            if df[dhead] == 1:
                for h in heads:
                    if h != dhead and df[h] == 1:
                        continue
            else:
                continue

            for dh, h in enumerate(heads):
                ims = myims.clone()
                sal_vis = getSaliency(img = ims.to(device), vis_model=vision_model, device=device, args=args,
                              head_choice = h, labs = dhead, heads=heads,text_embed = text_embed, text_labs = text_labs,tokenizer=None, transformer_model=None,
                              embedding=False, mod="vision" + dhead + "_pos_" + h + "_" + typ, camje = camje, camvision=camvision)
                ims = myims.clone()
                sal_je = getSaliency(img = ims.to(device), vis_model=je_vision_model, device=device, args=args,
                              head_choice = h, labs = dhead, heads=heads,text_embed = text_embed, text_labs = text_labs,tokenizer=tokenizer, transformer_model=transformer_model,
                              embedding=True, mod="vtext" + dhead + "_pos_" + h + "_" + typ, camje=camje, camvision=camvision)
            if df[dhead] == 1:
                break

    for dnumber, dhead in enumerate(heads):
        for i, res in enumerate(DL_synthetics[dnumber]):
            if i < 100:
                continue
            myims, df = res
            if df[dhead] == 1:
                for h in heads:
                    if h != dhead and df[h] == 1:
                        continue
            else:
                continue
            for dh, h in enumerate(heads):
                myims, df = DL_synthetics[dh].dataset[i]
                myims = myims.reshape(1, 3, 224, 224)
                ims = myims.clone()
                sal_vis = getSaliency(img=ims.to(device), vis_model=vision_model, device=device, args=args,
                                  head_choice = h, labs=dhead, heads=heads,text_embed = text_embed, text_labs = text_labs, tokenizer=None, transformer_model=None,
                                  embedding=False, mod="vision" + dhead + "_synth" + h + "_" + typ, synth=True, camje=camje, camvision=camvision)
                ims = myims.clone()
                sal_je = getSaliency(img=ims.to(device), vis_model=je_vision_model, device=device, args=args,
                                 head_choice = h, labs=dhead, heads=heads,text_embed = text_embed, text_labs = text_labs, tokenizer=tokenizer, transformer_model=transformer_model,
                                 embedding=True, mod="vtext" + dhead + "_synth" + h + "_" + typ, synth=True, camje=camje, camvision=camvision)
            if df[dhead] == 1: #if the shortcut doesn't match the label
                break




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--je_model_path', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/je_model/synth/exp4/')
    parser.add_argument('--model_path', type=str, default='../../../models/vision_model/vision_CNN_synthetic/', help='path for saving trained models')
    parser.add_argument('--je_model', type=str, default='je_model-12.pt')
    parser.add_argument('--model', type=str, default='model-14.pt', help='path from root to model')
    parser.add_argument('--results_dir', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/results/saliency/')
    parser.add_argument('--gradcam', type=bool, default=False, const=True, nargs='?')
    #vision_VIT_synthetic/model-90.pt
    #vision_CNN_synthetic/model-2.pt
    #synth/exp2/je_model-44.pt
    #synth/exp4/je_model-12.pt
    parser.add_argument('--sr', type=str, default='c') #c, co
    parser.add_argument('--subset', type=str, default='test')
    parser.add_argument('--synth', type=bool, default=False, const=True, nargs='?', help='Train on synthetic dataset')
    parser.add_argument('--usemimic', type=bool, default=False, const=True, nargs='?', help='Use mimic to alter zeroshot')
    parser.add_argument('--embed_size', type=int, default=512, help='dimension of word embedding vectors')
    parser.add_argument('--batch_size', type=int, default=1) #32 normally
    args = parser.parse_args()
    print(args)
    main(args)
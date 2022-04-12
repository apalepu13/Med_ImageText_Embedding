import sys
sys.path.insert(0, '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/src/models/joint_embedding_model/')
from Data_Loading import *
import torch
print("CUDA Available: " + str(torch.cuda.is_available()))
import argparse
from CNN import *
from Vision_Transformer import *
from jointEmbedding import JointEmbeddingModel
from Pretraining import *
from Transformer import *
import matplotlib.pyplot as plt
import numpy as np

print("CUDA Available: " + str(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def plotProbDists(im_list, model, heads, mod_name, df, im_number=None):
    with torch.no_grad():
        myfig, myax = plt.subplots(2, 3, figsize=(12, 8), sharey=True)
        labstr = "Labels: "
        orig = im_list[0].clone().to(device)
        origpred = model(orig)
        if mod_name == 'vision':
            origpred = torch.nn.Sigmoid()(origpred)
        else:
            origpred = origpred
        print(np.array(heads))
        height = np.squeeze(origpred.cpu().numpy())
        print(height.shape)
        myax[0,0].bar(np.array(heads), height)
        for i,h in enumerate(heads):
            if df:
                if df[h] == 1:
                    labstr += h
                    labstr += ", "
            img = im_list[i+1].clone().to(device)
            predsOut = model(img)
            if mod_name=='vision':
                predsOut = torch.nn.Sigmoid()(predsOut)
            else:
                predsOut=predsOut
            myax[(i+1)//3, (i+1)%3].bar(np.array(heads), np.squeeze(predsOut.cpu().numpy()))
            myax[(i+1)//3, (i+1)%3].set_title(h + " shortcut")
        myfig.suptitle("Predicted logits" + ", " + mod_name + "model")
        if labstr == "Labels: ":
            labstr = "No Finding"
        myax[0, 0].set_title(labstr)
        plt.savefig(args.results_dir + "Predicted_probs_" + h + "_" + mod_name + "_" + str(im_number) + ".png", bbox_inches='tight')



def main(args):
    heads = np.array(['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'])
    dat_normal = getDatasets(source=args.sr, subset=['test'], heads=heads, synthetic=False)
    dat_overwrites = getDatasets(source=args.sr, subset=['test'], heads=heads, synthetic=True, get_overwrites=True)
    [loader_normal] = getLoaders(dat_normal, subset=['test'])
    loader_synths = getLoaders(dat_overwrites, subset=heads)
    dat_normal = dat_normal['test']


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
    transformer_model = je_model.transformer
    tokenizer = Bio_tokenizer()
    if not hasattr(checkpoint['args'], 'vit') or not checkpoint['args'].vit:
        cnn = je_model.cnn
        je_vision_model = CNN_Similarity_Classifier(cnn_model=cnn, transformer_model=transformer_model, tokenizer=tokenizer,
                                                    device = device, avg_embedding=True, get_num=1)
    else:
        je_vision_model = je_model.vit

    je_vision_model.to(device)
    je_vision_model.eval()

    im_number = 99#79, 93

    normIm, normDf = dat_normal.__getitem__(im_number)
    normIm = normIm.reshape(1, 3, 224, 224)
    im_list = [normIm]
    for h in heads:
        myim, mydf = dat_overwrites[h].__getitem__(im_number)
        myim = myim.reshape(1,3,224,224)
        im_list.append(myim)

    plotProbDists(im_list, vision_model, heads, "vision", normDf, im_number)
    plotProbDists(im_list, je_vision_model, heads, "clip", normDf, im_number)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--je_model_path', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/je_model/synth/exp7/')
    parser.add_argument('--model_path', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/vision_model/vision_CNN_synthetic/', help='path for saving trained models')
    parser.add_argument('--je_model', type=str, default='je_model-12.pt')
    parser.add_argument('--model', type=str, default='model-12.pt', help='path from root to model')
    parser.add_argument('--results_dir', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/results/prob_dists/')
    parser.add_argument('--sr', type=str, default='c') #c, co
    parser.add_argument('--subset', type=str, default='test')
    parser.add_argument('--embed_size', type=int, default=512, help='dimension of word embedding vectors')
    parser.add_argument('--batch_size', type=int, default=1) #32 normally
    args = parser.parse_args()
    print(args)
    main(args)
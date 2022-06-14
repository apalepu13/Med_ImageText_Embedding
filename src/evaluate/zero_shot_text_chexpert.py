import argparse
import torch
import numpy as np
import sys
sys.path.insert(0, '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/src/models/joint_embedding_model/')
from CNN import *
from jointEmbedding import JointEmbeddingModel
from HelperFunctions import *
import pandas as pd
from Transformer import *
print("CUDA Available: " + str(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#qe = embeddings (num queries * esize)
#ce = embeddings (num candidates * esize)
#ql, cl are corresponding labels for queries and candidates

def getChexEmbeddings(dat_loader, embedding_model):
    theirheads = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']
    with torch.no_grad():
        for i, (ims, df) in enumerate(dat_loader):
            label = torch.zeros(ims.shape[0]) - 2
            for i, h in enumerate(theirheads):
                btorch = df[h]
                for j in np.arange(btorch.shape[0]):
                    if btorch[j] > 0:
                        assert label[j] < -1
                        label[j] = i
            try:
                myL = torch.cat((myL, label), axis=0)
            except:
                myL = label

            images = ims.to(device)
            try:
                myE = torch.cat((myE, embedding_model(images)), axis=0)
            except:
                myE = embedding_model(images)

    return myE, myL

def getTextEmbeddings(heads, transformer, tokenizer, use_convirt=False):
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
    for A, B in zip(heads, np.arange(np.array(heads).shape[0])):
        head_dict[A] = B
    outlabs = torch.tensor(lab.map(head_dict).values)
    print(e.shape)
    print(outlabs.shape)
    return torch.tensor(e), outlabs


#return the query labels, and corresponding descending ordered candidate labels ranked by cosine similarity
def getCandidates(qe, ce, ql, cl):
    qe = qe / qe.norm(dim=-1, keepdim=True)
    ce = ce / ce.norm(dim=-1, keepdim=True)
    cosineSimilarities = qe @ ce.t()
    c_rank = torch.argsort(cosineSimilarities, dim=-1, descending=True)
    lab_out = cl[c_rank]
    return ql, lab_out

#Get precision at a certain number of candidates
def precAt(query_labels, candidate_labels, k=1):
    c = candidate_labels[:, :k]
    correct = torch.eq(query_labels[:, None], c)
    return np.round(torch.mean(correct.float()).item(), 4)

def main(args):
    subset = ['candidates']
    chexpert_dat = getDatasets(source='c', subset=subset)
    [candidate_data_loader_chexpert] = getLoaders(chexpert_dat, args, subset=subset)

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
        cnn_model = je_model.cnn
    else:
        cnn_model = je_model.vit

    transformerModel = je_model.transformer
    tokenizer = Bio_tokenizer()
    heads = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']

    query_embeddings, query_labels = getTextEmbeddings(heads, transformerModel, tokenizer, use_convirt=True)
    candidate_embeddings, candidate_labels = getChexEmbeddings(candidate_data_loader_chexpert, cnn_model)

    query_labels, candLabels = getCandidates(query_embeddings, candidate_embeddings, query_labels, candidate_labels)
    print("Zero shot text-image retrieval")
    print("k =  1:",precAt(query_labels, candLabels, 1))
    print("k =  5:",precAt(query_labels, candLabels, 5))
    print("k = 10:",precAt(query_labels, candLabels, 10))
    print("k = 50:",precAt(query_labels, candLabels, 50))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/je_model/', help='path for saving trained models')
    parser.add_argument('--model', type=str, default='exp6/je_model-28.pt', help='path from root to model')
    #exp5/je_model-94.pt
    #exp4/je_model-90.pt
    #exp3/je_model-76.pt
    #exp2/finetuned/chexpert_unfrozen.pt
    #exp2/je_model-81.pt
    #super_baseline, baseline, frozen, unfrozen
    parser.add_argument('--embed_size', type=int, default=512, help='dimension of word embedding vectors')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    print(args)
    main(args)
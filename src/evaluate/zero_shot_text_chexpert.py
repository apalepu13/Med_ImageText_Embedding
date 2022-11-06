import argparse
import sys
sys.path.insert(0, '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/src/models/Patch_CLIP/')
import torch
import CLIP_Embedding
import MedDataHelpers
import utils
import numpy as np
print("CUDA Available: " + str(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#return the query labels, and corresponding descending ordered candidate labels ranked by cosine similarity
def getCandidates(qes, ce, cl, heads):
    cosineSimilarities = []
    ce = ce / ce.norm(dim=-1, keepdim=True)
    qls = []
    qeNF = qes['No Finding'].cpu()
    qeNF = qeNF/ qeNF.norm(dim=-1, keepdim=True)
    qeNF = qeNF.mean(dim=0, keepdim=False)[None, :]
    for i, h in enumerate(heads):
        qe = qes[h].cpu()
        #myNF = qeNF.repeat() #subtract from qe to "NO FINDING" normalize
        qe = qe / qe.norm(dim=-1, keepdim=True)
        cosineSimilarities.append(qe @ ce.t())
        for j in np.arange(qe.shape[0]):
            qls.append(i)

    qls = np.array(qls)
    qls = torch.tensor(qls)
    cosineSimilarities = torch.cat(cosineSimilarities, dim=0)
    c_rank = torch.argsort(cosineSimilarities, dim=-1, descending=True)
    cl = torch.argmax(torch.nan_to_num(cl), dim = 1)
    lab_out = cl[c_rank]
    return qls, lab_out #query labels, sorted cand labels per query label

#Get precision at a certain number of candidates
def precAt(query_labels, candidate_labels, k=1):
    c = candidate_labels[:, :k]
    correct = torch.eq(query_labels[:, None], c)
    return np.round(torch.mean(correct.float()).item(), 4)

def main(args):
    heads1 = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion', 'No Finding']
    heads2 = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']
    subset = ['candidates']
    clip_model = CLIP_Embedding.getCLIPModel(args.je_model_path)
    filters = MedDataHelpers.getFilters(args.je_model_path)
    chexpert_dat = MedDataHelpers.getDatasets(source='c', subset=subset, filters = filters)
    DLS = MedDataHelpers.getLoaders(chexpert_dat, args, shuffle=False)
    DL = DLS[subset[0]]

    query_embeddings = CLIP_Embedding.getLabelEmbeddings(clip_model, heads1, avg = False)
    candidate_embeddings, candidate_labels = utils.get_all_preds(DL, clip_model, im_embeds=True, heads = heads2)
    query_labels, candLabels = getCandidates(query_embeddings, candidate_embeddings[0], candidate_labels, heads2)
    print("Zero shot text-image retrieval")
    print("k =  1:",precAt(query_labels, candLabels, 1))
    print("k =  5:",precAt(query_labels, candLabels, 5))
    print("k = 10:",precAt(query_labels, candLabels, 10))
    print("k = 50:",precAt(query_labels, candLabels, 50))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--je_model_path', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/clip_regularized/exp1/', help='path for saving trained models')
    parser.add_argument('--embed_size', type=int, default=128, help='dimension of word embedding vectors')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    print(args)
    main(args)
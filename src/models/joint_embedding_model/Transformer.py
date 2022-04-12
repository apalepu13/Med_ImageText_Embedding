import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np

def getTextEmbeddings(heads, transformer, tokenizer, use_convirt = False, device = 'cuda', get_num=1, use_covid=False):
    if use_convirt:
        filename = '/n/data2/hms/dbmi/beamlab/chexpert/convirt-retrieval/text-retrieval/query_custom.csv'
    elif use_covid:
        filename= '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/data/covid_queries.csv'
    else:
        filename = '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/data/mimic_label_queries.csv'
    mycsv = pd.read_csv(filename)
    if not use_convirt:
        l = []
        for h in heads:
            temp = mycsv[mycsv['Variable'] == h]
            l.append(temp.sample(n=get_num))
        mycsv = pd.concat(l)

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

class Bio_tokenizer():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    def do_encode(self, texts):
        texts = [t for t in texts]
        encodings = self.tokenizer(texts, padding=True, truncation=True, max_length = 256, return_tensors="pt")
        return encodings

class Transformer_Embeddings(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        modules = [self.model.embeddings, *self.model.encoder.layer[:8]] #freeze bottom 8 layers only
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

        #print(self.model)
        self.linear1 = nn.Linear(768, self.embed_dim) #fc layer to embed dim
    def forward(self, text):
        #embeddings = self.model(text['input_ids'], text['token_type_ids'], text['attention_mask'])
        embeddings = self.model(input_ids = text['input_ids'], attention_mask = text['attention_mask'])
        embeddings = self.linear1(embeddings.pooler_output)
        return embeddings

if __name__ == '__main__':
    bt = Bio_tokenizer()
    te = Transformer_Embeddings(512)
    toks = bt.do_encode(["Hi my name is Anil. I am a good friend."])
    embeds = te(toks)
    print(te)
    print(toks)

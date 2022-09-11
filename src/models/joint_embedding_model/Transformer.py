import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def getTextEmbeddings(heads, transformer, tokenizer, use_convirt = False, device = 'cuda', get_num=1):
    if use_convirt:
        filename = '/n/data2/hms/dbmi/beamlab/chexpert/convirt-retrieval/text-retrieval/query_custom.csv'
    else:
        filename = '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/data/mimic_label_queries.csv'

    covid_filename = '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/data/covid_queries.csv'
    covidcsv = pd.read_csv(covid_filename)
    zeroshots = ['covid19', 'Pneumonia', 'No Finding', 'Lungs', 'LeftLung', 'RightLung']

    mycsv = pd.read_csv(filename)
    if not use_convirt:
        l = []
        for h in heads:
            if h in zeroshots:
                temp = covidcsv[covidcsv['Variable'] == h]
                l.append(temp.sample(n=get_num, replace=True))
            else:
                temp = mycsv[mycsv['Variable'] == h]
                l.append(temp.sample(n=get_num, replace=True))
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


class CNN_Similarity_Classifier(nn.Module):
    def __init__(self, cnn_model, transformer_model, tokenizer,
                 heads=np.array(['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'])
                 , use_convirt=False, get_num=20, avg_embedding=True, soft=False):

        super().__init__()
        self.cnn_model = cnn_model
        self.transformer_model = transformer_model
        self.tokenizer = tokenizer
        self.heads = heads
        self.device = device
        self.tembed, self.tlab = getTextEmbeddings(heads=self.heads, transformer=self.transformer_model,
                                                               tokenizer=self.tokenizer,
                                                               use_convirt=use_convirt, device=device,
                                                               get_num=get_num)
        self.get_num = get_num
        self.avg_embedding = avg_embedding
        self.softmax = nn.Softmax(dim=1)
        self.soft = soft

    def forward(self, image):
        with torch.no_grad():
            embedding = self.cnn_model(image)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            class_score = torch.zeros(embedding.shape[0], self.heads.shape[0]).to(self.device)
            for i, h in enumerate(self.heads):
                t = self.tembed[self.tlab == i, :]
                tembed = t / t.norm(dim=-1, keepdim=True)
                if self.get_num > 1:
                    tembed = tembed.mean(dim=0)
                tembed = tembed / tembed.norm(dim=-1, keepdim=True)
                head_sim = embedding @ tembed.t()
                head_sim = head_sim.squeeze()
                class_score[:, i] = head_sim

            return (self.softmax(class_score) if self.soft else class_score)


class Report_Tokenizer():
    def __init__(self):
        url = "microsoft/BiomedVLP-CXR-BERT-specialized"
        self.tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True)
    def encode(self, texts):
        texts = [t for t in texts]
        encodings = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=texts,
                                    add_special_tokens=True,
                                    padding=True, truncation=True, max_length=256,
                                    return_tensors='pt')
        return encodings

class Transformer_Embeddings(nn.Module):
    def __init__(self, embed_dim = 128, cls_only = False, args=None):
        super().__init__()
        self.embed_dim = embed_dim
        url = "microsoft/BiomedVLP-CXR-BERT-specialized"
        self.txt_layernorm = nn.LayerNorm(768)
        self.linear1 = nn.Linear(768, 128)
        self.model = AutoModel.from_pretrained(url, trust_remote_code=True)
        self.cls_only = cls_only
        modules = [self.model.bert.embeddings, *self.model.bert.encoder.layer[:8]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, text):
        if not self.cls_only:
            outputs = self.model(input_ids=text.input_ids,attention_mask=text.attention_mask,output_cls_projected_embedding=True, return_dict=True)
            token_embeddings = outputs.last_hidden_state
            token_embeddings = self.txt_layernorm(token_embeddings)
            token_embeddings = self.linear1(token_embeddings)
            return token_embeddings
        else:
            embeddings = self.model.get_projected_text_embeddings(input_ids=text.input_ids,attention_mask=text.attention_mask)
            return embeddings


if __name__ == '__main__':
    bt = Report_Tokenizer()
    te = Transformer_Embeddings(512)
    toks = bt.do_encode(["Hi my name is Anil. I am a nice guy."])
    embeds = te(toks)
    print(te)
    print(toks)

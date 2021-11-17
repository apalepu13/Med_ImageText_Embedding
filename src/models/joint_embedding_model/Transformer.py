import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
import time

class Bio_tokenizer():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    def do_encode(self, texts):
        #encodings = [torch.tensor([self.tokenizer.encode(t)]) for t in texts]
        texts = [t for t in texts]
        encodings = self.tokenizer(texts, padding=True, truncation=True, max_length = 512, return_tensors="pt")
        return encodings

class Transformer_Embeddings(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        for param in self.model.parameters(): #freezing transformer model
            param.requires_grad = False
        self.linear1 = nn.Linear(768, self.embed_dim) #except the fc layer
    def forward(self, text):
        #t = time.time()
        embeddings = self.model(text['input_ids'], text['token_type_ids'], text['attention_mask'])
        embeddings = self.linear1(embeddings.pooler_output)
        #print("transformer batch time " + str(time.time() - t))
        return embeddings

from torch import nn
import torch
import CNN
import Transformer
import numpy as np
from HelperFunctions import *

class JointEmbeddingModel(nn.Module):
    def __init__(self, embed_dim, args=None):
        super().__init__()
        self.cnn = CNN.CNN_Embeddings(embed_dim=embed_dim, freeze=False, args=args)
        self.transformer = Transformer.Transformer_Embeddings(embed_dim=embed_dim, args=args)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.embed_dim = embed_dim

    def similarity_matrix(self, emb1, emb2):
        image_features = emb1 / emb1.norm(dim=-1, keepdim=True)  # N E
        text_features = emb2 / emb2.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        return logits_per_image

    def forward(self, images, text):
        token_text = self.transformer(text)  # N T E, #batch, tokens, embeddings
        global_text = token_text[:, 0, :].view(-1, self.embed_dim) # N E
        #token_text = token_text/token_text.norm(dim=2, keepdim=True)  # N T
        im_features, im_logits, cross_weights = [], [], []
        token_im = self.cnn(images)
        token_im = token_im.permute(0, 2, 1)
        #token_im = token_im/token_im.norm(dim=2, keepdim=True)
        N, T, E = token_text.shape
        cross_weights_text = torch.bmm(token_text.repeat(len(images),1,1), token_im.permute(0, 2, 1)) #NTP
        for i, im in enumerate(images):
            global_im = torch.mean(token_im[(i*N):(i*N + N), :, :], dim=1)
            im_features.append(global_im)
            im_logits.append(self.similarity_matrix(global_im, global_text))
            cross_weights.append(cross_weights_text[(i*N):(i*N + N), :, :])

        aug_logits = None
        if len(images) > 1:
            aug_logits = []
            for i in np.arange(len(images)):
                for j in np.arange(len(images)):
                    if i <= j:
                        continue
                    imsims = self.similarity_matrix(im_features[i], im_features[j])
                    aug_logits.append(imsims)
        return im_logits, cross_weights, aug_logits #list per im, #list per im, #list per im-im pair

def getJEModel(path, embed_dim=128, mod = 'best_model.pt'):
    je_model = JointEmbeddingModel(embed_dim)
    checkpoint = torch.load(path + mod, map_location=torch.device(device))
    je_model.load_state_dict(checkpoint['model_state_dict'])
    return je_model

def getPatchEmbedder(je_model):
    return je_model.cnn
def getGlobalEmbedder(je_model):
    cnn =  je_model.cnn
    cnn.pool = True
    return cnn
def getTokenEmbedder(je_model):
    return je_model.transformer
def getTextEmbedder(je_model):
    transformer = je_model.transformer
    transformer.cls_only=True
    return transformer
def getCNNClassifier(je_model, embed_size=128):
    cnn = je_model.cnn
    classifier = CNN.CNN_Classifier(embed_size, cnn)
    return classifier
def getSimClassifier(je_model, heads, tokenizer = Transformer.Report_Tokenizer(),
                     use_convirt=False, get_num=20, avg_embedding=True, soft=False):
    cnn = getGlobalEmbedder(je_model)
    transformer = getTextEmbedder(je_model)
    classifier = Transformer.CNN_Similarity_Classifier(cnn, transformer, tokenizer, heads = heads,
                                                       use_convirt = use_convirt, get_num = get_num,
                                                       avg_embedding = avg_embedding, soft = soft)
    return classifier










if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    je_model = JointEmbeddingModel(embed_dim=128, use_attn=True).to(device)
    mimic_dat = getDatasets(source='m', subset=['tinytrain'], augs=2)
    [train_data_loader_mimic] = getLoaders(mimic_dat, subset=['tinytrain'], num_work=16)
    params = list(je_model.parameters())
    optimizer = torch.optim.Adam(params, lr=0.0001, weight_decay=0.000001)
    tokenizer = Transformer.Bio_tokenizer()
    je_model.train()
    for i, (im1, im2, dfs, texts) in enumerate(train_data_loader_mimic):
        loss = train(device, je_model, im1, im2, texts, tokenizer, use_attn = True)
        loss.backward()
        optimizer.step()
        break



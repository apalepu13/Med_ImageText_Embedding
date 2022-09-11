from torch import nn
import pandas as pd
import torch
import Vision_Model
import MedCLIP_Datasets
from transformers import AutoTokenizer, AutoModel
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
filename = '/n/data2/hms/dbmi/beamlab/chexpert/convirt-retrieval/text-retrieval/query_custom_anil.csv'
labeldescs = pd.read_csv(filename)
filename2 = '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/data/mimic_label_queries_tinytrain.csv'
mimicdescs = pd.read_csv(filename2)
mimlist = []
for h in np.unique(mimicdescs['Variable'].values):
    mimlist.append(mimicdescs[mimicdescs['Variable'] == h].sample(30, random_state=0))
mimicdescs = pd.concat(mimlist)

class MedCLIP(nn.Module):
    def __init__(self, freeze_transformer=True, eval=True):
        super().__init__()
        url = "microsoft/BiomedVLP-CXR-BERT-specialized"
        self.cnn = Vision_Model.get_biovil_resnet()
        self.tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True)
        self.transformer = AutoModel.from_pretrained(url, trust_remote_code=True)
        self.transformer.train(not eval)
        self.cnn.train(not eval)

        self.cls_projection_head = self.transformer.cls_projection_head
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        for param in self.transformer.parameters():
            param.requires_grad = True
        if freeze_transformer:
            modules = [self.transformer.bert.embeddings, *self.transformer.bert.encoder.layer[:8]]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False

    def get_im_embeddings(self, images, only_patches = False, only_ims = False):
        if not isinstance(images, list):
            images = [images]

        images = [im.to(device) for im in images]
        all_patches, all_im_embs = [], []
        for im in images:
            output = self.cnn(im)
            patch_embs = output.projected_patch_embeddings  # N E P1 P2
            all_patches.append(patch_embs.to(device))
            image_emb = output.projected_global_embedding  # N E
            all_im_embs.append(image_emb.to(device))

        if only_patches:
            return all_patches
        elif only_ims:
            return all_im_embs
        else:
            return all_patches, all_im_embs

    def get_text_embeddings(self, text, only_words = False, only_texts = False):
        token_output = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=text,
                                                        add_special_tokens=True, truncation=True,
                                                        padding='longest', max_length=128,
                                                        return_tensors='pt').to(device)
        text_output = self.transformer(token_output.input_ids, token_output.attention_mask)
        text_emb = self.transformer.get_projected_text_embeddings(input_ids=token_output.input_ids,
                                                                  attention_mask=token_output.attention_mask).to(device)
        word_embs_projected = [self.cls_projection_head(text_output.last_hidden_state[:, i, :])[:, None, :] for i in
                               np.arange(text_output.last_hidden_state.shape[1])]
        word_embs = torch.cat(word_embs_projected, dim=1).to(device)  # N T E

        if only_words:
            return word_embs
        elif only_texts:
            return text_emb
        else:
            return word_embs, text_emb

    def get_cross_weights(self, all_patches, word_embs):
        cross_weights = []
        N, T, E = word_embs.shape
        word_embs = word_embs / word_embs.norm(dim=2, keepdim=True)
        stack_patches = torch.cat(all_patches, dim=0).to(device)
        stack_patches = stack_patches.reshape(stack_patches.shape[0], E, -1)
        stack_patches = stack_patches / stack_patches.norm(dim=1, keepdim=True)
        cross_weights_text = torch.bmm(word_embs.repeat(len(all_patches), 1, 1), stack_patches)  # NTP
        for i, im_emb in enumerate(all_patches):
            cross_weights.append(cross_weights_text[(i * N):(i * N + N), :, :])
        return cross_weights

    def get_im_text_matrix(self, all_im_embs, text_emb):
        im_logits = []
        for i, im_emb in enumerate(all_im_embs):
            im_logits.append(self.similarity_matrix(im_emb, text_emb))
        return im_logits

    def get_im_im_matrix(self, all_im_embs):
        aug_logits = None
        if len(all_im_embs) > 1:
            aug_logits = []
            for i in np.arange(len(all_im_embs)):
                for j in np.arange(len(all_im_embs)):
                    if i <= j:
                        continue
                    imsims = self.similarity_matrix(all_im_embs[i], all_im_embs[j])
                    aug_logits.append(imsims)
        return aug_logits


    def similarity_matrix(self, emb1, emb2): #N E, N E
        image_features = emb1 / emb1.norm(dim=-1, keepdim=True)  # N E
        text_features = emb2 / emb2.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        return logits_per_image

    def forward(self, images, text): #k images
        if not isinstance(images, list):
            images = [images]

        images = [im[None, :] if im.dim() == 3 else im for im in images]

        word_embs, text_emb = self.get_text_embeddings(text)
        all_patches, all_im_embs = self.get_im_embeddings(images)
        cross_weights = self.get_cross_weights(all_patches, word_embs)
        imtext_logits = self.get_im_text_matrix(all_im_embs, text_emb)
        imim_logits = self.get_im_im_matrix(all_im_embs)
        return imtext_logits, cross_weights, imim_logits #list per im, #list per im, #list per im-im pair

class SimClassifier(nn.Module):
    def __init__(self, heads, clip_model = None, image_grad=False, transformer_grad=False):
        super().__init__()
        self.clip_model = clip_model
        self.clip_model.transformer.train(transformer_grad)
        self.clip_model.cnn.train(image_grad)
        label_embs = getLabelEmbeddings(clip_model, heads)
        label_embeds = torch.cat([label_embs[h][None, :] for h in heads], dim=0)
        self.label_embeds = label_embeds / label_embeds.norm(dim=1, keepdim=True)
        self.softmax = nn.Softmax(dim=1)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    def forward(self, image): #batch of images, not lists
        list_im_embeds = self.clip_model.get_im_embeddings([image], only_ims=True) #N E
        list_im_embeds = [im_embeds / im_embeds.norm(dim=1, keepdim=True) for im_embeds in list_im_embeds] #N E
        # N E x c E = N c
        list_preds = [im_embeds @ self.label_embeds.t() for im_embeds in list_im_embeds]
        logit_scale = self.logit_scale.exp()
        list_preds = [logit_scale * im_pred for im_pred in list_preds]
        #list_preds = [self.softmax(im_pred) for im_pred in list_preds]
        return list_preds[0]


def getCLIPModel(modelpath=None, modname='best_model.pt', embed_size = 128):
    clip_model = MedCLIP(embed_size).to(device)
    if modelpath:
        checkpoint = torch.load(modelpath + modname, map_location=device)
        clip_model.load_state_dict(checkpoint['model_state_dict'])
    return clip_model

def getSimModel(modelpath=None, modname='best_model.pt', embed_size=128, image_grad = False,
                heads = np.array(['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion', 'No Finding'])):
    clip_model = getCLIPModel(modelpath, modname, embed_size)
    sim_model = SimClassifier(heads, clip_model = clip_model, image_grad = image_grad)
    return sim_model

def getLabelEmbeddings(clip_model, heads, labeldescs = labeldescs, avg=True, convirt=True):
    if convirt:
        mydf = labeldescs
    else:
        mydf = mimicdescs
    mydf = mydf[mydf['Variable'].isin(heads)]
    lab_embeddings = {}
    for h in heads:
        lab = mydf[mydf['Variable'] == h]
        desc = lab['Text'].values
        head_embeddings = clip_model.get_text_embeddings(list(desc), only_texts=True)
        lab_embeddings[h] = torch.mean(head_embeddings, dim=0) if avg else head_embeddings

    return lab_embeddings




if __name__=='__main__':
    model = MedCLIP().to(device)
    dat = MedCLIP_Datasets.MedDataset(source = 'mimic_cxr', group='tinytrain', im_aug = 2,
                 out_heads = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'],
                 filters = [])
    res = dat.__getitem__(3033)
    ims = res['images']
    text = res['texts']
    il, cw, augl = model(ims, text)
    print(il[0].shape, cw[0].shape, augl[0].shape)
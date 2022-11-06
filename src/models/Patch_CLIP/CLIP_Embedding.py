from torch import nn
import pandas as pd
import torch
import Vision_Model
import MedCLIP_Datasets
from transformers import AutoTokenizer, AutoModel
import utils
import pickle
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
filename = '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/src/evaluate/query_custom_anil.csv'
labeldescs = pd.read_csv(filename)
try:
    filename2 = '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/data/mimic_label_queries_tinytrain.csv'
    mimicdescs = pd.read_csv(filename2)
    mimlist = []
    for h in np.unique(mimicdescs['Variable'].values):
        mimlist.append(mimicdescs[mimicdescs['Variable'] == h].sample(30, random_state=0))
    mimicdescs = pd.concat(mimlist)
except:
    mimidescs = labeldescs

class MedCLIP(nn.Module):
    def __init__(self, freeze_transformer=False, eval=True, freeze_CNN=False, findings_transformer = False):
        super().__init__()
        url = "microsoft/BiomedVLP-CXR-BERT-specialized"
        self.cnn = Vision_Model.get_biovil_resnet()
        self.tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True)
        self.transformer = AutoModel.from_pretrained(url, trust_remote_code=True)
        self.ftexists = findings_transformer
        if findings_transformer:
            self.findings_transformer = self.transformer #AutoModel.from_pretrained(url, trust_remote_code=True)

        self.freeze_transformer = freeze_transformer
        self.freeze_projector = False
        self.freeze_CNN = freeze_CNN
        self.freeze_attention = True
        self.train(not eval)

        self.cls_projection_head = self.transformer.cls_projection_head
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.sig = nn.Sigmoid()

        for param in self.cnn.parameters():
            param.requires_grad=True
        for param in self.transformer.parameters():
            param.requires_grad = True
        modules = [self.transformer.bert.embeddings, *self.transformer.bert.encoder.layer[:8]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def train(self, to_train=True):
        super().train(mode=to_train)
        self.transformer.train(to_train)
        self.cnn.train(to_train)
        if self.freeze_transformer:
            self.transformer.train(False)
        if self.freeze_CNN:
            self.cnn.encoder.train(False)
        if self.freeze_projector:
            self.cnn.projector.train(False)
        if self.freeze_attention:
            try:
                self.cnn.global_pooler.train(False)
            except:
                a = 0


    def get_im_embeddings(self, images,guiding_embs=None, only_patches = False, only_ims = False):
        if not isinstance(images, list):
            images = [images]

        images = [im.to(device) for im in images]
        all_patches, all_im_embs, all_pool_weights = [], [], []
        for im in images:
            output = self.cnn(im, guiding_embs.detach() if guiding_embs is not None else None)
            patch_embs = output.projected_patch_embeddings  # N E P1 P2
            all_patches.append(patch_embs.to(device))
            image_emb = output.projected_global_embedding  # N E
            all_im_embs.append(image_emb.to(device))
            all_pool_weights.append(output.pool_weights)


        if only_patches:
            return all_patches
        elif only_ims:
            return all_im_embs
        else:
            return all_patches, all_im_embs, all_pool_weights

    def get_text_embeddings(self, text, only_words = False, only_texts = False, usefindings_transformer=False):
        token_output = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=text,
                                                        add_special_tokens=True, truncation=True,
                                                        padding='longest', max_length=128,
                                                        return_tensors='pt').to(device)
        if usefindings_transformer and self.ftexists:
            text_output = self.findings_transformer(token_output.input_ids, token_output.attention_mask)
            text_emb = self.findings_transformer.get_projected_text_embeddings(input_ids=token_output.input_ids,
                                                                      attention_mask=token_output.attention_mask).to(device)
        else:
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
            cross_weights.append(cross_weights_text[(i * N):(i * N + N), :, :]) # a cross weight for each patch.
        return cross_weights

    def get_patch_text_matrix(self, all_patches, text_emb):
        im_logits = []
        for i, patch_emb in enumerate(all_patches):
            patch_logits = []
            for p1 in range(patch_emb.shape[2]):
                for p2 in range(patch_emb.shape[3]):
                    patch_logits.append(self.similarity_matrix(patch_emb[:, :, p1, p2], text_emb))
            im_logits.append(torch.cat(patch_logits[None, :, :], dim=0).mean(dim=0))
        return im_logits

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

    def forward(self, images, text=None,findings=None, get_im_embeds = False): #k images
        if not isinstance(images, list):
            images = [images]

        images = [im[None, :] if im.dim() == 3 else im for im in images]

        word_embs, text_emb = self.get_text_embeddings(text)
        all_patches, all_im_embs, all_pool_weights = self.get_im_embeddings(images)

        cross_weights = self.get_cross_weights(all_patches, word_embs)
        imtext_logits = self.get_im_text_matrix(all_im_embs, text_emb)

        if self.ftexists:
            findings_words, findings_text = self.get_text_embeddings(findings, usefindings_transformer=True)
            finding_impression_logits = self.get_im_text_matrix([text_emb], findings_text)
            imtext_findings = self.get_im_text_matrix(all_im_embs, findings_text)
            cross_findings = self.get_cross_weights(all_patches, findings_words)
            return imtext_logits, cross_findings,imtext_findings, finding_impression_logits

        imim_logits = self.get_im_im_matrix(all_im_embs)
        return imtext_logits, cross_weights, imim_logits, all_pool_weights #list per im, #list per im, #list per im-im pair


class SimClassifier(nn.Module):
    def __init__(self, heads, clip_model = None, image_grad=False, transformer_grad=False):
        super().__init__()
        self.clip_model = clip_model
        self.clip_model.transformer.train(transformer_grad)
        self.clip_model.cnn.train(image_grad)

        label_embs = getLabelEmbeddings(clip_model, heads)
        label_embeds = torch.cat([label_embs[h][None, :] for h in heads], dim=0)
        self.label_embeds = label_embeds / label_embeds.norm(dim=1, keepdim=True)

        neg_label_embs = getLabelEmbeddings(clip_model, heads, getneg=True)
        neg_label_embeds = torch.cat([neg_label_embs[h][None, :] for h in heads], dim=0)
        self.neg_label_embeds = neg_label_embeds/label_embeds.norm(dim=1, keepdim=True)

        self.softmax = nn.Softmax(dim=1)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    def forward(self, image): #batch of images, not lists
        list_im_embeds = self.clip_model.get_im_embeddings([image], only_ims=True) #N E
        list_im_embeds = [im_embeds / im_embeds.norm(dim=1, keepdim=True) for im_embeds in list_im_embeds] #N E
        # N E x c E = N c
        list_preds = [im_embeds @ self.label_embeds.t() for im_embeds in list_im_embeds]
        list_neg_preds = [im_embeds @ self.neg_label_embeds.t() for im_embeds in list_im_embeds]
        for i, preds in enumerate(list_preds):
            neg_preds = list_neg_preds[i]
            preds = torch.stack([preds[:, :, None], neg_preds[:, :, None]], dim=2)  # N C 2
            preds = torch.nn.Softmax(dim=2)(preds)[:, :, 0].squeeze(dim=2)
            list_preds[i] = preds
        logit_scale = self.logit_scale.exp()
        list_preds = [logit_scale * im_pred for im_pred in list_preds]
        return list_preds[0]

def getSimModel(modelpath=None, modname='best_model.pt', embed_size=128, image_grad = False,
                heads = np.array(['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion', 'No Finding'])):
    clip_model = getCLIPModel(modelpath, modname, embed_size)
    sim_model = SimClassifier(heads, clip_model = clip_model, image_grad = image_grad)
    return sim_model


def getCLIPModel(modelpath=None, modname='best_model.pt', num_models=1, checkpoint=None, eval=True, freezeText=False, freezeCNNEncoder = False):
    if num_models == 1:
        clip_model = MedCLIP(eval = eval).to(device)
        if checkpoint:
            clip_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif modelpath:
            print("loading", modelpath)
            checkpoint = torch.load(modelpath + modname, map_location=device)
            clip_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if freezeText:
            clip_model.freezeTransformer = True
        if freezeCNNEncoder:
            clip_model.cnn.freeze_encoder=True

        return clip_model
    else:
        clip_models = []
        for i in range(num_models):
            clip_model = MedCLIP(eval = eval).to(device)
            if modelpath:
                print("loading",modelpath,i)
                checkpoint = torch.load(modelpath + "best_model_" + str(i) + ".pt", map_location=device)
                clip_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                if freezeText:
                    clip_model.freezeTransformer = True
                if freezeCNNEncoder:
                    clip_model.cnn.freeze_encoder = True
            clip_models.append(clip_model)
        return clip_models

def getLabelEmbeddings(clip_model, heads, labeldescs = labeldescs, avg=True, convirt=True, customdescs=[], getneg = False):
    if convirt:
        mydf = labeldescs
    else:
        mydf = mimicdescs

    #mydf = mydf[mydf['Variable'].isin(heads)]
    lab_embeddings = {}
    if len(customdescs) > 0:
        for c in customdescs:
            mylist = [c + "is present"] if not getneg else [] #[utils.getNegative(c, customdescs=False)]
            if mylist == []:
                lab = mydf[mydf['Variable'] == 'No Finding']
                desc = lab['Text'].values
                desc = np.append(desc, np.array(["No " + c + "."]))
                head_embedding = clip_model.get_text_embeddings(list(desc), only_texts=True)
                lab_embeddings[c] = torch.mean(head_embedding, dim=0)
            else:
                emb = clip_model.get_text_embeddings(mylist, only_texts=True)
                lab_embeddings[c] = torch.mean(emb, dim=0)
        return lab_embeddings

    for h in heads:
        myh = utils.getNegative(h) if getneg else h
        if myh == 'covid19' or myh == 'Pneumonia':
            lab = mydf[mydf['Variable'] == myh + 'custom']
        else:
            lab = mydf[mydf['Variable'] == myh]
        desc = lab['Text'].values
        if getneg:
            desc = np.append(desc, np.array(["No " + h + "."]))
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
import regex as re
from MedCLIP_Datasets import *
import CLIP_Embedding
import torch
import torch.nn as nn
import Vision_Model
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from sklearn import metrics
import pandas as pd
from matplotlib import pyplot as plt

def getExperiment(args, mp):
    if args.debug:
        return "debug"

    if args.resume and args.resume > 0:
        fp =  os.path.join(mp, 'exp'+str(args.resume))
        if os.path.exists(fp):
            return fp
        else:
            raise Exception("Experiment doesn't exist, cannot resume exp " + fp)

    if not os.listdir(os.path.join(mp)):
        if args.resume:
            raise Exception("No experiment exist, cannot resume last one.")
        print("No models exist, creating directory")
        fp = os.path.join(mp, 'exp1')
    else:
        all_files = os.listdir(os.path.join(mp))
        je_exps = [exp for exp in all_files if 'exp' in exp]
        num = [int(re.search('\d+', exp).group(0)) for exp in je_exps]
        highest_ind = np.argmax(np.array(num))
        highest = num[highest_ind]
        if not args.resume:
            highest = highest + 1
        fp = os.path.join(mp, 'exp'+str(highest))
    return fp

def writeArgs(fp, args):
    '''
    Document args used to train
    '''
    writestr = str(args)
    with open(fp + '/args.txt', 'w') as f:
        f.write(writestr)

def startExperiment(args, fp, cnn=False, pretrained=True):
    '''
    Initialize variables for experiment:
    start (epoch), je_model, params, optimizer, best_val_loss
    '''
    je_model = CLIP_Embedding.MedCLIP(eval=False, findings_transformer=args.findings_transformer).to(device) if not cnn else Vision_Model.getCNN(pretrained=pretrained, classifier=True).to(device)
    params = list(je_model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=0.000001)
    if fp == "debug":
        return 0, je_model, params, optimizer, 100000

    if args.resume:
        if os.listdir(os.path.join(fp)):
            all_files = os.listdir(os.path.join(fp))
            je_files = [file for file in all_files if 'je_model' in file]
            num = [int(re.search('\d+', file).group(0)) for file in je_files]
            highest = np.argmax(np.array(num))
            loadpath = os.path.join(fp, np.array(je_files)[highest])
            print("Loading " + loadpath)
            checkpoint = torch.load(loadpath)
            je_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss'] if 'best_val_loss' in checkpoint.keys() else checkpoint['val_loss']
        else:
            raise Exception("Experiment doesn't exist: " + fp)
    else:
        print("Starting from scratch")
        start = 0
        best_val_loss = 1000000000
        if not args.debug:
            os.makedirs(fp)
            writeArgs(fp, args)
    return start, je_model, params, optimizer, best_val_loss

def attn_penalty(cross_weights, soft = nn.Softmax(dim=2), lam = (0.0, 0.0)):
    attn_loss = 0 #lam patch * entropy of each word similarity (across patches)
    losses = []   #lam words * entropy of each patch similarity (across words)
    eps = 1e-7
    for c in cross_weights:
        entropy = soft(c) + eps #NTP
        entropy = -entropy * torch.log(entropy)
        entropy_text = torch.sum(entropy, dim=2) #N, T
        entropy_text = torch.mean(entropy_text, dim=(1, 0)) #1

        entropy_im = soft(c.permute(0, 2, 1)) + eps
        entropy_im = -entropy_im * torch.log(entropy_im)
        entropy_im = torch.sum(entropy_im, dim=2) #N, P
        entropy_im = torch.mean(entropy_im, dim=(1, 0)) # 1

        loss = (entropy_text * float(lam[0])) + (entropy_im * float(lam[1]))
        losses.append(loss.cpu().detach())
        attn_loss += loss
    return attn_loss, losses #1

def clip_loss(im_logits, aug_logits = None, loss_weight = 1, criterion = nn.CrossEntropyLoss()):
    text_logits = [im.t() for im in im_logits]
    clip_loss = 0
    losses = []
    for i in np.arange(len(im_logits)): #for each image augmentation-text matrix (usually just 1)
        samp = torch.tensor(np.arange(im_logits[i].shape[0]))
        loss_a = criterion(im_logits[i], samp.to(device))
        loss_b = criterion(text_logits[i], samp.to(device))
        closs = (loss_a + loss_b) / 2
        losses.append(closs.cpu().detach())
        clip_loss += closs * loss_weight
    if aug_logits is not None: #for each image augmentation-image augmentation matrix (usually 0)
        for i in np.arange(len(aug_logits)):
            samp = torch.tensor(np.arange(im_logits[i].shape[0]))
            imloss = criterion(im_logits[i], samp.to(device))
            losses.append(imloss.cpu().detach())
            clip_loss += imloss
    assert len(losses) == int((len(im_logits) + (len(im_logits) * (len(im_logits) -1)/2.0)))
    return clip_loss, losses

def compute_loss(je_model, samples, args, attn_lam_words = 0.0, attn_lam_patches = 0.0):
    ims = samples['images']
    texts = samples['texts']
    pool_loss = None
    if not args.findings_transformer:
        im_logits, crosses, aug_logits, pool_weights = je_model(ims, texts)
        if pool_loss:
            pool_probs_list = [nn.Softmax(dim=1)(p.reshape(-1, 49)) for p in pool_weights]
            pool_entropy_list = [-p * torch.log(p) for p in pool_probs_list]
            pool_loss_list = [(0.2 * p.sum(dim=1).mean(dim=0)) for p in pool_entropy_list]
        else:
            pool_loss_list = []

        cl, cl_losses = clip_loss(im_logits, aug_logits)
        attn_pen, attn_losses = attn_penalty(crosses, lam=(attn_lam_words, attn_lam_patches))
        cl_count = len(cl_losses)
        attn_count = len(attn_losses)
        pool_loss = sum(pool_loss_list) / len(pool_loss_list) if len(pool_loss_list) > 0 else 0
        loss = cl / cl_count + attn_pen / attn_count + pool_loss
        all_losses = cl_losses + attn_losses + pool_loss_list
        return loss, torch.tensor(all_losses)
    else:
        findings = samples['findings']
        im_logits, crosses, findings_logits, text_to_text_logits = je_model(ims, texts, findings)
        imp_l, imp_losses = clip_loss(im_logits)
        find_l, find_losses = clip_loss(findings_logits)
        text_l, text_losses = clip_loss(text_to_text_logits)
        attn_pen, attn_losses = attn_penalty(crosses, lam=(attn_lam_words, attn_lam_patches))
        loss = imp_l + find_l + text_l + attn_pen
        all_losses = imp_losses + find_losses + text_losses + attn_losses
        return loss, torch.tensor(all_losses)



def train(train_data_loader, je_model, args, epoch, optimizer, total_step_mimic=-1, lam_words = -1, lam_patch = -1):
    mean_loss, mean_losses, ct = 0.0, 0.0, 0
    if lam_words < 0:
        lam_words = args.lam_words
    if lam_patch < 0:
        lam_patch = args.lam_patches

    je_model.train(True)
    for i, samples in enumerate(train_data_loader):
        je_model.zero_grad(set_to_none=True)
        loss, all_losses = compute_loss(je_model, samples, args, attn_lam_words=lam_words, attn_lam_patches = lam_patch)
        # Forward, backward and optimize
        loss.backward()
        optimizer.step()
        if total_step_mimic > 0:
            if i % args.log_step == 0:
                print('MIMIC Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, args.num_epochs, i, total_step_mimic, loss.item()))
                print(all_losses)
        l = loss.cpu().detach().numpy()
        mean_loss += l
        mean_losses += all_losses
        ct += 1
    if ct > 0:
        mean_loss = mean_loss/ct
        mean_losses = mean_losses/ct

    return mean_loss, mean_losses

def validate(val_data_loader, je_model, args):
    val_losses = []
    avg_loss, ct = 0.0, 0
    je_model.train(False)
    with torch.no_grad():
        for j, samples in enumerate(val_data_loader):
            loss, all_losses = compute_loss(je_model, samples, args, attn_lam_words=args.lam_words, attn_lam_patches = args.lam_patches)
            val_losses.append(all_losses.view(-1,1))
            avg_loss += loss
            ct += 1
    avg_loss = avg_loss/ct

    val_losses = torch.cat(val_losses, dim=1) #num batches x num losses
    avg_losses = torch.mean(val_losses, dim=1)
    assert avg_losses.shape == all_losses.shape


    if avg_losses.shape[0] == 5:
        names = ['im1-t', 'im2-t', 'im1-im2', 'attn im1-t', 'attn im2-t']
        lossstr = ""
        for i in range(len(names)):
            lossstr += (", " + names[i] + ": " + str(avg_losses[i].item()))
        print("Val losses" + lossstr)
    elif avg_losses.shape[0] == 4:
        names = ['impression-im', 'findings-im', 'impression-finding', 'attn-finding']
        lossstr = ""
        for i in range(len(names)):
            lossstr += (", " + names[i] + ": " + str(avg_losses[i].item()))
        print("Val losses" + lossstr)
    elif avg_losses.shape[0] == 3:
        names = ['im-t', 'attn im-t', 'pool entropy']
        lossstr = ""
        for i in range(len(names)):
            lossstr += (", " + names[i] + ": " + str(avg_losses[i].item()))
        print("Val losses" + lossstr)
    elif avg_losses.shape[0] == 2:
        names = ['im-t', 'attn im-t']
        lossstr = ""
        for i in range(len(names)):
            lossstr += (", " + names[i] + ": " + str(avg_losses[i].item()))
        print("Val losses" + lossstr)

    return avg_loss.item(), avg_losses

def b_loss(cnn_model, samples, args, heads, criterion=torch.nn.BCEWithLogitsLoss(reduction='mean')):
    im = samples['images'][0].to(device)
    impreds = cnn_model(im).class_logits.to(device)
    impreds = impreds.squeeze(dim=2)
    labels = samples['labels']
    losses = torch.zeros(len(heads))
    for i, h in enumerate(heads):
        label = labels[h]
        label[label == -1.0] = float('nan')
        label[label == 0.0] = 0
        label[label == 1.0] = 1
        label = label.float().to(device)
        mypreds = impreds[torch.logical_not(torch.isnan(label)), i]
        mylabels = label[torch.logical_not(torch.isnan(label))]
        losses[i] = criterion(mypreds, mylabels)
    losses = losses[torch.logical_not(torch.isnan(losses))]
    loss = torch.mean(losses)
    if torch.isnan(loss):
        loss = 0
    return loss

def pseudo_b_loss(clip_model, samples, args, heads, pos_embeds, neg_embeds, criterion=torch.nn.BCEWithLogitsLoss(reduction='mean'), useSimLoss = False):
    relu = nn.ReLU()
    im = samples['images']
    labels = samples['labels']
    losses = torch.zeros(len(heads))
    list_im_embeds = clip_model.get_im_embeddings(im, only_ims=True)
    list_im_embeds = [im_embeds / im_embeds.norm(dim=1, keepdim=True) for im_embeds in list_im_embeds]
    list_preds = [im_embeds @ pos_embeds.t() for im_embeds in list_im_embeds]  # N c
    list_neg_preds = [im_embeds @ neg_embeds.t() for im_embeds in list_im_embeds]
    for i in range(len(list_preds)):
        list_preds[i] = torch.stack([list_preds[i][:, :, None], list_neg_preds[i][:, :, None]], dim=2)  # N C 2
        list_preds[i] = torch.nn.Softmax(dim=2)(list_preds[i])[:, :, 0].squeeze(dim=2)  # N C 1

    impreds = list_preds[0]

    for i, h in enumerate(heads):
        label = labels[h]
        label[label == -1.0] = float('nan')
        label[label == 0.0] = 0
        label[label == 1.0] = 1
        label = label.float().to(device)
        mypredsProb = impreds[torch.logical_not(torch.isnan(label)), i]
        mylabels = label[torch.logical_not(torch.isnan(label))]
        mypreds = (1/0.05) * torch.log(mypredsProb/(1-mypredsProb))

        if useSimLoss:
            if torch.sum(mylabels==1) < 1:
                lossPos = 0
            else:
                lossPos = torch.mean(1-mypreds[mylabels == 1])

            if torch.sum(mylabels == 0) < 1:
                lossNeg = 0
            else:
                lossNeg = torch.mean(relu(mypreds[mylabels == 0]))
            losses[i] = lossPos + lossNeg
        else:
            losses[i] = criterion(mypreds, mylabels)
    losses = losses[torch.logical_not(torch.isnan(losses))]
    loss = torch.mean(losses)
    if torch.isnan(loss):
        loss = 0
    return loss

def train_vision(train_data_loader, cnn_model, args, epoch, optimizer, totstep= None, heads=None, list_mods = False, je_inds = [], all_pos_embeds=None, all_neg_embeds=None):
    if list_mods:
        full_loss = [0 for i in cnn_model]
    else:
        full_loss = [0]
        cnn_model = [cnn_model]
        optimizer = [optimizer]

    for i, samples in enumerate(train_data_loader):
        for j, cnn_mod in enumerate(cnn_model):
            cnn_mod.train(True)
            cnn_mod.zero_grad(set_to_none=True)
            if j not in je_inds:
                loss = b_loss(cnn_mod, samples, args, heads)
            else:
                cnn_mod.cnn.train(True)
                loss = pseudo_b_loss(cnn_mod, samples, args, heads, all_pos_embeds[j], all_neg_embeds[j])
            loss.backward()
            # Forward, backward and optimize

            optimizer[j].step()
            if (totstep) and (i % args.log_step == 0):
                print('MIMIC Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, args.num_epochs, i, totstep,
                                                                               loss.item()))
            full_loss[j] += loss.detach().cpu().item()
    avg_loss = [f/(i + 1) for f in full_loss]
    return avg_loss[0] if not list_mods else avg_loss

def validate_vision(val_data_loader, cnn_model, args=None, heads=None, list_mods = False, je_inds = [], all_pos_embeds = None, all_neg_embeds = None):
    if list_mods:
        avg_loss = [0.0 for mod in cnn_model]
    else:
        avg_loss = [0.0]
        cnn_model = [cnn_model]

    ct = 0
    with torch.no_grad():
        for i, samples in enumerate(val_data_loader):
            for j, mod in enumerate(cnn_model):
                mod.train(False)
                if j not in je_inds:
                    loss = b_loss(mod, samples, args, heads)
                else:
                    loss = pseudo_b_loss(mod, samples, args, heads, all_pos_embeds[j], all_neg_embeds[j])
                avg_loss[j] += loss
            ct += 1
    avg_loss = [a/ ct for a in avg_loss]
    print("Val loss: " + str(avg_loss))
    return avg_loss[0].item() if not list_mods else [a.item() for a in avg_loss]

def getLabels(df, heads, replace_nan = False):
    labels = None
    for i, h in enumerate(heads):
        label = df[h].float()
        label[label==-1.0] = float('nan')
        if replace_nan:
            label = torch.nan_to_num(label)
        if labels is None:
            labels = label
            labels = labels[:, None]
        else:
            labels = torch.cat((labels, label[:, None]), axis=1)

    return labels #N x c

def getRadLabels(df, heads, suffix):
    labels = None
    for i, h in enumerate(heads):
        label = df[h + '_' + suffix].float()
        label[label==-1.0] = float('nan')
        if labels is None:
            labels = label
            labels = labels[:, None]
        else:
            labels = torch.cat((labels, label[:, None]), axis=1)

    return labels #N x c

def getZeroShotAcc(checkpoint, DL, heads = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'], usecheckpoint=True):
    aucs, tprs, fprs, thresholds = {}, {}, {}, {}
    if usecheckpoint:
        clip_model = CLIP_Embedding.getCLIPModel(checkpoint=checkpoint, eval=True)
    else:
        clip_model = checkpoint
    preds, labels = get_all_preds(DL, clip_model, similarity=True, heads = heads, normalization=True)
    preds = preds[0].cpu().detach().numpy()
    targs = labels.cpu().int().detach().numpy()
    for i, h in enumerate(heads):
        myt = targs[:, i]
        myp = preds[:, i]
        fprs[h], tprs[h], thresholds[h] = metrics.roc_curve(myt, myp)
        aucs[h] = metrics.auc(fprs[h], tprs[h])
    return np.mean(np.array([aucs[h] for h in heads]))

def getNegative(h, customdescs=False):
    if h == 'normal' or 'No Finding':
        return 'Abnormal findings'
    if customdescs:
        return h + " not present"
    return "No Finding"

def get_all_preds(DL, mod=None,similarity=False,im_embeds=False,patch_similarity=False, only_labels=False,
                  heads = ['covid19', 'No Finding'], convirt=True, getlabels=True, normalization=False,
                  im_classifier=False, squash_studies=False, autochoose='', textAttention=False, also_studies=False, getrad=False):

    if autochoose != '':
        if 'Zeroshot' in autochoose or 'CNN' not in autochoose:
            similarity, normalization =True, True
        elif 'Finetuned' in autochoose or 'CNN' in autochoose:
            im_classifier=True

    if getrad:
        rad1, rad2, rad3 = [], [], []

    if only_labels:
        tstudies = []
        tt = []
        for i, samples in enumerate(DL):
            if also_studies:
                tstudies+= samples['study']
            labels = getLabels(samples['labels'], heads)
            tt.append(labels)
        tt = torch.cat(tt, axis=0)
        if also_studies:
            return tt, tstudies
        return tt

    with torch.no_grad():
        if similarity or patch_similarity:
            label_embeds = CLIP_Embedding.getLabelEmbeddings(mod, heads, convirt=convirt)
            embed_list = [label_embeds[h][None, :] for h in heads]
            label_embeds = torch.cat(embed_list, dim=0)
            label_embeds = label_embeds/label_embeds.norm(dim=1, keepdim=True)
            if normalization:
                neg_label_embeds = CLIP_Embedding.getLabelEmbeddings(mod, heads, convirt=convirt, getneg=True)
                neg_embed_list = [neg_label_embeds[h][None, :] for h in heads]
                neg_label_embeds = torch.cat(neg_embed_list, dim=0)
                neg_label_embeds = neg_label_embeds / neg_label_embeds.norm(dim=1, keepdim=True)

        all_studies = []
        for i, samples in enumerate(DL):
            if squash_studies:
                all_studies += samples['study']
            images = samples['images']
            if i == 0:
                tt = []
                tps = [[] for im in images]

            if im_classifier:
                mod.train(False)
                list_preds = [mod(im.to(device)).class_logits.squeeze(dim=2) for im in images]
                if list_preds[0].shape[1] > len(heads):
                    list_preds = [l[:, :len(heads)] for l in list_preds]

            elif similarity:
                list_im_embeds = mod.get_im_embeddings(images, only_ims = True)
                list_im_embeds = [im_embeds/im_embeds.norm(dim=1, keepdim=True) for im_embeds in list_im_embeds]
                # N P E x c E = N c
                list_preds = [im_embeds @ label_embeds.t() for im_embeds in list_im_embeds]  # N c

                if normalization:
                    list_neg_preds = [im_embeds @ neg_label_embeds.t() for im_embeds in list_im_embeds] #N c
                    for i, sim_preds in enumerate(list_preds):
                        list_preds[i] = list_preds[i]- list_neg_preds[i] # if just subtraction

            elif patch_similarity:
                list_im_embeds = mod.get_im_embeddings(images, only_patches=True) #N E P1 P2
                list_im_embeds = [im_embeds.reshape(im_embeds.shape[0], im_embeds.shape[1], im_embeds.shape[2] * im_embeds.shape[3]).permute(0, 2, 1) for im_embeds in list_im_embeds] #N P E
                list_im_embeds = [im_embeds/im_embeds.norm(dim=2, keepdim=True) for im_embeds in list_im_embeds]
                N, P, E = list_im_embeds[0].shape
                C = len(heads)
                # N P E x c E = N P c
                list_preds = [torch.bmm(im_embeds, label_embeds.t()[None, :, :].repeat(im_embeds.shape[0], 1, 1)) for im_embeds in list_im_embeds]
                if normalization:
                    list_neg_preds = [torch.bmm(im_embeds, neg_label_embeds.t()[None, :, :].repeat(im_embeds.shape[0], 1, 1)) for im_embeds in list_im_embeds]
                    for j, sim_preds in enumerate(list_preds):
                        # list_preds[i] = list_preds[i] - list_neg_preds[i] # if just subtraction
                        list_preds[j] = torch.stack([list_preds[j][:, :,:,  None], list_neg_preds[j][:, :,:, None]],
                                                    dim=3)  # N C 2
                        list_preds[j] = torch.nn.Softmax(dim=3)(list_preds[j])[:, :,:, 0].squeeze(dim=3)  # N P c
                        list_preds[j] = list_preds[j].mean(dim=1)

            elif im_embeds:
                list_im_embeds = mod.get_im_embeddings(images, only_ims=True)
                list_preds = [im_embeds / im_embeds.norm(dim=1, keepdim=True) for im_embeds in list_im_embeds]

            labels = getLabels(samples['labels'], heads) if getlabels else None
            tt.append(labels) if getlabels else None
            if getrad:
                rad1.append(getRadLabels(samples['labels'], heads, 'rad1'))
                rad2.append(getRadLabels(samples['labels'], heads, 'rad2'))
                rad3.append(getRadLabels(samples['labels'], heads, 'rad3'))

            for j, pred in enumerate(list_preds):
                tps[j].append(pred.cpu())


        tplist = [torch.cat(tp, axis=0) for tp in tps]
        tt = torch.cat(tt, axis=0) if getlabels else None
        if getrad:
            rad1 = torch.cat(rad1, axis=0)
            rad2 = torch.cat(rad2, axis=0)
            rad3 = torch.cat(rad3, axis=0)

        if squash_studies:
            print(len(all_studies))
            for i, tp in enumerate(tplist):
                df = pd.DataFrame(data=tp.cpu().numpy(), columns=heads)
                df['Study'] = np.array(all_studies)
                df = df.groupby('Study').mean().reset_index()
                tplist[i] = torch.tensor(df.values[:, 1:].astype(float))
            df = pd.DataFrame(data=tt.cpu().numpy(), columns=heads)
            df['Study'] = np.array(all_studies)
            df = df.groupby('Study').mean().reset_index()
            tt = torch.tensor(df.values[:, 1:].astype(int))
            print(tt.shape)

            if getrad:
                df = pd.DataFrame(data=rad1.cpu().numpy(), columns=heads)
                df['Study'] = np.array(all_studies)
                df = df.groupby('Study').mean().reset_index()
                rad1 = torch.tensor(df.values[:, 1:].astype(int))
                print(rad1.shape)

                df = pd.DataFrame(data=rad2.cpu().numpy(), columns=heads)
                df['Study'] = np.array(all_studies)
                df = df.groupby('Study').mean().reset_index()
                rad2 = torch.tensor(df.values[:, 1:].astype(int))
                print(rad2.shape)

                df = pd.DataFrame(data=rad3.cpu().numpy(), columns=heads)
                df['Study'] = np.array(all_studies)
                df = df.groupby('Study').mean().reset_index()
                rad3 = torch.tensor(df.values[:, 1:].astype(int))
                print(rad3.shape)

        if getrad:
            return tplist, tt, rad1, rad2, rad3
        else:
            return tplist, tt

def getPadPredictions(DL, models = None, ensemble=True, soft_norm=True, only_labels = False):
    tt = []
    for z, sample in enumerate(DL):
        if z == 0:
            name_list = list(sample['labels'].keys())
            label_embeds_all = []
            neg_label_embeds_all = []
            if not only_labels:
                tps_models = [[] for model in models]
                with torch.no_grad():
                    for j, m in enumerate(models):
                        label_embeds = CLIP_Embedding.getLabelEmbeddings(m, name_list, customdescs=name_list)
                        embed_list = [label_embeds[h][None, :] for h in name_list]
                        label_embeds = torch.cat(embed_list, dim=0)
                        label_embeds = label_embeds / label_embeds.norm(dim=1, keepdim=True)
                        label_embeds_all.append(label_embeds)
                        print("labelembed shape",label_embeds_all[0].shape)

                        neg_label_embeds = CLIP_Embedding.getLabelEmbeddings(m, name_list, customdescs=name_list, getneg=True)
                        neg_embed_list = [neg_label_embeds[h][None, :] for h in name_list]
                        neg_label_embeds = torch.cat(neg_embed_list, dim=0)
                        neg_label_embeds = neg_label_embeds / neg_label_embeds.norm(dim=1, keepdim=True)
                        neg_label_embeds_all.append(neg_label_embeds)
                    l = label_embeds[2, :]
                    n = neg_label_embeds[2, :]
                    l1 = label_embeds[1, :]
                    print(torch.sum(l * n), torch.sum(l * l1))

        labels = getLabels(sample['labels'], name_list)
        tt.append(labels)
        if only_labels:
            continue

        images = sample['images']
        if z == 0:
            firsttwo = images[0][:2, :, :, :]
            firsttwo = normalize(firsttwo, getOne=False)
            plt.figure(figsize=(8,8))
            plt.imshow(firsttwo[0, :, :, :].squeeze().detach().cpu().numpy(), cmap='gray')
            plt.savefig('im0.png', bbox_inches='tight')
            plt.figure(figsize=(8, 8))
            plt.imshow(firsttwo[1, :, :, :].squeeze().detach().cpu().numpy(), cmap='gray')
            plt.savefig('im1.png', bbox_inches='tight')


        for j, m in enumerate(models):
            with torch.no_grad():
                im_embeds = m.get_im_embeddings(images, only_ims=True)[0]
                im_embeds = im_embeds / im_embeds.norm(dim=1, keepdim=True)
                # N P E x c E = N c
                preds = im_embeds @ label_embeds_all[j].t()
                if soft_norm:
                    neg_preds = im_embeds @ neg_label_embeds_all[j].t() # N c
                    preds= torch.stack([preds[:, :, None], neg_preds[:, :, None]],
                                                    dim=2)  # N C 2
                    preds = torch.nn.Softmax(dim=2)(preds)[:, :, 0].squeeze(dim=2)

                tps_models[j].append(preds.cpu())
        if z == 0:
            print(torch.sum(im_embeds[0, :] * im_embeds[1, :]))
    if not only_labels:
        tps_models = [torch.cat(tps, dim=0) for tps in tps_models]#list of models, list of ims,tensor preds
    tt = torch.cat(tt, dim=0)
    if only_labels:
        return tt

    if ensemble:
        if len(models)>1:
            tps_models = [modelpred[None, :, :] for modelpred in tps_models]  # list of models prediction tensors
            tps_avg = torch.cat(tps_models, dim = 0).mean(dim = 0, keepdim=False) #stacked prediction ten
        else:
            tps_models = [modelpred for modelpred in tps_models]  # list of models prediction tensors
            tps_avg = tps_models[0]
        print(tps_avg.shape)
        print(tt.shape)
        assert tt.shape == tps_avg.shape
        tps_models = tps_avg

    return tps_models, tt, name_list

def get_all_preds_list(DL, mod=None, normalize = True,
                  heads=['covid19', 'No Finding'], autochooses=['']):

    models = mod
    similaritys = []
    im_classifiers = []
    for autochoose in autochooses:
        if autochoose != '':
            if 'Zeroshot' in autochoose or 'CNN' not in autochoose:
                similaritys.append(True)
                im_classifiers.append(False)
            elif 'CNN' in autochoose:
                similaritys.append(False)
                im_classifiers.append(True)

    label_embeds_all = []
    neg_label_embeds_all = []
    for k, m in enumerate(models):
        if similaritys[k]:
            label_embeds = CLIP_Embedding.getLabelEmbeddings(m, heads, convirt=True)
            embed_list = [label_embeds[h][None, :] for h in heads]
            label_embeds = torch.cat(embed_list, dim=0)
            label_embeds = label_embeds / label_embeds.norm(dim=1, keepdim=True)
            label_embeds_all.append(label_embeds)
        else:
            label_embeds_all.append([])

        if normalize and similaritys[k]:
            neg_label_embeds = CLIP_Embedding.getLabelEmbeddings(m, heads, convirt=True, getneg=True)
            neg_embed_list = [neg_label_embeds[h][None, :] for h in heads]
            neg_label_embeds = torch.cat(neg_embed_list, dim=0)
            neg_label_embeds = neg_label_embeds / neg_label_embeds.norm(dim=1, keepdim=True)
            neg_label_embeds_all.append(neg_label_embeds)
        else:
            neg_label_embeds_all.append([])


    with torch.no_grad():
        for i, samples in enumerate(DL):
            images = samples['images']
            if i == 0:
                tt = []
                tps_models = [[[] for im in images]  for model in models]

            for k, mod in enumerate(models):
                im_classifier = im_classifiers[k]
                similarity = similaritys[k]
                if im_classifier:
                    mod.train(False)
                    list_preds = [mod(im.to(device)).class_logits.squeeze(dim=2) for im in images]
                    if list_preds[0].shape[1] > len(heads):
                        list_preds = [l[:, :len(heads)] for l in list_preds]

                elif similarity:
                    list_im_embeds = mod.get_im_embeddings(images, only_ims=True)
                    list_im_embeds = [im_embeds / im_embeds.norm(dim=1, keepdim=True) for im_embeds in list_im_embeds]
                    # N P E x c E = N c
                    list_preds = [im_embeds @ label_embeds_all[k].t() for im_embeds in list_im_embeds]
                    if normalize:
                        list_neg_preds = [im_embeds @ neg_label_embeds_all[k].t() for im_embeds in list_im_embeds]  # N c
                        for i, sim_preds in enumerate(list_preds):
                            # list_preds[i] = list_preds[i] - list_neg_preds[i] # if just subtraction
                            list_preds[i] = torch.stack([list_preds[i][:, :, None], list_neg_preds[i][:, :, None]],
                                                        dim=2)  # N C 2
                            list_preds[i] = torch.nn.Softmax(dim=2)(list_preds[i])[:, :, 0].squeeze(dim=2)  # N C 1

                for j, pred in enumerate(list_preds):
                    tps_models[k][j].append(pred.cpu())

            labels = getLabels(samples['labels'], heads)
            tt.append(labels)

        tps_models = [[torch.cat(tp, axis=0) for tp in tps] for tps in tps_models]
        tt = torch.cat(tt, axis=0)
        print(tt.shape)
        return tps_models, tt #list of list of tensors (#mod, #augs, preds), tensor

def normalize(image, getOne = True):
    img = torch.clone(image)
    img[:, 0, :, :] = (img[:, 0, :, :] * .229) + .485
    img[:, 1, :, :] = (img[:, 1, :, :] * .224) + .456
    img[:, 2, :, :] = (img[:, 2, :, :] * .225) + .406
    if getOne:
        img = img.permute(0, 2, 3, 1)[0, :, :, :].squeeze()
    else:
        img = img.permute(0, 2, 3, 1)
    return img

def getLabelSimilarities(mod, heads, label_embeds=None, compare_mimic = False):
    with torch.no_grad():
        if compare_mimic:
            label_embeds = CLIP_Embedding.getLabelEmbeddings(mod, heads)
            label_embeds_mimic = CLIP_Embedding.getLabelEmbeddings(mod, heads, convirt=False)
            for i, h in enumerate(heads):
                print(h, torch.dot(label_embeds[h] / label_embeds[h].norm(dim=-1, keepdim=True),
                                       label_embeds_mimic[h] / label_embeds_mimic[h].norm(dim=-1, keepdim=True)).cpu())
        else:
            if not label_embeds:
                label_embeds = CLIP_Embedding.getLabelEmbeddings(mod, heads)
            for i, h in enumerate(heads):
                for j, h2 in enumerate(heads):
                    if i < j:
                        print(h, h2, torch.dot(label_embeds[h] / label_embeds[h].norm(dim=-1, keepdim=True),
                                               label_embeds[h2] / label_embeds[h2].norm(dim=-1, keepdim=True)).cpu())




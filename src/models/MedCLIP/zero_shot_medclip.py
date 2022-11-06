
from medclip import MedCLIPModel, MedCLIPVisionModelViT
from medclip import MedCLIPProcessor
from medclip import PromptClassifier
from medclip.prompts import generate_chexpert_class_prompts, process_class_prompts
import sys
sys.path.insert(0, '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/src/models/Patch_CLIP/')
import MedCLIP_Datasets
import MedDataHelpers
import utils
from sklearn import metrics
import pickle
import numpy as np
import os
import torch
import torchvision.transforms as T
transform = T.ToPILImage()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_all_preds(DL):
    heads = np.array(['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion'])
    tp = []
    tt = []
    processor = MedCLIPProcessor()
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    model.from_pretrained()
    clf = PromptClassifier(model, ensemble=True)
    clf.cuda()
    cls_prompts = process_class_prompts(generate_chexpert_class_prompts(n=10))
    for i, sample in enumerate(DL):
        image = sample['orig_image']
        image = transform(image[0, :, :, :].squeeze(dim=0))
        inputs = processor(images=image, return_tensors="pt")
        inputs['prompt_inputs'] = cls_prompts
        output = clf(**inputs)
        if i == 0:
            print(output)
        tp.append(output['logits'].cpu().detach())
        tt.append(utils.getLabels(sample['labels'], heads).cpu().detach())
    tp = torch.cat(tp, dim=0)
    tt = torch.cat(tt, dim=0)
    return tp.cpu().detach().numpy(), tt.cpu().detach().numpy()

heads = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
subset=['test']
aucs, aucs_synth, aucs_adv, tprs, fprs, thresholds = {}, {}, {}, {}, {}, {}
filters = []
modname = "medclip-vit"
dat = MedDataHelpers.getDatasets(source='chextest', subset=subset, synthetic=False, filters=filters,
                                 heads=heads)  # Real
DLs = MedDataHelpers.getLoaders(dat, shuffle=False, bsize=1)
DL = DLs[subset[0]]
test_preds, test_targets = get_all_preds(DL)

for i, h in enumerate(heads):
    targs = test_targets[:, i]
    tpreds = test_preds[:, i]
    fprs[h], tprs[h], thresholds[h] = metrics.roc_curve(targs, tpreds)
    aucs[h] = np.round(metrics.auc(fprs[h], tprs[h]), 5)

aucs['Total'] = np.round(np.mean(np.array([aucs[h] for h in heads])), 5)

print("Normal")
print("Total AUC avg: ", aucs['Total'])
for i, h in enumerate(heads):
    print(h, aucs[h])


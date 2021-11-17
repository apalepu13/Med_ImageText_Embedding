import os
import pandas as pd
import xml.etree.ElementTree as ET
import regex as re

indiana_root_dir='/n/scratch3/users/a/anp2971/datasets/'
flist_txt = os.listdir(indiana_root_dir + 'indiana_cxr_reports/ecgen-radiology')
all_images = []
all_texts = []
all_pats = []


def parseXML(xmlfile):
    # create element tree object
    tree = ET.parse(xmlfile)
    # get root element
    text = ""
    ims = []
    pat = ""
    for elem in tree.iter():
        if elem.tag == 'AbstractText' and not (elem.text is None) and 'None' not in elem.text:
            text = text + elem.text
            if text[-2:] != ". ":
                if text[-1] == ".":
                    text = text + " "
                elif text[-1] == " ":
                    text = text[:-1] + ". "
                else:
                    text = text + ". "
        elif 'caption' in elem.tag and not (elem.text is None) and 'None' not in elem.text:
            if elem.text[1:-1] not in text:
                text = text + elem.text
            if text[-2:] != ". ":
                if text[-1] == ".":
                    text = text + " "
                elif text[-1] == " ":
                    text = text[:-1] + ". "
                else:
                    text = text + ". "
        elif elem.tag == 'url':
            full_url = elem.text
            just_imfile = re.findall('.+/(.*\.jpg)', full_url)[0]
            pat = re.findall('CXR(.+?)_', full_url)[0]
            ims.append(just_imfile)
        print("->", elem.tag, elem.text)
    for i in ims:
        #print(ims)
        all_images.append(i)
        all_texts.append(text)
        all_pats.append(pat)

for f in flist_txt[::-1]:
    print("STUDY!")
    parseXML(indiana_root_dir + 'indiana_cxr_reports/ecgen-radiology/' + f)

df = pd.DataFrame({'images': all_images, 'texts': all_texts, 'patient': all_pats})
print(df)
df.to_csv('../../../data/indiana_cxr_list.csv')
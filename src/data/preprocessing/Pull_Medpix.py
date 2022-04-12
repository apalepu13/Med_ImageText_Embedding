import requests
import numpy as np
import pandas as pd
import re
import urllib.request as ur



maxes = np.arange(100, 8200, 100)
urls = []
for m in maxes:
    case_max = m
    case_min = str(case_max - 99)
    case_max = str(case_max)
    urls.append('https://openi.nlm.nih.gov/api/search?coll=mpx&m=' + case_min + '&n=' + case_max)

myDF = pd.DataFrame(columns=['uid', 'pmcid', 'title', 'fulltext_html_url', 'medpixImageURL', 'Problems', 'im_id', 'im_caption', 'imgurl'])
imgbase = 'https://medpix.nlm.nih.gov/images/full/'

def get_name(im_id):
    lol = re.search('.*_(.*)', im_id).group(1)
    return lol

datadir = '/n/scratch3/users/a/anp2971/datasets/medpix/'
for u in urls:
    print(u)
    r = requests.get(u)
    myfiles = r.json()
    file_list = myfiles['list']
    for f in file_list:
        break
        myDF = myDF.append({'uid':f['uid'], 'pmcid':f['pmcid'], 'title':f['title'], 'fulltext_html_url':f['fulltext_html_url'],
                     'medpixImageURL':f['medpixImageURL'], 'Problems':f['Problems'],
                     'im_id':f['image']['id'], 'im_caption':f['image']['caption'], 'imgurl':f['imgLarge']},
                ignore_index = True)
        imid = f['image']['id']
        imageurl = imgbase + get_name(imid) + '.jpg'
        #print(imageurl)

        f = open(datadir + imid + '.jpg', 'wb')
        f.write(ur.urlopen(imageurl).read())
        f.close()
    break


print(myDF.head())
#myDF.to_csv('/n/scratch3/users/a/anp2971/datasets/' + 'medpix_info.csv')



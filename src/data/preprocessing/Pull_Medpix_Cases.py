import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from requests_html import HTMLSession

session = HTMLSession()

medpix_info = pd.read_csv('/n/data2/hms/dbmi/beamlab/medpix/medpix_info.csv')

for i in np.arange(medpix_info.shape[0]):
	if (i % 100 == 0):
		print(str(i) + '/' + str(medpix_info.shape[0]))
	url = medpix_info['medpixImageURL'][i]# + '#content'
	print(url)
	page = session.get(url)
	page.html.render()

	parsed = BeautifulSoup(page.html.html, "html.parser")
	findings = parsed.find_all(class_="section ng-scope")
	for f in findings:
		if not f.text.isspace():
			print(f)
			print(f.text)
	break
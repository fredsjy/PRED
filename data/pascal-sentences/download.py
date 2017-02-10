import urllib.request
import os
import re
from scrapy import selector

url = "http://vision.cs.uiuc.edu/pascal-sentences/"
page = urllib.request.urlopen(url).read()
sel = selector.Selector(text=page)

# select images urls and lables
imgs_sel = sel.xpath('//@src')
imgs_url = []
lables = []
for i in imgs_sel:
    s = i.extract()
    imgs_url.append('http://vision.cs.uiuc.edu/pascal-sentences/'+s)
    lables.append(s.split('/')[0])

# select captions
nodes_captions = sel.xpath('//td/table')
captions = []
for node in nodes_captions:
    captions.append(re.split(r'</td></tr>\n<tr><td>\s*', re.sub(r'(<\S+><\S+>\n)*<\/*table>(\n*<\S+><\S+>\s)*', '',node.extract())))

# create directory
path_imgs = r'ps_images'
path_labels = r'labels.txt'
path_captions = r'ps_captions'
if not os.path.exists(path_imgs):
    os.makedirs(path_imgs)
if not os.path.exists(path_captions):
    os.makedirs(path_captions)

# write captions
for i in range(len(imgs_url)):
    fileName = path_imgs + r"/{0}.jpg".format(i+1)
    urllib.request.urlretrieve(imgs_url[i], fileName)
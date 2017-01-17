import urllib.request
import os

urls = []
with open("SBU_captioned_photo_dataset_urls.txt") as file_urls:
    line = file_urls.readline()
    while line:
        urls.append(line)
        line = file_urls.readline()

captions = []
with open("SBU_captioned_photo_dataset_captions.txt") as file_cap:
    line = file_cap.readline()
    while line:
        captions.append(line)
        line = file_cap.readline()

path = r"sbu_images/"
if not os.path.exists(path):
    os.makedirs(path)

for i in range(100):  # length of urls needing to be downloaded
    # data = urllib.request.urlopen(urls[i]).read()
    fileName = path + r"{0}.{1}.jpg".format(i+1,captions[i])
    urllib.request.urlretrieve(urls[i], fileName)


import os
import shutil

path_images_original = r'ps_images/'
path_images_train = r'ps_images_train/'
path_images_test = r'ps_images_test/'
path_images_valid = r'ps_images_valid/'

path_captions_original = r'ps_captions/'
path_captions_train = r'ps_captions_train/'
path_captions_test = r'ps_captions_test/'
path_captions_valid = r'ps_captions_valid/'

# divide images captions
# for i in range(1,1001):
#     if i % 10 == 7 or i % 10 == 8:
#         #if os.path.exists(path_images_original+'{0}.jpg'.format(i)):
#         shutil.copy2(path_images_original+'{0}.jpg'.format(i),path_images_test+'{0}.jpg'.format(i))
#         shutil.copy2(path_captions_original + '{0}.txt'.format(i), path_captions_test + '{0}.txt'.format(i))
#     elif i % 10 == 9 or i % 10 == 0:
#         shutil.copy2(path_images_original + '{0}.jpg'.format(i), path_images_valid + '{0}.jpg'.format(i))
#         shutil.copy2(path_captions_original + '{0}.txt'.format(i), path_captions_valid + '{0}.txt'.format(i))
#     else:
#         shutil.copy2(path_images_original + '{0}.jpg'.format(i), path_images_train + '{0}.jpg'.format(i))
#         shutil.copy2(path_captions_original + '{0}.txt'.format(i), path_captions_train + '{0}.txt'.format(i))

#divde labels
with open('labels.txt','r') as labels_original, \
        open('labels_train.txt', 'w') as labels_train,\
        open('labels_test.txt', 'w') as labels_test,\
        open('labels_valid.txt', 'w') as labels_valid:
    for i in range(1, 1001):
        line = labels_original.readline()
        ii = i % 10
        if ii == 7 or ii == 8:
            labels_test.write(line)
        elif ii == 9 or ii ==0:
            labels_valid.write(line)
        else:
            labels_train.write(line)


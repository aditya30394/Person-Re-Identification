import os,sys

#IDs of all the training images are extracted
train_images = os.listdir('./bounding_box_train')

idx = [int(im.split('_')[0]) for im in train_images]

#this will write duplicates as well, but later all the duplicate ids are removed.
with open('train_idx.txt', 'w') as f:
    for item in idx:
       f.write("%s\n" % item)

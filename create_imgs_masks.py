import os
import json
import cv2
import numpy as np
from random import shuffle


# ### Read all the images in a dict together with plate coordinates

dir_images = '/ssd480/data/metadata'
path_to_images = '/ssd480/data/nn_images/'
path_to_save = '/ssd480/grisha/images/'
all_images = {}


num_train = 1000
num_test = 100
shift = 5


print('Loading json files')
files = os.listdir(dir_images)
json_list = []
for file in files:
    if file.endswith(".json"):
        json_list.append(file)

data_all = []
for json_file in json_list:
    with open(dir_images+json_file) as f:
        data = json.load(f)
        data_all.append(data)
        for i, item in enumerate(data['results']):
            
            # add first image from two
            img_name = item['firstOct']['photoProof']['link'].split('/')[-1]
            left = item['firstOct']['photoProof']['bounds']['leftBorder']
            top = item['firstOct']['photoProof']['bounds']['topBorder']
            right = item['firstOct']['photoProof']['bounds']['rightBorder']
            bottom = item['firstOct']['photoProof']['bounds']['bottomBorder']
            all_images[img_name] = (left, top - shift, right, bottom + shift)
            
            # add second image from two
            img_name = item['secondOct']['photoProof']['link'].split('/')[-1]
            left = item['secondOct']['photoProof']['bounds']['leftBorder']
            top = item['secondOct']['photoProof']['bounds']['topBorder']
            right = item['secondOct']['photoProof']['bounds']['rightBorder']
            bottom = item['secondOct']['photoProof']['bounds']['bottomBorder']
            all_images[img_name] = (left, top - shift, right, bottom + shift)


# ### Copy images and create masks

def create_mask(filename, area):
    im = cv2.imread(filename)
    size = (im.shape[0], im.shape[1])
    x1, x2 = area[0], area[2]
    y1, y2 = area[1], area[3]
    mask = np.zeros(size, dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def copy_images(images, type_of_set):
    for img in images:
        try:
            os.system('cp ' + path_to_images + img + ' ' + path_to_save + type_of_set)
            mask = create_mask(os.path.join(path_to_save + type_of_set, img), all_images[img])
            path_to_save_mask = os.path.join(path_to_save + type_of_set, img.split('.')[0] + '_mask.jpg')
            cv2.imwrite(path_to_save_mask, mask)
        except:
            pass


print('Copying images to train and test folders')
images = list(all_images.keys())
shuffle(images)
images_train = images[:num_train]
shuffle(images)
images_test = images[:num_test]
copy_images(images_train, 'train')
copy_images(images_test, 'test')


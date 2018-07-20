from os import listdir
from os.path import join, exists
import json
import cv2
import sys
import numpy as np
from skimage.io import imread
from random import shuffle
from utils.unet import img_cols, img_rows


dir_jsons = '/ssd480/data/metadata/'
path_images = '/ssd480/data/nn_images'
# dir_jsons = '/home/grigorii/Desktop/primary_search/2017-10-03T00_00_01__2017-11-01T00_00_00'
# path_images = '/home/grigorii/Desktop/primary_search/2017-10-03T00_00_01__2017-11-01T00_00_00/nn_images'

if len(sys.argv) > 1:
    num_train = int(sys.argv[1])
    num_valid = int(sys.argv[2])
else:
    num_train = 1000
    num_valid = 20


def printing(s):
    print('-' * 30)
    print(s)
    print('-' * 30)


def load_jsons():
    if exists('all_files.json'):
        with open('all_files.json', 'r') as f:
            all_images = json.load(f)
        return all_images

    all_images = {}
    shift = 5

    printing('Loading json files')
    files = listdir(dir_jsons)
    json_list = []
    for file in files:
        if file.endswith(".json"):
            json_list.append(file)

    data_all = []
    for json_file in json_list:
        with open(join(dir_jsons, json_file)) as f:
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

    with open('all_files.json', 'w') as outfile:
        json.dump(all_images, outfile)
    return all_images


def reduced_dict(all_images, images_list):
    new_dict = {x: all_images[x] for x in images_list}
    return new_dict


def create_mask(filename, area):
    im = cv2.imread(filename)
    size = (im.shape[0], im.shape[1])
    x1, x2 = area[0], area[2]
    y1, y2 = area[1], area[3]
    mask = np.zeros(size, dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def check_images_existence(imgs):
    count = 0
    for item in imgs:
        if not exists(join(path_images, item)):
            imgs.remove(item)
            count += 1
    printing(str(count) + ' images deleted from dataset')


def get_images_to_train():
    all_images = load_jsons()
    images = list(all_images.keys())
    shuffle(images)
    images_t = images[:num_train]
    check_images_existence(images_t)
    images_d = reduced_dict(all_images, images_t)
    return images_t, images_d


images_train, images_dict = get_images_to_train()
def generator(batch_size):
    while True:
        shuffle(images_train)
        for i in range(0, len(images_train), batch_size):
            batch_list = images_train[i:i + batch_size]
            imgs = np.ndarray((batch_size, img_rows, img_cols), dtype=np.float32)
            imgs_mask = np.ndarray((batch_size, img_rows, img_cols), dtype=np.float32)

            for j, item in enumerate(batch_list):
                img = imread(join(path_images, item), as_gray=True)
                img_mask = create_mask(join(path_images, item), images_dict[item])

                img = cv2.resize(img, (img_cols, img_rows))
                img_mask = cv2.resize(img_mask, (img_cols, img_rows), interpolation=cv2.INTER_LANCZOS4)

                img /= 255.
                # img_mask /= 255.

                img = np.array([img])
                img_mask = np.array([img_mask])

                imgs[j] = img
                imgs_mask[j] = img_mask

                imgs = imgs[..., np.newaxis]
                imgs_mask = imgs_mask[..., np.newaxis]

            yield (imgs, imgs_mask)

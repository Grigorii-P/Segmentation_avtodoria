import os
from os import listdir
from os.path import join, exists
import json
import cv2
import numpy as np
from skimage.io import imread
from random import shuffle, uniform, choice
from utils.unet import img_cols, img_rows


path_to_npy = '/ssd480/grisha/data_npy/'
dir_jsons = '/ssd480/data/metadata/'
path_all_images = '/ssd480/data/nn_images'
path_train_test_imgs = '/ssd480/grisha/images/'

# dir_jsons = '/home/grigorii/Desktop/primary_search/2017-10-03T00_00_01__2017-11-01T00_00_00'
# path_all_images = '/home/grigorii/Desktop/primary_search/2017-10-03T00_00_01__2017-11-01T00_00_00/nn_images'

num_train = 40000
num_valid = 200
all_images = {}


def printing(s):
    print('-' * 30)
    print(s)
    print('-' * 30)


def create_npy(data_type):
    if os.path.exists(os.path.join(path_to_npy, data_type + '.npy')):
        return

    path = os.path.join(path_train_test_imgs, data_type)
    images = os.listdir(path)
    total = round(len(images) / 2)

    imgs = np.ndarray((total, img_rows, img_cols), dtype=np.float32)
    imgs_mask = np.ndarray((total, img_rows, img_cols), dtype=np.float32)

    i = 0
    printing('Creating '+data_type+' images...')
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.jpg'
        img = imread(os.path.join(path, image_name), as_gray=True)
        img_mask = imread(os.path.join(path, image_mask_name), as_gray=True)

        img = cv2.resize(img, (img_cols, img_rows))
        img_mask = cv2.resize(img_mask, (img_cols, img_rows), interpolation=cv2.INTER_LANCZOS4)
        img_mask = np.array(img_mask, dtype=np.float32)
        img_mask /= 255.

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(os.path.join(path_to_npy, data_type + '.npy'), imgs)
    np.save(os.path.join(path_to_npy, data_type + '_mask.npy'), imgs_mask)
    print('Saving to .npy files done.')


def load_data(data_type):
    imgs_train = np.load(os.path.join(path_to_npy, data_type + '.npy'))
    imgs_mask_train = np.load(os.path.join(path_to_npy, data_type + '_mask.npy'))
    return imgs_train, imgs_mask_train


def load_jsons():
    global all_images

    if exists('all_files.json'):
        with open('all_files.json', 'r') as f:
            all_images = json.load(f)
        return all_images

    shift = 5

    printing('Loading json files')
    files = listdir(dir_jsons)
    json_list = []
    for file in files:
        if file.endswith(".json"):
            json_list.append(file)

    for json_file in json_list:
        with open(join(dir_jsons, json_file)) as f:
            data = json.load(f)
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


def create_mask(shape, area):
    x1, x2 = area[0], area[2]
    y1, y2 = area[1], area[3]
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def check_images_existence(images_train):
    count = 0
    new_list = images_train.copy()
    for item in new_list:
        if not exists(join(path_all_images, item)):
            images_train.remove(item)
            count += 1
    printing(str(count) + ' images deleted from dataset')


def import_images_to_train():
    global all_images

    load_jsons()
    images = list(all_images.keys())
    shuffle(images)
    images_train = images[:num_train]
    check_images_existence(images_train)
    images_dict = {x: all_images[x] for x in images_train}
    return images_train, images_dict


def generator(batch_size, images_train, images_dict):
    shapes = [(1700, 1700), (2000, 2000)]
    p_decrease_size = 0.5
    p_inverse = 0.2

    while True:
        shuffle(images_train)
        for i in range(0, len(images_train), batch_size):
            batch_list = images_train[i:i + batch_size]
            imgs = np.ndarray((batch_size, img_rows, img_cols), dtype=np.float32)
            imgs_mask = np.ndarray((batch_size, img_rows, img_cols), dtype=np.float32)

            for j, item in enumerate(batch_list):
                area = images_dict[item]
                img = imread(join(path_all_images, item), as_gray=True)
                shape = (img.shape[0], img.shape[1])

                try:
                    p = uniform(0, 1)
                    if p <= p_decrease_size:
                        shape = choice(shapes)
                        bg = np.zeros(shape=shape)
                        center = (round(shape[0] / 2), round(shape[1] / 2))
                        x_offset = center[0] - area[0]
                        y_offset = center[1] - area[1]
                        # crashes at this point due to borders
                        bg[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = img
                        img = bg
                        area = (area[0] + x_offset, area[1] + y_offset, area[2] + x_offset, area[3] + y_offset)
                except:
                    shape = (img.shape[0], img.shape[1])

                p = uniform(0, 1)
                if p <= p_inverse:
                    img = (1. - img)

                img_mask = create_mask(shape, area)
                img = cv2.resize(img, (img_cols, img_rows))
                img_mask = cv2.resize(img_mask, (img_cols, img_rows), interpolation=cv2.INTER_LANCZOS4)
                img_mask = np.array(img_mask, dtype=np.float32)
                img_mask /= 255.

                img = np.array([img])
                img_mask = np.array([img_mask])
                imgs[j] = img
                imgs_mask[j] = img_mask

            imgs = imgs[..., np.newaxis]
            imgs_mask = imgs_mask[..., np.newaxis]

            yield (imgs, imgs_mask)

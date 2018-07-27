import os
from os import listdir
from os.path import join, exists
import json
import cv2
import numpy as np
from skimage.io import imread
from random import shuffle, uniform, choice, randint
from utils.unet import img_cols, img_rows

# dir_jsons = '/home/grigorii/Desktop/primary_search/2017-10-03T00_00_01__2017-11-01T00_00_00'
# path_all_images = '/home/grigorii/Desktop/primary_search/2017-10-03T00_00_01__2017-11-01T00_00_00/nn_images'

path_to_npy = '/ssd480/grisha/data_npy/'
dir_jsons = '/ssd480/data/metadata/'
path_all_images = '/ssd480/data/nn_images'
path_train_test_imgs = '/ssd480/grisha/images/'

num_train = 40000
num_valid = 200
all_images = {}
shapes = [[1000, 1000], [1100, 1100], [1200, 1200]]
p_decrease_size = -1.0  # overlay the target image on a black screen of a bigger size
p_inverse = 0.01  # either to make an image inversed

# decrease window size in this case - not 512x512 !!
p_prepodobniy_linux = 1.0  # crop plates with a little shift for both axes


def printing(s):
    print('-' * 30)
    print(s)
    print('-' * 30)


def create_npy_from_folder(data_type):
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


#TODO more comments
def create_valid_npy_for_generator(images_valid):
    # for comments take a look at 'generator(...) below'
    shuffle(images_valid)

    imgs = np.ndarray((num_valid, img_rows, img_cols), dtype=np.float32)
    imgs_mask = np.ndarray((num_valid, img_rows, img_cols), dtype=np.float32)

    printing('Creating validation .npy files...')
    for i, item in enumerate(images_valid):
        area = all_images[item]
        img = imread(join(path_all_images, item), as_gray=True)
        #TODO think about it
        # img /= img.max()
        shape = [img.shape[0], img.shape[1]]

        p = uniform(0, 1)
        if p <= p_decrease_size:
            shape = choice(shapes)
            img_shape = [img.shape[0], img.shape[1]]
            while not (shape[0] >= img_shape[0] and shape[1] >= img_shape[1]):
                shape[0] = img_shape[0] + 200
                shape[1] = img_shape[1] + 200
            bg = np.zeros(shape=shape)
            rand_shift_x = randint(0, bg.shape[1] - img.shape[1])
            rand_shift_y = randint(0, bg.shape[0] - img.shape[0])
            bg[rand_shift_y:rand_shift_y + img.shape[0], rand_shift_x:rand_shift_x + img.shape[1]] = img
            img = bg
            area[0] += rand_shift_x
            area[2] += rand_shift_x
            area[1] += rand_shift_y
            area[3] += rand_shift_y
        else:
            p = uniform(0, 1)
            if p <= p_prepodobniy_linux:
                shift = 40
                dx = area[2] - area[0]
                dy = area[3] - area[1]
                shape = [2 * shift + dy, 2 * shift + dx]
                if not (area[1] - shift < 0 or area[3] + shift > img.shape[0] or
                        area[0] - shift < 0 or area[2] + shift > img.shape[1]):
                    img = img[area[1] - shift: area[3] + shift, area[0] - shift:area[2] + shift]
                else:
                    bg = np.zeros(shape=shape)
                    try:
                        bg[shift:shift + dy, shift:shift + dx] = img[area[1]:area[3], area[0]:area[2]]
                    except ValueError:
                        sh = img[area[1]:area[3], area[0]:area[2]].shape
                        bg[shift:shift + sh[0], shift:shift + sh[1]] = img[area[1]:area[3], area[0]:area[2]]
                    img = bg
                area = [shift, shift, dx + shift, dy + shift]

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
        imgs[i] = img
        imgs_mask[i] = img_mask

    printing('Loading done.')
    np.save(os.path.join(path_to_npy, 'valid.npy'), imgs)
    np.save(os.path.join(path_to_npy, 'valid_mask.npy'), imgs_mask)
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
    printing(str(count) + ' image(-s) deleted from dataset')


def import_images_train_valid():
    global all_images

    load_jsons()
    images = list(all_images.keys())
    shuffle(images)
    images_train = images[:num_train]
    shuffle(images)
    images_valid = images[:num_valid]
    check_images_existence(images_train)
    check_images_existence(images_valid)
    images_dict = {x: all_images[x] for x in images_train}
    return images_train, images_valid, images_dict


def generator(batch_size, images_train, images_dict):
    while True:
        shuffle(images_train)
        for i in range(0, len(images_train), batch_size):
            batch_list = images_train[i:i + batch_size]
            imgs = np.ndarray((batch_size, img_rows, img_cols), dtype=np.float32)
            imgs_mask = np.ndarray((batch_size, img_rows, img_cols), dtype=np.float32)

            for j, item in enumerate(batch_list):
                area = images_dict[item]
                img = imread(join(path_all_images, item), as_gray=True)
                #TODO think about it
                # img /= img.max()
                shape = [img.shape[0], img.shape[1]]

                p = uniform(0, 1)
                # decrese the size of a car relative to the screen size
                if p <= p_decrease_size:
                    shape = choice(shapes)
                    img_shape = [img.shape[0], img.shape[1]]
                    while not (shape[0] >= img_shape[0] and shape[1] >= img_shape[1]):
                        shape[0] = img_shape[0] + 200
                        shape[1] = img_shape[1] + 200
                    #TODO add real image as background instead of black box
                    bg = np.zeros(shape=shape)
                    rand_shift_x = randint(0, bg.shape[1] - img.shape[1])
                    rand_shift_y = randint(0, bg.shape[0] - img.shape[0])
                    bg[rand_shift_y:rand_shift_y + img.shape[0], rand_shift_x:rand_shift_x + img.shape[1]] = img
                    img = bg
                    area[0] += rand_shift_x
                    area[2] += rand_shift_x
                    area[1] += rand_shift_y
                    area[3] += rand_shift_y
                else:
                    p = uniform(0, 1)
                    # if we dont decrease a relative size, we take cropped image
                    # as if it was an output from VJ, then increase its size by 'shift'
                    if p <= p_prepodobniy_linux:
                        shift = 40
                        dx = area[2] - area[0]
                        dy = area[3] - area[1]
                        shape = [2 * shift + dy, 2 * shift + dx]

                        if not (area[1] - shift < 0 or area[3] + shift > img.shape[0] or
                                area[0] - shift < 0 or area[2] + shift > img.shape[1]):
                            img = img[area[1] - shift: area[3] + shift, area[0] - shift:area[2] + shift]
                        else:
                            bg = np.zeros(shape=shape)
                            try:
                                bg[shift:shift + dy, shift:shift + dx] = img[area[1]:area[3], area[0]:area[2]]
                            except ValueError:
                                sh = img[area[1]:area[3], area[0]:area[2]].shape
                                bg[shift:shift + sh[0], shift:shift + sh[1]] = img[area[1]:area[3], area[0]:area[2]]
                            img = bg
                        area = [shift, shift, dx + shift, dy + shift]

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

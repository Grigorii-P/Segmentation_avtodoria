import os
import cv2
import numpy as np
from skimage.io import imread


data_path = '/home/grigorii/Desktop/Segmentation/images/'
path_to_npy = '/home/grigorii/Desktop/Segmentation/data_npy/'

img_rows = 128
img_cols = 128


def create_npy(data_type):

    if os.path.exists(os.path.join(path_to_npy, data_type+'.npy')):
        return

    path = os.path.join(data_path, data_type)
    images = os.listdir(path)
    total = round(len(images) / 2)

    imgs = np.ndarray((total, img_rows, img_cols), dtype=np.float32)
    imgs_mask = np.ndarray((total, img_rows, img_cols), dtype=np.float32)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.jpg'
        img = imread(os.path.join(path, image_name), as_gray=True)
        img_mask = imread(os.path.join(path, image_mask_name), as_gray=True)
        
        img = cv2.resize(img, (img_cols, img_rows))
        img_mask = cv2.resize(img_mask, (img_cols, img_rows), interpolation=cv2.INTER_LANCZOS4)
        
        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(os.path.join(path_to_npy, data_type+'.npy'), imgs)
    np.save(os.path.join(path_to_npy, data_type+'_mask.npy'), imgs_mask)
    print('Saving to .npy files done.')


def load_data(data_type):
    imgs_train = np.load(os.path.join(path_to_npy, data_type+'.npy'))
    imgs_mask_train = np.load(os.path.join(path_to_npy, data_type+'_mask.npy'))
    return imgs_train, imgs_mask_train

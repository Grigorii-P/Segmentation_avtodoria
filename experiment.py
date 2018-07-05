import os
import cv2
import numpy as np
from skimage.io import imread


data_path = '/home/grigorii/Desktop/Segmentation/images/'
image_rows = 300
image_cols = 500

def create_train_data():
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    total = round(len(images) / 2)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.float32)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.float32)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.jpg'
        img = imread(os.path.join(train_data_path, image_name), as_gray=True)
        img_mask = imread(os.path.join(train_data_path, image_mask_name), as_gray=True)
        
        img = cv2.resize(img, (image_cols, image_rows))
        img_mask = cv2.resize(img_mask, (image_cols, image_rows), interpolation=cv2.INTER_LANCZOS4)
        
        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('train.npy', imgs)
    np.save('train_mask.npy', imgs_mask)
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load('train.npy')
    imgs_mask_train = np.load('train_mask.npy')
    return imgs_train, imgs_mask_train
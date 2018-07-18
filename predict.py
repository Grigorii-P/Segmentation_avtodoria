# from __future__ import print_function
from unet import unet_original
import numpy as np
import cv2
import os
from skimage.io import imsave
from train import preprocess
from create_npy import load_data
from train import path_weights


# data_path = '/home/grigorii/Desktop/Segmentation/images/test_two_cars'
# path_to_save_preds = '/home/grigorii/Desktop/Segmentation/images/test_two_cars_preds'
# path_to_npy = '/home/grigorii/Desktop/Segmentation/data_npy/'
# path_weights = os.path.join('/home/grigorii/Desktop/Segmentation', 'weights.h5')
data_path = '/ssd480/grisha/images/test'
path_to_save_preds = '/ssd480/grisha/images/preds'


def predict():
    model = unet_original()
    model.load_weights(path_weights)

    imgs_test = load_data('test')
    imgs_test = imgs_test.astype('float32')
    mean = np.mean(imgs_test)
    std = np.std(imgs_test)
    imgs_test -= mean
    imgs_test /= std
    imgs_test = preprocess(imgs_test)

    pred_mask_test = model.predict(imgs_test, verbose=1)
    # np.save(os.path.join(path_to_npy, 'pred_mask_test.npy'), pred_mask_test)

    for i, image in enumerate(pred_mask_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        image = cv2.resize(image, (300, 300))
        imsave(os.path.join(path_to_save_preds, str(i) + '_pred.png'), image)

    imgs_test *= std
    imgs_test += mean
    for i, image in enumerate(imgs_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        image = cv2.resize(image, (300, 300))
        imsave(os.path.join(path_to_save_preds, str(i) + '_origin.png'), image)


if __name__ == '__main__':
    predict()

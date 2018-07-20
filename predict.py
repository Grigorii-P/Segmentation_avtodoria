# from __future__ import print_function
import numpy as np
import cv2
import os
from skimage.io import imread, imsave
from train import preprocess, path_weights, model, load_data, path_to_npy
from utils.unet import img_rows, img_cols


# data_path = '/home/grigorii/Desktop/Segmentation/images/test_two_cars'
# path_to_save_preds = '/home/grigorii/Desktop/Segmentation/images/test_two_cars_preds'
# path_to_npy = '/home/grigorii/Desktop/Segmentation/data_npy/'
# path_weights = os.path.join('/home/grigorii/Desktop/Segmentation', 'weights.h5')
data_path = '/ssd480/grisha/images/images_hard_to_recognize'
path_to_save_preds = '/ssd480/grisha/images/preds'


def load_isolated_dataset():
    path = os.path.join(data_path)
    images = os.listdir(path)
    total = len(images)

    imgs = np.ndarray((total, img_rows, img_cols), dtype=np.float32)

    i = 0
    for image_name in images:
        img = imread(os.path.join(path, image_name), as_gray=True)
        img = cv2.resize(img, (img_cols, img_rows))

        img = np.array([img])
        imgs[i] = img
        i += 1

    return imgs


def predict():
    model.load_weights(path_weights)

    # imgs_test, _ = load_data('test')
    imgs_test = load_isolated_dataset()
    print()
    print(type(imgs_test))
    print()
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

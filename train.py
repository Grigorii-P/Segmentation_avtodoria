# from __future__ import print_function
import numpy as np
import cv2
import os
from utils.unet import unet_small, img_rows, img_cols
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.utils import plot_model
from skimage.io import imread


# path_to_npy = '/home/grigorii/Desktop/Segmentation/data_npy/'
# path_weights = os.path.join('/home/grigorii/Desktop/Segmentation', 'weights.h5')
# images_path = '/home/grigorii/Desktop/Segmentation/images/'
path_to_npy = '/ssd480/grisha/data_npy/'
path_weights = '/ssd480/grisha/weights.h5'
images_path = '/ssd480/grisha/images'


def tf_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.log_device_placement = True
    config.gpu_options.visible_device_list = "0"
    sess = tf.Session(config=config)
    set_session(sess)


def printing(s):
    print('-' * 30)
    print(s)
    print('-' * 30)


def create_npy(data_type):
    if os.path.exists(os.path.join(path_to_npy, data_type + '.npy')):
        return

    path = os.path.join(images_path, data_type)
    images = os.listdir(path)
    total = round(len(images) / 2)

    imgs = np.ndarray((total, img_rows, img_cols), dtype=np.float32)
    imgs_mask = np.ndarray((total, img_rows, img_cols), dtype=np.float32)

    i = 0
    printing('Creating training images...')
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

    np.save(os.path.join(path_to_npy, data_type + '.npy'), imgs)
    np.save(os.path.join(path_to_npy, data_type + '_mask.npy'), imgs_mask)
    print('Saving to .npy files done.')


def load_data(data_type):
    imgs_train = np.load(os.path.join(path_to_npy, data_type + '.npy'))
    imgs_mask_train = np.load(os.path.join(path_to_npy, data_type + '_mask.npy'))
    return imgs_train, imgs_mask_train


def preprocess(imgs):
    imgs = imgs[..., np.newaxis]
    return imgs


def train():
    tf_config()

    printing('Creating npy files...')
    create_npy('train')
    create_npy('test')

    printing('Loading and preprocessing train and test data...')
    imgs_train, imgs_mask_train = load_data('train')

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)
    std = np.std(imgs_train)
    np.save(os.path.join(path_to_npy, 'mean_std_train.npy'), np.array([mean, std]))

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.

    printing('Creating and compiling model...')
    model = unet_small()
    plot_model(model, to_file='model_'+str(img_rows)+'x'+str(img_cols)+'.png', show_shapes=True, show_layer_names=True)
    model_checkpoint = ModelCheckpoint(path_weights, monitor='val_loss', save_best_only=True)

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    printing('Fitting model...')
    model.fit(imgs_train, imgs_mask_train, batch_size=128, epochs=15, verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint])
    printing('Done')


if __name__ == '__main__':
    train()

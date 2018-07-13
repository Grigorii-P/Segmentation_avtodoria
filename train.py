# from __future__ import print_function
from unet import *
from skimage.io import imsave
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.utils import plot_model


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
config.gpu_options.visible_device_list = "0"
sess = tf.Session(config=config)
set_session(sess)


path_to_save_preds = '/home/grigorii/Desktop/Segmentation/images/'
path_to_npy = '/home/grigorii/Desktop/Segmentation/data_npy/'
path_weights = os.path.join('/home/grigorii/Desktop/Segmentation', 'weights.h5')


def printing(s):
    print('-' * 30)
    print(s)
    print('-' * 30)


def preprocess(imgs):
    imgs = imgs[..., np.newaxis]
    return imgs


def train():
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
    model = get_unet()
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    model_checkpoint = ModelCheckpoint(path_weights, monitor='val_loss', save_best_only=True)

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    printing('Fitting model...')
    model.fit(imgs_train, imgs_mask_train, batch_size=32, epochs=15, verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint])
    printing('Done')


if __name__ == '__main__':
    train()
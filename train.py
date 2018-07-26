# from __future__ import print_function
import numpy as np
from utils.unet import unet_original, unet_small, img_rows, img_cols
from data_generator import printing, create_npy_from_folder, create_valid_npy_for_generator, \
    load_data, import_images_train_valid, num_train, generator
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.backend import get_session
from keras.backend.tensorflow_backend import set_session
from keras.utils import plot_model
from time import time


# path_to_npy = '/home/grigorii/Desktop/Segmentation/data_npy/'
# path_weights = os.path.join('/home/grigorii/Desktop/Segmentation', 'weights.h5')
# images_path = '/home/grigorii/Desktop/Segmentation/images/'

path_weights = '/ssd480/grisha/weights.h5'


def tf_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.log_device_placement = True
    config.gpu_options.visible_device_list = "0"
    sess = tf.Session(config=config)
    set_session(sess)


def train():
    tf_config()

    printing('Creating and compiling model...')
    model = unet_original()
    # model = unet_small()

    plot_model(model, to_file='model_'+str(img_rows)+'x'+str(img_cols)+'.png', show_shapes=True, show_layer_names=True)
    model_checkpoint = ModelCheckpoint(path_weights, monitor='val_loss', save_best_only=True)

    get_session().run(tf.global_variables_initializer())
    images_train, images_valid, images_dict = import_images_train_valid()
    create_valid_npy_for_generator(images_valid)
    validation_imgs, validation_mask = load_data('valid')

    batch = 10
    epochs = 8
    printing('Fitting model...')
    t0 = time()
    model.fit_generator(generator(batch_size=batch, images_train=images_train, images_dict=images_dict),
                        steps_per_epoch=round(num_train / batch),
                        epochs=epochs, validation_data=(validation_imgs, validation_mask),
                        verbose=1, callbacks=[model_checkpoint])
    printing('Trainin time - %.1f min' % ((time()-t0)/60))
    printing('Done')


if __name__ == '__main__':
    train()

import numpy as np
from time import time
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization
from keras.optimizers import Adam
from data_generator import printing
from keras.utils import plot_model
from utils.unet import dice_coef, dice_coef_loss


average_among = 100
depth = 1
verbose = 0
# shapes = [[96, 96], [256, 256], [512, 512]]
shapes = [[256, 256]]


def unet_original(img_rows, img_cols):
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='CONV_1')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    # conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='CONV_2')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    # conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', name='CONV_3')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    # conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', name='CONV_4')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    # conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='CONV_5')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    # conv5 = BatchNormalization()(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    # conv6 = BatchNormalization()(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    # conv7 = BatchNormalization()(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    # conv8 = BatchNormalization()(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    # conv9 = BatchNormalization()(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    # model.compile(optimizer=Adam(lr=1e-3), loss=losses.binary_crossentropy, metrics=[dice_coef])

    return model


def unet_small(img_rows, img_cols):
    inputs = Input((img_rows, img_cols, 1))

    init_kernel_num = 32

    conv1 = Conv2D(init_kernel_num, (3, 3), activation='relu', padding='same', name='CONV_1')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    # conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(init_kernel_num * 2, (3, 3), activation='relu', padding='same', name='CONV_2')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    # conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(init_kernel_num * 4, (3, 3), activation='relu', padding='same', name='CONV_3')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    # conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(init_kernel_num * 8, (3, 3), activation='relu', padding='same', name='CONV_4')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    # conv4 = BatchNormalization()(conv4)

    up7 = concatenate([Conv2DTranspose(init_kernel_num * 4, (2, 2), strides=(2, 2), padding='same', name='UP_7')(conv4), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', name='CONV_7')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    # conv7 = BatchNormalization()(conv7)

    up8 = concatenate([Conv2DTranspose(init_kernel_num * 2, (2, 2), strides=(2, 2), padding='same', name='UP_8')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', name='CONV_8')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    # conv8 = BatchNormalization()(conv8)

    up9 = concatenate([Conv2DTranspose(init_kernel_num, (2, 2), strides=(2, 2), padding='same', name='UP_9')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', name='CONV_9')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    # conv9 = BatchNormalization()(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid', name='CONV_10')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    # model.compile(optimizer=Adam(lr=1e-3), loss=losses.binary_crossentropy, metrics=[dice_coef])

    return model


# unets = {'Small': unet_small, 'Original': unet_original}
unets = {'Small': unet_small}
for unet in unets.keys():
    for shape in shapes:
        imgs = np.random.rand(depth, shape[0], shape[1], 1)
        model = unets[unet](shape[0], shape[1])
        plot_model(model, to_file='model.png', show_shapes=True,
                   show_layer_names=True)

        # t0 = time()
        # for j in range(average_among):
        #     preds = model.predict(imgs, verbose=verbose)
        #
        # printing('%s (%d, %d) %.3f sec' % (unet, shape[0], shape[1], ((time() - t0)/average_among)))


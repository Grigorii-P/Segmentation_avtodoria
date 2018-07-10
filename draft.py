import keras.backend as K
import numpy as np
from create_npy import *
from skimage.io import imsave


# smooth = 1.
# def dice_coef(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#
#
# y = np.array([0,0,1,1,0])
# yy = np.array([0,0,0.1,0.99,1])
#
# res = y * yy

preds = np.load('pred_mask_test.npy')

print()
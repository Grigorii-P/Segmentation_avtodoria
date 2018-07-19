from utils.unet import *
import cv2
from skimage.io import imread
import os
import numpy as np
from keras.utils import plot_model

model = unet_small()
plot_model(model, to_file='model' + str(img_rows) + 'x' + str(img_cols) + '.png', show_shapes=True,
           show_layer_names=True)


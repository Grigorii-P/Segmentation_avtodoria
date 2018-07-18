from unet import *
from keras.utils import plot_model

model = unet_original()
plot_model(model, to_file='model_test.png', show_shapes=True, show_layer_names=True)


print()

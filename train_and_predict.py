# from __future__ import print_function
from unet import *
from keras.callbacks import ModelCheckpoint

path_to_save_preds = '/home/grigorii/Desktop/Segmentation/images/'


def printing(s):
    print('-' * 30)
    print(s)
    print('-' * 30)


def preprocess(imgs):
    imgs = imgs[..., np.newaxis]
    return imgs


def train_and_predict():
    create_npy('train')
    create_npy('test')

    printing('Loading and preprocessing train and test data...')
    imgs_train, imgs_mask_train = load_data('train')

    imgs_train = imgs_train.astype('float32')
    # TODO save mean and std to .npy
    mean = np.mean(imgs_train)
    std = np.std(imgs_train)

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.

    printing('Creating and compiling model...')
    model = get_unet()
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    printing('Fitting model...')
    model.fit(imgs_train, imgs_mask_train, batch_size=32, epochs=10, verbose=1, shuffle=True,
              validation_split=0.15,
              callbacks=[model_checkpoint])

    printing('Loading and preprocessing test data...')
    imgs_test, imgs_mask_test = load_data('test')

    imgs_test = imgs_test.astype('float32')
    # TODO save to .npy
    imgs_test -= mean
    imgs_test /= std

    printing('Loading saved weights...')
    model.load_weights('weights.h5')

    imgs_test = preprocess(imgs_test)

    printing('Predicting masks on test data...')
    pred_mask_test = model.predict(imgs_test, verbose=1)
    np.save('pred_mask_test.npy', pred_mask_test)

    printing('Saving predicted masks to files...')
    pred_dir = 'preds'
    dir_to_save = os.path.join(path_to_save_preds, pred_dir)
    if not os.path.exists(dir_to_save):
        os.mkdir(dir_to_save)
    for i, image in enumerate(pred_mask_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(dir_to_save, str(i) + '_pred.png'), image)


if __name__ == '__main__':
    train_and_predict()
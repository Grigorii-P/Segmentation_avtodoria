# from __future__ import print_function
from unet import *
from skimage.io import imsave
from train import printing, preprocess


path_to_save_preds = '/home/grigorii/Desktop/Segmentation/images/'
path_to_npy = '/home/grigorii/Desktop/Segmentation/data_npy/'
path_weights = os.path.join('/home/grigorii/Desktop/Segmentation', 'weights.h5')


def predict():
    model = get_unet()

    printing('Loading and preprocessing test data...')
    imgs_test, imgs_mask_test = load_data('test')

    imgs_test = imgs_test.astype('float32')
    mean = np.mean(imgs_test)
    std = np.std(imgs_test)
    np.save(os.path.join(path_to_npy, 'mean_std_test.npy'), np.array([mean, std]))

    imgs_test -= mean
    imgs_test /= std

    printing('Loading saved weights...')
    model.load_weights(path_weights)

    imgs_test = preprocess(imgs_test)

    printing('Predicting masks on test data...')
    pred_mask_test = model.predict(imgs_test, verbose=1)
    np.save(os.path.join(path_to_npy, 'pred_mask_test.npy'), pred_mask_test)

    printing('Saving predicted masks to files...')
    pred_dir = 'preds'
    dir_to_save = os.path.join(path_to_save_preds, pred_dir)
    if not os.path.exists(dir_to_save):
        os.mkdir(dir_to_save)
    for i, image in enumerate(pred_mask_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        image = cv2.resize(image, (300, 300))
        imsave(os.path.join(dir_to_save, str(i) + '_pred.png'), image)

    imgs_test, _ = load_data('test')
    for i, image in enumerate(imgs_test):
        image = (image[:, :] * 255.).astype(np.uint8)
        image = cv2.resize(image, (300, 300))
        imsave(os.path.join(dir_to_save, str(i) + '_test.png'), image)


if __name__ == '__main__':
    predict()

import glob
import os
import warnings
from os.path import dirname, join

import cv2
import numpy as np
import numpy.typing as npt
from dotenv import load_dotenv
from keras.preprocessing.image import load_img, img_to_array
from sklearn.decomposition import PCA

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter('ignore', DeprecationWarning)

load_dotenv(verbose=True)

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

AFC = os.environ.get("after_cv2")
IMP = os.environ.get("images_path")
IMF = os.environ.get("images_file")
TRF = os.environ.get("train_folder")
VAF = os.environ.get("validation_folder")


def compare_all():
    after = cv2.imread(str(AFC))
    after_rgb = cv2.cvtColor(after, cv2.COLOR_RGB2BGR)
    after_rgb_sh = after_rgb.shape  # (262, 350, 3)

    train_dir = str(TRF)
    test_dir = str(VAF)
    file_type = str(IMF)

    train_list_path = glob.glob(str(IMP) + train_dir + '/*.' + file_type)
    test_list_path = glob.glob(str(IMP) + test_dir + '/*.' + file_type)

    train_list = []
    test_list = []
    train_shape = after_rgb_sh
    test_shape = (256, 256, 3)

    for img in train_list_path:
        temp_img = load_img(img, target_size=train_shape)
        temp_img_array = img_to_array(temp_img) / 255
        train_list.append(temp_img_array)

    for img in test_list_path:
        temp_img = load_img(img, target_size=test_shape)
        temp_img_array = img_to_array(temp_img) / 255
        test_list.append(temp_img_array)

    x_train = np.array(train_list)
    x_test = np.array(test_list)

    pca = PCA(
        n_components=1,
        copy=True,
        whiten=True,
        svd_solver='auto',
        tol=0.0,
        iterated_power='auto',
        n_oversamples=10,
        power_iteration_normalizer='auto',
        random_state=0,
    )

    x_train_shape = np.array(x_train).reshape(-1, 1)
    x_test_shape = np.array(x_test).reshape(-1, 1)

    pca.fit(x_train_shape)
    pca.fit(x_test_shape)

    train_transform = pca.inverse_transform(pca.transform(x_train_shape))
    test_transform = pca.inverse_transform(pca.transform(x_test_shape))

    count_file_train = (sum(os.path.isfile(os.path.join(IMP + TRF, name)) for name in os.listdir(IMP + TRF)))
    count_file_validation = (sum(os.path.isfile(os.path.join(IMP + VAF, name)) for name in os.listdir(IMP + VAF)))

    result: npt.DTypeLike = np.floor(sum(train_transform) / 1000).astype(int) / (1000 * count_file_train)
    test_result: npt.DTypeLike = np.floor(sum(test_transform) / 1000).astype(int) / (1000 * count_file_validation)

    print("Approximate value : {:.2f}".format(np.float64(result)) + " in " + train_dir + " folder.")
    # Approximate value : 0.15 in train folder.

    print("Approximate value : {:.2f}".format(np.float64(test_result)) + " in " + test_dir + " folder.")
    # Approximate value : 0.11 in validation folder.


def compare_train():
    after = cv2.imread(str(AFC))
    after_rgb = cv2.cvtColor(after, cv2.COLOR_RGB2BGR)
    after_rgb_sh = after_rgb.shape  # (262, 350, 3)

    train_dir = str(TRF)
    file_type = str(IMF)

    train_list_path = glob.glob(str(IMP) + train_dir + '/*.' + file_type)

    train_list = []
    train_shape = after_rgb_sh

    for img in train_list_path:
        temp_img = load_img(img, target_size=train_shape)
        temp_img_array = img_to_array(temp_img) / 255
        train_list.append(temp_img_array)

    x_train = np.array(train_list)

    pca = PCA(
        n_components=1,
        copy=True,
        whiten=True,
        svd_solver='auto',
        tol=0.0,
        iterated_power='auto',
        n_oversamples=10,
        power_iteration_normalizer='auto',
        random_state=0,
    )

    x_train_shape = np.array(x_train).reshape(-1, 1)

    pca.fit(x_train_shape)

    train_transform = pca.inverse_transform(pca.transform(x_train_shape))

    count_file_train = (sum(os.path.isfile(os.path.join(IMP + TRF, name)) for name in os.listdir(IMP + TRF)))

    result: npt.DTypeLike = np.floor(sum(train_transform) / 1000).astype(int) / (1000 * count_file_train)

    print("{:.2f}".format(np.float64(result)))  # 0.15


def compare_validation():
    test_dir = str(VAF)
    file_type = str(IMF)

    test_list_path = glob.glob(str(IMP) + test_dir + '/*.' + file_type)

    test_list = []
    test_shape = (256, 256, 3)

    for img in test_list_path:
        temp_img = load_img(img, target_size=test_shape)
        temp_img_array = img_to_array(temp_img) / 255
        test_list.append(temp_img_array)

    x_test = np.array(test_list)

    pca = PCA(
        n_components=1,
        copy=True,
        whiten=True,
        svd_solver='auto',
        tol=0.0,
        iterated_power='auto',
        n_oversamples=10,
        power_iteration_normalizer='auto',
        random_state=0,
    )

    x_test_shape = np.array(x_test).reshape(-1, 1)

    pca.fit(x_test_shape)

    test_transform = pca.inverse_transform(pca.transform(x_test_shape))

    count_file_validation = (sum(os.path.isfile(os.path.join(IMP + VAF, name)) for name in os.listdir(IMP + VAF)))

    test_result: npt.DTypeLike = np.floor(sum(test_transform) / 1000).astype(int) / (1000 * count_file_validation)

    print("{:.2f}".format(np.float64(test_result)))  # 0.11

import glob
import os
import warnings

import cv2
import numpy as np
import numpy.typing as npt
from keras.preprocessing.image import load_img, img_to_array
from sklearn.decomposition import PCA

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter('ignore', DeprecationWarning)

after = cv2.imread('../images/Sample/after/face.gif')
after_rgb = cv2.cvtColor(after, cv2.COLOR_RGB2BGR)
after_rgb_sh = after_rgb.shape  # (262, 350, 3)

train_dir = 'train'
test_dir = 'validation'
file_type = 'gif'

train_list_path = glob.glob('../images/' + train_dir + '/*.' + file_type)
test_list_path = glob.glob('../images/' + test_dir + '/*.' + file_type)

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

hyoka_0: npt.DTypeLike = np.floor(train_transform[0] * 1000).astype(int) / 1000
hyoka_1: npt.DTypeLike = np.floor(train_transform[1] * 1000).astype(int) / 1000
hyoka_2: npt.DTypeLike = np.floor(train_transform[2] * 1000).astype(int) / 1000
hyoka_3: npt.DTypeLike = np.floor(train_transform[3] * 1000).astype(int) / 1000
hyoka_4: npt.DTypeLike = np.floor(train_transform[4] * 1000).astype(int) / 1000
hyoka_5: npt.DTypeLike = np.floor(train_transform[5] * 1000).astype(int) / 1000
result = (hyoka_0 + hyoka_1 + hyoka_2 + hyoka_3 + hyoka_4 + hyoka_5) / 6

print("Approximate value : {:.2f}".format(np.float64(result)) + " in " + train_dir + " folder.")
# Approximate value : 0.52 in train folder.

hyoka_test_0: npt.DTypeLike = np.floor(test_transform[0] * 1000).astype(int) / 1000
hyoka_test_1: npt.DTypeLike = np.floor(test_transform[1] * 1000).astype(int) / 1000
hyoka_test_2: npt.DTypeLike = np.floor(test_transform[2] * 1000).astype(int) / 1000
hyoka_test_3: npt.DTypeLike = np.floor(test_transform[3] * 1000).astype(int) / 1000
hyoka_test_4: npt.DTypeLike = np.floor(test_transform[4] * 1000).astype(int) / 1000
hyoka_test_5: npt.DTypeLike = np.floor(test_transform[5] * 1000).astype(int) / 1000
test_result = (hyoka_test_0 + hyoka_test_1 + hyoka_test_2 + hyoka_test_3 + hyoka_test_4 + hyoka_test_5) / 6

print("Approximate value : {:.2f}".format(np.float64(test_result)) + " in " + test_dir + " folder.")
# Approximate value : 0.74 in validation folder.

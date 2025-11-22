import glob
import os

import cv2
import numpy as np
import numpy.typing as npt
from keras.preprocessing.image import load_img, img_to_array
from sklearn.decomposition import PCA

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

after = cv2.imread('./images/Sample/after/face.gif')
after_rgb = cv2.cvtColor(after, cv2.COLOR_RGB2BGR)
after_rgb_sh = after_rgb.shape  # (262, 350, 3)

train_dir = 'train'
test_dir = 'validation'
file_type = 'gif'

train_list_path = glob.glob('./images/' + train_dir + '/*.' + file_type)
test_list_path = glob.glob('./images/' + test_dir + '/*.' + file_type)

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

# # train data
# after = cv2.imread('./images/Sample/after/face.gif')
# after_rgb = cv2.cvtColor(after, cv2.COLOR_RGB2BGR)
# # after_dt = after.dtype
# after_sh = after.shape  # (262, 350, 3)
# after_rgb_sh = after_rgb.shape  # (262, 350, 3)
# after_rgb_np = np.array(after_rgb_sh).reshape(-1, 1)
# after_np = np.array(after_sh).reshape(-1, 1)
# # py
# # [[262][350][3]]
# # [[262][350][3]]
#
# # test data
# before = cv2.imread('./images/Sample/before/face.gif')
# before_rgb = cv2.cvtColor(before, cv2.COLOR_RGB2BGR)
# before_dt = before.dtype
# before_sh = before.shape  # (262, 350, 3)
# before_rgb_sh = before_rgb.shape  # (262, 350, 3)
# before_rgb_np = np.array(before_rgb_sh).reshape(-1, 1)
# before_np = np.array(before_sh).reshape(-1, 1)
# # py
# # [[262][350][3]]
# # [[262][350][3]]
#
# # train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)
# X_train, X_test, Y_train, Y_test = train_test_split(after_rgb_np, before_rgb_np, test_size=0.3)

# print('X_train')
# print(X_train)
# [[350]
#  [262]]
# print('X_test')
# print(X_test)
# [[3]]

# kmeans = KMeans(
#     copy_x=True,
#     init='k-means++',
#     max_iter=300,
#     n_clusters=2,
#     n_init=10,
#     random_state=0,
#     tol=0.0001,
#     verbose=0,
# )

# x_kmean = kmeans.fit(X_train)
# x_reconstructed_means = kmeans.cluster_centers_[kmeans.predict(X_test)]
# print(x_reconstructed_means)  # [[262.]]

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

# print("PCA component shape: {}".format(pca.components_.shape))
# PCA component shape: (1, 1)
# print(train_transform.shape[0])
# # 1650600 <class 'tuple'>
# print(test_transform.shape[0])
# 1376256 <class 'tuple'>

# print(type(train_transform[0]))
hyoka_0: npt.DTypeLike = np.floor(train_transform[0] * 1000).astype(int) / 1000
hyoka_1: npt.DTypeLike = np.floor(train_transform[1] * 1000).astype(int) / 1000
hyoka_2: npt.DTypeLike = np.floor(train_transform[2] * 1000).astype(int) / 1000
hyoka_3: npt.DTypeLike = np.floor(train_transform[3] * 1000).astype(int) / 1000
hyoka_4: npt.DTypeLike = np.floor(train_transform[4] * 1000).astype(int) / 1000
hyoka_5: npt.DTypeLike = np.floor(train_transform[5] * 1000).astype(int) / 1000
result = (hyoka_0 + hyoka_1 + hyoka_2 + hyoka_3 + hyoka_4 + hyoka_5) / 6
# print(train_transform[0])
# print(train_transform[1])
# print(train_transform[2])
# print(train_transform[3])
# print(train_transform[4])
# print(train_transform[5])
# [0.5529412]
# [0.57254905]
# [0.53333336]
# [0.49019608]
# [0.50980395]
# [0.48235294]
print("Approximate: {:.2f}".format(np.float64(result)))
# Approximate: 0.52
# Approximate_value
# myself_20 ~ 30 years old about.

hyoka_test_0: npt.DTypeLike = np.floor(test_transform[0] * 1000).astype(int) / 1000
hyoka_test_1: npt.DTypeLike = np.floor(test_transform[1] * 1000).astype(int) / 1000
hyoka_test_2: npt.DTypeLike = np.floor(test_transform[2] * 1000).astype(int) / 1000
hyoka_test_3: npt.DTypeLike = np.floor(test_transform[3] * 1000).astype(int) / 1000
hyoka_test_4: npt.DTypeLike = np.floor(test_transform[4] * 1000).astype(int) / 1000
hyoka_test_5: npt.DTypeLike = np.floor(test_transform[5] * 1000).astype(int) / 1000
test_result = (hyoka_test_0 + hyoka_test_1 + hyoka_test_2 + hyoka_test_3 + hyoka_test_4 + hyoka_test_5) / 6
# print(test_transform[0])
# print(test_transform[1])
# print(test_transform[2])
# print(test_transform[3])
# print(test_transform[4])
# print(test_transform[5])
# [0.7137255]
# [0.69803923]
# [0.6745098]
# [0.78431374]
# [0.78431374]
# [0.7764706]
print("Approximate: {:.2f}".format(np.float64(test_result)))
# Approximate: 0.74
# Approximate_value
# myself_31 ~ 40 years old about.

# print(len(transform.shape))
# 2 / "Length" or "Number of elements"

# print(transform)
# [[0.7137255 ]
#  [0.69803923]
#  [0.6745098 ]
#  ...
#  [0.29411766]
#  [0.26666668]
#  [0.25882354]]

# nmf = NMF(
#     n_components='auto',
#     init=None,
#     solver='cd',
#     beta_loss='frobenius',
#     tol=0.0001,
#     max_iter=200,
#     random_state=None,
#     alpha_W=0.0,
#     alpha_H='same',
#     l1_ratio=0.0,
#     verbose=0,
#     shuffle=False,
# )
#
# nmf.fit(X_train)
# study = nmf.fit_transform(X_train)
# print(study)
# result = nmf.components_
# result_np = np.floor(result * 1000, dtype=np.float64) / 1000
# # DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
# print(np.float64(result_np))  # Ans, 16.186 / 18.708 / 20.909


# x_reconstructed_nmf = np.dot(nmf.transform(X_test), nmf.components_)

# nmf.py:1728: ConvergenceWarning:
# Maximum number of iterations 200 reached.
# Increase it to improve convergence.
# warnings.warn(

# Maximum number of iterations 200 | max_iter = 200, GitHub disccussion on no problem.

# print('NMF')
# print(x_reconstructed_nmf)  # [[350.]] / [[3.]]

# DecisionTreeClassifier / accuracy_score
# clf = DecisionTreeClassifier()
# clf.fit(X_train, Y_train)
# y_pred = clf.predict(X_test)
# acc = accuracy_score(Y_test, y_pred)
# print("Accuracy:", acc)  # Accuracy: 0.0

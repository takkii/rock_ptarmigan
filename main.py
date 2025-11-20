import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.decomposition import NMF

# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split

# from sklearn.tree import DecisionTreeClassifier

# train data
after = cv2.imread('./images/Sample/after/face.gif')
after_rgb = cv2.cvtColor(after, cv2.COLOR_RGB2BGR)
# after_dt = after.dtype
# after_sh = after.shape  # (262, 350, 3)
after_rgb_sh = after_rgb.shape  # (262, 350, 3)
# after_rgb_np = np.array(after_rgb_sh).reshape(-1, 1)
# after_np = np.array(after_sh).reshape(-1, 1)
# py
# [[262][350][3]]
# [[262][350][3]]

# https://www.tensorflow.org/guide/keras/sequential_model?hl=ja
model = keras.Sequential()
model.add(keras.Input(shape=after_rgb_sh))  # 250x250 RGB images
model.add(layers.Conv2D(32, 5, strides=2, activation="relu"))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(3))

# Can you guess what the current output shape is at this point? Probably not.
# Let's just print it:
model.summary()

model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(3))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(2))

# And now?
model.summary()

# Now that we have 4x4 feature maps, time to apply global max pooling.
model.add(layers.GlobalMaxPooling2D())

# Finally, we add a classification layer.
model.add(layers.Dense(10))

# Model: "sequential"
# ┌─────────────────────────────────┬────────────────────────┬───────────────┐
# │ Layer (type)                    │ Output Shape           │       Param # │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ conv2d (Conv2D)                 │ (None, 129, 173, 32)   │         2,432 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ conv2d_1 (Conv2D)               │ (None, 127, 171, 32)   │         9,248 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ max_pooling2d (MaxPooling2D)    │ (None, 42, 57, 32)     │             0 │
# └─────────────────────────────────┴────────────────────────┴───────────────┘
#  Total params: 11,680 (45.62 KB)
#  Trainable params: 11,680 (45.62 KB)
#  Non-trainable params: 0 (0.00 B)
# Model: "sequential"
# ┌─────────────────────────────────┬────────────────────────┬───────────────┐
# │ Layer (type)                    │ Output Shape           │       Param # │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ conv2d (Conv2D)                 │ (None, 129, 173, 32)   │         2,432 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ conv2d_1 (Conv2D)               │ (None, 127, 171, 32)   │         9,248 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ max_pooling2d (MaxPooling2D)    │ (None, 42, 57, 32)     │             0 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ conv2d_2 (Conv2D)               │ (None, 40, 55, 32)     │         9,248 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ conv2d_3 (Conv2D)               │ (None, 38, 53, 32)     │         9,248 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ max_pooling2d_1 (MaxPooling2D)  │ (None, 12, 17, 32)     │             0 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ conv2d_4 (Conv2D)               │ (None, 10, 15, 32)     │         9,248 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ conv2d_5 (Conv2D)               │ (None, 8, 13, 32)      │         9,248 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ max_pooling2d_2 (MaxPooling2D)  │ (None, 4, 6, 32)       │             0 │
# └─────────────────────────────────┴────────────────────────┴───────────────┘
#  Total params: 48,672 (190.12 KB)
#  Trainable params: 48,672 (190.12 KB)
#  Non-trainable params: 0 (0.00 B)

# test data
# before = cv2.imread('./images/Sample/before/face.gif')
# before_rgb = cv2.cvtColor(before, cv2.COLOR_RGB2BGR)
# before_dt = before.dtype
# before_sh = before.shape  # (262, 350, 3)
# before_rgb_sh = before_rgb.shape  # (262, 350, 3)
# before_rgb_np = np.array(before_rgb_sh).reshape(-1, 1)
# before_np = np.array(before_sh).reshape(-1, 1)
# py
# [[262][350][3]]
# [[262][350][3]]

# train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)
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

# pca = PCA(
#     n_components=None,
#     copy=True,
#     whiten=False,
#     svd_solver='auto',
#     tol=0.0,
#     iterated_power='auto',
#     n_oversamples=10,
#     power_iteration_normalizer='auto',
#     random_state=0,
# )

# pca.fit(X_train)
# transform = pca.inverse_transform(pca.transform(X_test))
# print(transform)  # [[350.0]]

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

# nmf.fit(X_train)
# study = nmf.fit_transform(X_train)
# print(study)
# result = nmf.components_
# result_np = np.floor(result * 1000, dtype=np.float64) / 1000
# DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
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

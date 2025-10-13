import cv2
import numpy as np

# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split

after = cv2.imread('./images/Sample/after/face.gif')
# after_rgb = cv2.cvtColor(after, cv2.COLOR_RGB2BGR)
# after_dt = after.dtype
after_sh = after.shape  # (262, 350, 3)
after_np = np.array(after_sh).reshape(-1, 1)
# py
# [[262][350][3]]
# [[262][350][3]]

before = cv2.imread('./images/Sample/before/face.gif')
# before_rgb = cv2.cvtColor(before, cv2.COLOR_RGB2BGR)
# before_dt = before.dtype
before_sh = before.shape  # (262, 350, 3)
before_np = np.array(before_sh).reshape(-1, 1)
# py
# [[262][350][3]]
# [[262][350][3]]

X_train, X_test, Y_train, Y_test = train_test_split(after_np, before_np, train_size=0.8)

# kmeans = KMeans(n_clusters=3, random_state=0)
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

nmf = NMF(
    n_components='auto',
    init=None,
    solver='cd',
    beta_loss='frobenius',
    tol=0.0001,
    max_iter=200,
    random_state=None,
    alpha_W=0.0,
    alpha_H='same',
    l1_ratio=0.0,
    verbose=0,
    shuffle=False,
)

nmf.fit(X_train)
x_reconstructed_nmf = np.dot(nmf.transform(X_test), nmf.components_)

# nmf.py:1728: ConvergenceWarning: Maximum number of iterations 200 reached. Increase it to improve convergence.
# warnings.warn(

# Maximum number of iterations 200 | max_iter = 200, GitHub disccussion on no problem.

print(x_reconstructed_nmf)  # [[350.]]

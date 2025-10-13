import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

after = cv2.imread('./images/Sample/after/face.gif')
after_rgb = cv2.cvtColor(after, cv2.COLOR_RGB2BGR)
after_dt = after.dtype
after_sh = after.shape  # (262, 350, 3)
after_np = np.array(after_sh).reshape(-1, 1)
# py
# [[262][350][3]]
# [[262][350][3]]

before = cv2.imread('./images/Sample/before/face.gif')
before_rgb = cv2.cvtColor(before, cv2.COLOR_RGB2BGR)
before_dt = before.dtype
before_sh = before.shape  # (262, 350, 3)
before_np = np.array(before_sh).reshape(-1, 1)
# py
# [[262][350][3]]
# [[262][350][3]]

X_train, X_test, Y_train, Y_test = train_test_split(after_np, before_np, train_size=0.8)

# kmeans = KMeans(n_clusters=3, random_state=0)
kmeans = KMeans(
    copy_x=True,
    init='k-means++',
    max_iter=300,
    n_clusters=2,
    n_init=10,
    random_state=None,
    tol=0.0001,
    verbose=0,
)

x_kmean = kmeans.fit(X_train)
x_reconstructed_means = kmeans.cluster_centers_[kmeans.predict(X_test)]

print(x_reconstructed_means)  # [[262.]]

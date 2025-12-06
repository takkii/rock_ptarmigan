import gc
import glob
import os
import threading
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

AFC = os.environ.get("if_after_cv2")
IMP = os.environ.get("if_images_path")
IMF = os.environ.get("images_file")
TRF = os.environ.get("train_folder")
NAT = os.environ.get("num_approximate_train")


# face class
class iFace(threading.Thread):

    # use thread
    def __init__(self):
        threading.Thread.__init__(self)

    # run method
    def run(self):
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

        num_result = np.float64(result)
        nat_str = str(NAT)

        print('Check, train data < ' + nat_str)

        if num_result < float(nat_str):
            print("train data is {:.2f}".format(np.float64(result)))
            pass
        else:
            print("train data is {}".format(np.float64(result)))
        # Check, train
        # data < 0.149
        # train data is 0.15016666666666667


# try ~ except ~ finally.
try:
    thread = iFace()
    thread.run()
# Custom Exception, raise throw.
except ValueError as ext:
    print(ext)
    raise RuntimeError from None

# Once Exec.
finally:
    # GC collection.
    gc.collect()

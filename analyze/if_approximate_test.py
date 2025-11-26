import gc
import glob
import os
import threading
import warnings
from os.path import dirname, join

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
VAF = os.environ.get("validation_folder")
NAE = os.environ.get("num_approximate_test")


# face class
class iFace(threading.Thread):

    # use thread
    def __init__(self):
        threading.Thread.__init__(self)

    # run method
    def run(self):
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

        num_result = np.float64(test_result)

        print('Check, train data < ' + NAE)

        if num_result < float(NAE):
            print("test data is {:.2f}".format(np.float64(test_result)))
            pass
        else:
            print("test data is {}".format(np.float64(test_result)))
        # Check, train
        # data < 0.10
        # test data is 0.10542857142857143


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

import gc
import threading

import rock_ptarmigan as rock


# face class
class Face(threading.Thread):

    # use thread
    def __init__(self):
        threading.Thread.__init__(self)

    # run method
    def run(self):
        # rock_ptarmigan version: x.x.x
        print("rock_ptarmigan version: " + rock.__version__)
        # Approximate value evaluation to all
        rock.compare_all()
        # Approximate value evaluation to train data.
        rock.compare_train()
        # Approximate valute evaluation to test data.
        rock.compare_validation()


# try ~ except ~ finally.
try:
    thread = Face()
    thread.run()
# Custom Exception, raise throw.
except ValueError as ext:
    print(ext)
    raise RuntimeError from None

# Once Exec.
finally:
    # GC collection.
    gc.collect()

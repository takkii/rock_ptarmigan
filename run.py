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
        # rock_ptarmigan version.
        print("rock_ptarmigan_version: " + rock.__version__)
        # Approximate value evaluation.
        rock.compare_after()


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

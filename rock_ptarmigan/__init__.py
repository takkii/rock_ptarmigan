from importlib.metadata import version

from .main import compare_all
from .main import compare_train
from .main import compare_validation

# function call
__all__ = ['compare_all', 'compare_train', 'compare_validation']

# version
__version__ = version(__package__)      # type: ignore

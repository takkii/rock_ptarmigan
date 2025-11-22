from importlib.metadata import version

from .main import compare_after

# function call
__all__ = ['compare_after']

# version
__version__ = version(__package__)

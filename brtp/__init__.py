"""BRTP: bounded residual transport for prompt learning.

Importing the package only resolves the bundled CLIP dependency path.
"""

from . import _bootstrap

del _bootstrap

__version__ = "1.01"

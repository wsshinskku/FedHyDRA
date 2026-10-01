"""Use the same utility module through either package name."""

import sys

from fedhydra import utils

sys.modules[__name__] = utils

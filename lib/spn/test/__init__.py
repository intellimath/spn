# {{LICENCE}}

from spn.test.test_laspn import *

import sys
# _PY36 = sys.version_info[:2] >= (3, 6)


def test_all():
    import unittest
    unittest.main(verbosity=3)

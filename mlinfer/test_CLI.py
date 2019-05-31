import unittest

from holo.__main__ import get_args

class TestCLI(unittest.TestCase):
    def test_get_args(self):
        args = get_args(['show', 'foo', '--metadata', 'bar'])
        isok = isinstance(args, dict)
        self.assertTrue(isok)


if __name__ == '__main__':
    unittest.main()
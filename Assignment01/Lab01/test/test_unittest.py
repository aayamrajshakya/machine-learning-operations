import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from calculator import *

class TestCalculator(unittest.TestCase):

    def test_fun1(self):
        self.assertEqual(fun1(2,3), 5)

    def test_fun2(self):
        self.assertEqual(fun2(2,3), -1)

    def test_fun3(self):
        self.assertEqual(fun3(2,3), 6)

    def test_fun4(self):
        self.assertEqual(fun4(2,3), 10)

if __name__ == '__main__':
    unittest.main()
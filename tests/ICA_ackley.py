import unittest
from ica import ICA
from test_functions import ackley


class MyTestCase(unittest.TestCase):
    def test_something(self):
        ica = ICA(ackley, 10, 3, 1)
        result = ica.eval()
        self.assertAlmostEqual(0.0, ackley.function(result), 3)


if __name__ == '__main__':
    unittest.main()

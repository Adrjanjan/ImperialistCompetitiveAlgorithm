import unittest
from ica import ICA
from test_functions import ackley


class MyTestCase(unittest.TestCase):
    def test_ackley(self):
        ica = ICA(ackley, 10, 3, 3, log=True)
        result = ica.eval()
        print(result)
        self.assertAlmostEqual(0.0, ackley.function(result).numpy()[0], 3)


if __name__ == '__main__':
    unittest.main()

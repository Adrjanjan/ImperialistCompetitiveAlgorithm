import unittest
from ica.ica import ICA
from algorithm_evaluation.test_functions import *
import tensorflow as tf

tf.config.run_functions_eagerly(True)


class MyTestCase(unittest.TestCase):

    # ---------- 500 -----------------

    def test_a_ackley_500(self):
        ackley = CostFunction(CostFunction.ackley_func, 10.0, -10.0, 1000)
        ica = ICA(ackley, 100, 3, 500, log=True)
        result = ica.eval()
        print(result)
        self.assertAlmostEqual(0.0, ackley.function(result).numpy(), 3)

    def test_a_square_500(self):
        square = CostFunction(lambda x: tf.reduce_sum(tf.square(x)), 100.0, -100.0, 1000)
        ica = ICA(square, 100, 3, 500, log=True)
        result = ica.eval()
        print(result)
        self.assertAlmostEqual(0.0, square.function(result).numpy(), 3)

    def test_a_schwefel_500(self):
        schwefel = CostFunction(CostFunction.schwefel_func, 100.0, -100.0, 1000)
        ica = ICA(schwefel, 100, 3, 50, log=True)
        result = ica.eval()
        print(result)
        self.assertAlmostEqual(0.0, schwefel.function(result).numpy()[0], 3)

    def test_a_rosenbrock_500(self):
        rosenbrock = CostFunction(CostFunction.rosenbrock_func, 100.0, -100.0, 1000)
        ica = ICA(rosenbrock, 100, 3, 500, log=True)
        result = ica.eval()
        print(result)
        self.assertAlmostEqual(0.0, rosenbrock.function(result).numpy(), 3)

    def test_a_rastrigin_500(self):
        rastrigin = CostFunction(CostFunction.rastrigin_func, 100.0, -100.0, 1000)
        ica = ICA(rastrigin, 100, 3, 500, log=True)
        result = ica.eval()
        print(result)
        self.assertAlmostEqual(0.0, rastrigin.function(result).numpy(), 3)

    # ---------- 1000 -----------------

    def test_b_ackley_1000(self):
        ackley = CostFunction(CostFunction.ackley_func, 10.0, -10.0, 1000)
        ica = ICA(ackley, 100, 3, 1000, log=True)
        result = ica.eval()
        print(result)
        self.assertAlmostEqual(0.0, ackley.function(result).numpy(), 3)

    def test_b_square_1000(self):
        square = CostFunction(lambda x: tf.reduce_sum(tf.square(x)), 100.0, -100.0, 1000)
        ica = ICA(square, 100, 3, 1000, log=True)
        result = ica.eval()
        print(result)
        self.assertAlmostEqual(0.0, square.function(result).numpy(), 3)

    def test_b_schwefel_1000(self):
        schwefel = CostFunction(CostFunction.schwefel_func, 100.0, -100.0, 1000)
        ica = ICA(schwefel, 100, 3, 1000, log=True)
        result = ica.eval()
        print(result)
        self.assertAlmostEqual(0.0, schwefel.function(result).numpy()[0], 3)

    def test_b_rosenbrock_1000(self):
        rosenbrock = CostFunction(CostFunction.rosenbrock_func, 100.0, -100.0, 1000)
        ica = ICA(rosenbrock, 100, 3, 1000, log=True)
        result = ica.eval()
        print(result)
        self.assertAlmostEqual(0.0, rosenbrock.function(result).numpy(), 3)

    def test_b_rastrigin_1000(self):
        rastrigin = CostFunction(CostFunction.rastrigin_func, 100.0, -100.0, 1000)
        ica = ICA(rastrigin, 100, 3, 1000, log=True)
        result = ica.eval()
        print(result)
        self.assertAlmostEqual(0.0, rastrigin.function(result).numpy(), 3)

    # ---------- 10000 -----------------

    def test_c_ackley_10000(self):
        ackley = CostFunction(CostFunction.ackley_func, 10.0, -10.0, 1000)
        ica = ICA(ackley, 100, 3, 10000, log=True)
        result = ica.eval()
        print(result)
        self.assertAlmostEqual(0.0, ackley.function(result).numpy(), 3)

    def test_c_square_10000(self):
        square = CostFunction(lambda x: tf.reduce_sum(tf.square(x)), 100.0, -100.0, 1000)
        ica = ICA(square, 100, 3, 10000, log=True)
        result = ica.eval()
        print(result)
        self.assertAlmostEqual(0.0, square.function(result).numpy(), 3)

    def test_c_schwefel_10000(self):
        schwefel = CostFunction(CostFunction.schwefel_func, 100.0, -100.0, 1000)
        ica = ICA(schwefel, 100, 3, 10000, log=True)
        result = ica.eval()
        print(result)
        self.assertAlmostEqual(0.0, schwefel.function(result).numpy()[0], 3)

    def test_c_rosenbrock_10000(self):
        rosenbrock = CostFunction(CostFunction.rosenbrock_func, 100.0, -100.0, 1000)
        ica = ICA(rosenbrock, 100, 3, 10000, log=True)
        result = ica.eval()
        print(result)
        self.assertAlmostEqual(0.0, rosenbrock.function(result).numpy(), 3)

    def test_c_rastrigin_10000(self):
        rastrigin = CostFunction(CostFunction.rastrigin_func, 100.0, -100.0, 1000)
        ica = ICA(rastrigin, 100, 3, 10000, log=True)
        result = ica.eval()
        print(result)
        self.assertAlmostEqual(0.0, rastrigin.function(result).numpy(), 3)


if __name__ == '__main__':
    unittest.main()

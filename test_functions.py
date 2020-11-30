import tensorflow as tf
import constants


class CostFunction:
    def __init__(self, function, upper, lower, dimension):
        self.function = function
        self.upper = upper
        self.lower = lower
        self.dimension = dimension


# @tf.function
def rastrigin_func(vector: tf.Tensor):
    ten = tf.constant(10.0, tf.float64)
    return tf.foldl((lambda accumulator, x: accumulator + x * x + tf.math.cos(constants.two_pi * x)), vector,
                    ten * tf.dtypes.cast(tf.shape(vector), tf.float64))


# @tf.function
def ackley_func(vector: tf.Tensor):
    a = tf.constant(20.0, tf.float64)
    b = tf.constant(0.2, tf.float64)
    c = tf.constant(constants.two_pi, tf.float64)
    n = tf.dtypes.cast(tf.shape(vector), tf.float64)
    sum1 = tf.reduce_sum(tf.square(vector))
    sum2 = tf.reduce_sum(tf.math.cos(tf.multiply(c, vector)))
    return -a * tf.math.exp(-b * tf.math.sqrt(tf.math.divide(sum1, n))) \
           - tf.math.exp(tf.math.divide(sum2, n)) + a + constants.e


# @tf.function
def schwefel_func(vector: tf.Tensor):
    a = tf.constant(418.9829, tf.float64)
    n = tf.dtypes.cast(tf.shape(vector), tf.float64)
    sum1 = tf.foldl((lambda accumulator, x: accumulator + x * tf.math.sin(tf.math.sqrt(tf.math.abs(x)))), vector,
                    constants.zero)
    return a * n - sum1


# @tf.function
def rosenbrock_func(vector: tf.Tensor):
    a = tf.constant(100.0, tf.float64)
    index_sum = (tf.constant(0), constants.zero)

    # @tf.function
    def condition(index, _):
        return tf.less(index, tf.subtract(tf.shape(vector)[0], 1))

    # @tf.function
    def body(index, summation):
        result = tf.add(summation,
                        a * tf.math.squared_difference(
                            tf.gather(vector, tf.add(index, 1)),
                            tf.math.square(tf.gather(vector, index))
                        ) + tf.math.squared_difference(tf.gather(vector, index), constants.one))
        return tf.add(index, 1), result

    return tf.while_loop(condition, body, index_sum)[1]


rastrigin = CostFunction(rastrigin_func, 100.0, -100.0, 10)
ackley = CostFunction(ackley_func, 10.0, -10.0, 2)
schwefel = CostFunction(schwefel_func, 100.0, -100.0, 10)
rosenbrock = CostFunction(rosenbrock_func, 100.0, -100.0, 10)
square = CostFunction(lambda x: tf.reduce_sum(tf.square(x)), 100.0, -100.0, 2)

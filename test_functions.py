import tensorflow as tf
import constants


class CostFunction:
    def __init__(self, function, upper, lower, dimension):
        self.function = function
        self.upper = upper
        self.lower = lower
        self.dimension = dimension


@tf.function
def rastrigin_func(vector: tf.Tensor):
    init = tf.constant(10.0, tf.float64) * tf.dtypes.cast(tf.shape(vector), tf.float64)
    return init + tf.reduce_sum(tf.square(vector) + tf.math.cos(constants.two_pi * vector))


@tf.function
def ackley_func(vector: tf.Tensor):
    a = tf.constant(20.0, tf.float64)
    b = tf.constant(0.2, tf.float64)
    c = tf.constant(constants.two_pi, tf.float64)
    n = tf.dtypes.cast(tf.shape(vector), tf.float64)
    sum1 = tf.reduce_sum(tf.square(vector))
    sum2 = tf.reduce_sum(tf.math.cos(tf.multiply(c, vector)))
    return -a * tf.exp(-b * tf.sqrt(tf.truediv(sum1, n))) - tf.exp(tf.divide(sum2, n)) + a + constants.e


@tf.function
def schwefel_func(vector: tf.Tensor):
    a = tf.constant(418.9829, tf.float64)
    n = tf.dtypes.cast(tf.shape(vector), tf.float64)
    sum1 = tf.reduce_sum(vector * tf.sin(tf.sqrt(tf.abs(vector))))
    return a * n - sum1


@tf.function
def rosenbrock_func(vector: tf.Tensor):
    a = tf.constant(100.0, tf.float64)
    sum1 = a * tf.square(vector[1:] - tf.square(vector[:-1]))
    sum2 = tf.square(vector[:-1] - constants.one)
    return tf.reduce_sum(sum1 + sum2)

import tensorflow as tf
import constants


class CostFunction:
    def __init__(self, function, upper, lower, dimension, o_vector=None, p_vector=None, r_25=None, r_50=None,
                 r_100=None, s=None, w=None):
        self.function = function
        self.upper = upper
        self.lower = lower
        self.dimension = dimension
        self.powers = tf.range(0, self.dimension, dtype=tf.float64) / (self.dimension - 1)

        if o_vector is not None:
            self.o_vector = self.read_vector(o_vector, tf.float64)
        if p_vector is not None:
            self.p_vector = self.read_matrix(p_vector, tf.int32)
        if r_25 is not None:
            self.r_25 = self.read_matrix(r_25, tf.float64)
        if r_50 is not None:
            self.r_50 = self.read_matrix(r_50, tf.float64)
        if r_100 is not None:
            self.r_100 = self.read_matrix(r_100, tf.float64)
        if s is not None:
            self.s = self.read_vector(s, tf.int32)
        if w is not None:
            self.w = self.read_vector(w, tf.float64)

    @staticmethod
    def read_vector(filename, type):
        with open(filename, 'r') as file:
            lines = [float(f) for f in file.readlines()]
            result = tf.constant(lines, type)
            return result

    @staticmethod
    def read_matrix(filename, type):
        with open(filename, 'r') as file:
            matrix = [[float(num) for num in line.split(',')] for line in file]
            return tf.constant(matrix, type)

    @tf.function
    def transform_osz(self, x: tf.Tensor):
        return tf.sign(x) * tf.exp(self.hat(x) + 0.049 *
                                   (tf.sin(self.c1(x) * self.hat(x)) +
                                    tf.sin(self.c2(x) * self.hat(x)))
                                   )

    @tf.function
    def hat(self, x: tf.Tensor):
        condition = tf.equal(constants.zero, x)
        return tf.where(condition, tf.zeros_like(x), tf.math.log(tf.math.abs(x)))

    @tf.function
    def c1(self, x: tf.Tensor):
        condition = tf.greater(x, constants.zero)
        return tf.where(condition, tf.ones_like(x) * 10, tf.ones_like(x) * 5.5)

    @tf.function
    def c2(self, x: tf.Tensor):
        condition = tf.greater(x, constants.zero)
        return tf.where(condition, tf.ones_like(x) * 7.9, tf.ones_like(x) * 3.1)

    @tf.function
    def Lambda(self, x, alpha):
        return x * tf.pow(alpha, 0.5 * self.powers)

    @tf.function
    def transform_asy(self, x: tf.Tensor, beta):
        condition = tf.greater(x, constants.zero)
        asy = tf.pow(x, 1 + beta * self.powers * tf.sqrt(x))
        return tf.where(condition, asy, x)

    @tf.function
    def rastrigin_func(self, vector: tf.Tensor):
        init = tf.constant(10.0, tf.float64) * tf.cast(tf.shape(vector), tf.float64)
        return init + tf.reduce_sum(tf.square(vector) + tf.math.cos(constants.two_pi * vector))

    @tf.function
    def ackley_func(self, vector: tf.Tensor):
        a = tf.constant(20.0, tf.float64)
        b = tf.constant(0.2, tf.float64)
        c = tf.constant(constants.two_pi, tf.float64)
        sum1 = tf.reduce_sum(tf.square(vector))
        sum2 = tf.reduce_sum(tf.math.cos(tf.multiply(c, vector)))
        return -a * tf.exp(-b * tf.sqrt(tf.truediv(sum1, self.dimension))) - \
               tf.exp(tf.divide(sum2, self.dimension)) + a + constants.e

    @tf.function
    def schwefel_func(self, vector: tf.Tensor):
        a = tf.constant(418.9829, tf.float64)
        sum1 = tf.reduce_sum(vector * tf.sin(tf.sqrt(tf.abs(vector))))
        return a * self.dimension - sum1

    @tf.function
    def rosenbrock_func(self, vector: tf.Tensor):
        a = tf.constant(100.0, tf.float64)
        sum1 = a * tf.square(vector[1:] - tf.square(vector[:-1]))
        sum2 = tf.square(vector[:-1] - constants.one)
        return tf.reduce_sum(sum1 + sum2)

    @tf.function
    def elliptic_func(self, vector: tf.Tensor):
        base = tf.constant(1.0e6, tf.float64)
        return tf.reduce_sum(tf.math.pow(base, self.powers) * tf.square(self.transform_osz(vector)))

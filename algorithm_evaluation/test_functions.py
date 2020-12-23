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
        self.base_powered = tf.math.pow(10e6, self.powers)
        self.s_size = None
        self.overlap = None

        if o_vector is not None:
            self.o_vector = self.read_vector(o_vector, tf.float64)
        if p_vector is not None:
            self.p_vector = tf.squeeze(tf.subtract(self.read_matrix(p_vector, tf.int32), 1))
        if r_25 is not None:
            self.r_25 = self.read_matrix(r_25, tf.float64)
        if r_50 is not None:
            self.r_50 = self.read_matrix(r_50, tf.float64)
        if r_100 is not None:
            self.r_100 = self.read_matrix(r_100, tf.float64)
        if s is not None:
            self.s = self.read_vector(s, tf.int32)
            self.rotation_matrix = self.create_rotation_matrix()
        if w is not None:
            self.w = self.read_vector(w, tf.float64)

    @staticmethod
    def read_vector(filename, elements_type):
        with open(filename, 'r') as file:
            lines = [float(f) for f in file.readlines()]
            result = tf.cast(tf.constant(lines), elements_type)
            return result

    @staticmethod
    def read_matrix(filename, elements_type):
        with open(filename, 'r') as file:
            matrix = [[float(num) for num in line.split(',')] for line in file]
            return tf.cast(tf.constant(matrix), elements_type)

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
    def transform_asy(self, x: tf.Tensor, beta):
        condition = tf.greater(x, constants.zero)
        asy = tf.pow(x, 1 + beta * self.powers * tf.sqrt(x))
        return tf.where(condition, asy, x)

    @tf.function
    def rastrigin_func(self, matrix: tf.Tensor):
        constant = tf.constant(10.0, tf.float64)
        init = constant * self.dimension
        return init + tf.reduce_sum(tf.square(matrix) - constant * tf.math.cos(constants.two_pi * matrix), axis=1)

    @tf.function
    def ackley_func(self, matrix: tf.Tensor):
        a = tf.constant(20.0, tf.float64)
        b = tf.constant(0.2, tf.float64)
        sum1 = tf.reduce_sum(tf.square(matrix), axis=1)
        sum2 = tf.reduce_sum(tf.math.cos(tf.multiply(constants.two_pi, matrix)), axis=1)
        return -a * tf.exp(-b * tf.sqrt(tf.truediv(sum1, self.dimension))) - \
               tf.exp(tf.truediv(sum2, self.dimension)) + a + constants.e

    @tf.function
    def schwefel_func(self, matrix: tf.Tensor):
        return tf.reduce_sum(tf.square(tf.cumsum(matrix, axis=1)), axis=1)

    @tf.function
    def rosenbrock_func(self, matrix):
        a = tf.constant(100.0, tf.float64)
        x_shift = matrix[:, 1:]
        x_cut = matrix[:, :-1]
        return tf.reduce_sum(a * tf.square(tf.square(x_cut) - x_shift) + tf.square(x_cut - 1), axis=1)

    @tf.function
    def elliptic_func(self, matrix: tf.Tensor, start=None, end=None):
        return tf.reduce_sum(self.base_powered[start:end, ] * tf.square(self.transform_osz(matrix)), axis=1)

    @tf.function
    def rotate_vector(self, vector, start, size):
        rotated = tf.gather(vector, self.p_vector[start:start + size])
        multiplied = tf.case([
            (tf.equal(size, 25), lambda: tf.matmul(self.r_25, rotated)),
            (tf.equal(size, 50), lambda: tf.matmul(self.r_50, rotated)),
            (tf.equal(size, 100), lambda: tf.matmul(self.r_100, rotated)),
        ])
        return tf.squeeze(multiplied)

    @tf.function
    def rotate_vector_conform(self, vector, start, size, index):
        shift = index * self.overlap
        rotated = tf.gather(vector, self.p_vector[start - shift:start + size - shift], axis=1)
        multiplied = tf.case([
            (tf.equal(size, 25), lambda: tf.linalg.matvec(self.r_25, rotated)),
            (tf.equal(size, 50), lambda: tf.linalg.matvec(self.r_50, rotated)),
            (tf.equal(size, 100), lambda: tf.linalg.matvec(self.r_100, rotated)),
        ])
        return tf.squeeze(multiplied)

    @tf.function
    def rotate_vector_conflict(self, vector, start, size, index):
        shift = index * self.overlap
        rotated = tf.subtract(tf.gather(vector, self.p_vector[start - shift:start + size - shift], axis=1),
                              self.o_vector[start - shift:start + size - shift])

        multiplied = tf.case([
            (tf.equal(size, 25), lambda: tf.linalg.matvec(self.r_25, rotated)),
            (tf.equal(size, 50), lambda: tf.linalg.matvec(self.r_50, rotated)),
            (tf.equal(size, 100), lambda: tf.linalg.matvec(self.r_100, rotated)),
        ])
        return tf.squeeze(multiplied)

    @tf.function
    def create_rotation_matrix(self):
        operators = []
        for x in self.s:
            if x == 25:
                operators.append(tf.linalg.LinearOperatorFullMatrix(self.r_25))
            elif x == 50:
                operators.append(tf.linalg.LinearOperatorFullMatrix(self.r_50))
            else:
                operators.append(tf.linalg.LinearOperatorFullMatrix(self.r_100))
        s_sum = tf.reduce_sum(self.s)
        if s_sum < self.dimension:
            difference = self.dimension - s_sum
            operators.append(tf.linalg.LinearOperatorFullMatrix(tf.eye(difference, difference, dtype=tf.float64)))

        rotation_matrix = tf.linalg.LinearOperatorBlockDiag(operators)
        return rotation_matrix

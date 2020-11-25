import timeit
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import dtypes
from test_functions import CostFunction
import constants


class ICA:

    def __init__(self, cost_function: CostFunction, num_of_countries=100, num_of_imperialist=4, max_iterations=10000,
                 deviation_assimilation=np.pi / 4, direct_assimilation=0.5, avg_colonies_power=0.1, revolution_rate=0.2,
                 log=False, seed=42):
        self.evaluation_time = -1
        self.cost_function = cost_function
        self.num_of_countries = num_of_countries
        self.num_of_imperialist = num_of_imperialist
        self.max_iterations = max_iterations
        self.deviation_assimilation = deviation_assimilation
        self.direct_assimilation = direct_assimilation
        self.avg_colonies_power = avg_colonies_power
        self.revolution_rate = revolution_rate
        self.is_loggable = log
        self.timeit = 0
        self.seed = seed

    # @tf.function
    def eval(self):
        # @tf.function
        def condition(index, *unused):
            # TODO think about stop condition where there is only one empire
            return tf.greater(index, constants.int_zero)

        # @tf.function
        def body(index: tf.Tensor, empires: tf.Variable, colonies: tf.Variable, empires_indexes: tf.Variable):
            empires, colonies, empires_indexes = self.assimilation(empires, colonies, empires_indexes)
            empires, colonies, empires_indexes = self.revolution(empires, colonies, empires_indexes)
            empires, colonies, empires_indexes = self.swap_strongest(empires, colonies, empires_indexes)
            empires, colonies, empires_indexes = self.competition(empires, colonies, empires_indexes)
            empires, colonies, empires_indexes = self.elimination(empires, colonies, empires_indexes)
            empires, colonies, empires_indexes = self.merging_of_similar(empires, colonies, empires_indexes)
            tf.print(countries)
            tf.print(empires)
            tf.print(empires_indexes)
            return tf.subtract(index, 1), empires, colonies, empires_indexes

        # country is a row of matrix where each value is parameter of some cost function
        # colonies is matrix, where each row represents colony country, no duplicates
        # empires is matrix, where each row represents empire country, duplicates
        # corresponding row in matrices colonies and empires represents connection
        # between colony belonging to an empire

        self.start_benchmark()
        countries = self.initialize_countries()
        empires, colonies, empires_index = self.create_empires(countries)
        index_and_results = (tf.constant(self.max_iterations), empires, colonies, empires_index)
        _, empires, _, _ = tf.while_loop(condition, body, index_and_results)
        self.finish_benchmark()
        solution = empires[0]
        return solution

    # @tf.function
    def initialize_countries(self):
        return tf.Variable(tf.random.uniform(shape=(self.num_of_countries, self.cost_function.dimension),
                                             minval=self.cost_function.lower,
                                             maxval=self.cost_function.upper,
                                             dtype=dtypes.float64,
                                             seed=self.seed
                                             ),
                           name="Countries"
                           )

    # @tf.function
    def create_empires(self, countries: tf.Variable):
        # @tf.function
        def calculate_number_of_colonies(top_powers: tf.Tensor):
            max_cost = tf.reduce_max(top_powers[:])
            normalised_cost = tf.where(max_cost > 0, tf.subtract(tf.multiply(1.3, max_cost), top_powers),
                                       tf.subtract(tf.multiply(0.7, max_cost), top_powers))
            num_of_colonies_per_empire = tf.round(
                tf.multiply(tf.truediv(normalised_cost, tf.reduce_sum(normalised_cost)),
                            self.num_of_countries - self.num_of_imperialist))

            def add_one_to_top_empire(n):
                return tf.concat([tf.reshape(tf.add(n[0], 1), 1), n[1:]], 0)

            return tf.cast(add_one_to_top_empire(num_of_colonies_per_empire), tf.int32)

        def condition(index, *unused):
            return tf.greater_equal(index, constants.int_zero)

            # @tf.function

        def body(index: tf.Tensor, number_of_colonies, top_indexes, empires, empires_indexes):
            current_num_of_colonies = tf.gather(number_of_colonies, index)
            current_empire_index = tf.gather(top_indexes, index)
            new_empire_matrix = tf.reshape(
                tf.tile(tf.gather(countries, current_empire_index), [current_num_of_colonies]),
                [current_num_of_colonies, self.cost_function.dimension])
            new_empire_indexes = tf.reshape(tf.tile([index], [current_num_of_colonies]),
                                            [current_num_of_colonies, 1])

            concatenated_empires = tf.concat([empires, new_empire_matrix], 0)
            concatenated_empires_indexes = tf.concat([empires_indexes, new_empire_indexes], 0)
            new_index = tf.subtract(index, 1)
            return new_index, number_of_colonies, top_indexes, concatenated_empires, concatenated_empires_indexes

        power = self.evaluate_countries_power(countries)
        # (-1)*power is important to get lowest power first
        top_powers, top_indexes = tf.nn.top_k(-power, self.num_of_imperialist)
        index_empires_matrix = (tf.constant(self.num_of_imperialist - 1),
                                calculate_number_of_colonies(-top_powers),
                                tf.constant(top_indexes),
                                tf.zeros((0, self.cost_function.dimension), tf.float64),
                                tf.zeros((0, 1), tf.int32),
                                )

        _, _, _, empires, empires_indexes = tf.while_loop(condition, body, index_empires_matrix)
        colonies = self.delete_rows(countries, top_indexes)
        return empires, colonies, tf.squeeze(empires_indexes)

    # @tf.function
    def assimilation(self, empires: tf.Variable, colonies: tf.Variable, empires_index: tf.Variable):
        # TODO zrewidować poprawność działania i zweryfikować ze wzorem z prac
        def calculate_distribution():
            return tf.random.uniform(shape=empires.shape,
                                     minval=constants.zero,
                                     maxval=tf.multiply(empires, self.direct_assimilation),
                                     dtype=dtypes.float64,
                                     seed=self.seed
                                     )

        uniform = calculate_distribution()
        new_colonies = tf.clip_by_value(
            tf.add(colonies, tf.multiply(uniform, self.normalize_vectors(tf.subtract(empires, colonies)))),
            self.cost_function.lower,
            self.cost_function.upper
        )
        return empires, new_colonies, empires_index

    # @tf.function
    def revolution(self, empires, colonies, empires_index: tf.Variable):
        def calculate_revolving():
            return tf.random.uniform(shape=empires_index.shape,
                                     minval=constants.zero,
                                     maxval=constants.one,
                                     dtype=dtypes.float64
                                     )

        def possible_revolution_positions():
            return tf.random.uniform(shape=colonies.shape,
                                     minval=self.cost_function.lower,
                                     maxval=self.cost_function.upper,
                                     dtype=tf.float64
                                     )

        to_revolve = calculate_revolving()
        new_colonies = tf.where(self.broadcast_boolean_mask(tf.less(to_revolve, self.revolution_rate), colonies.shape),
                                possible_revolution_positions(),
                                colonies)
        return empires, new_colonies, empires_index

    # @tf.function
    def swap_strongest(self, empires, colonies, empires_index: tf.Variable):
        def condition(index, *unused):
            return tf.less(index, self.num_of_imperialist)

        def body(current_empire_index, empires, colonies, empires_index):
            def swap():
                current_empire = self.get_first_by_index(current_empire_index, empires, empires_index)
                current_colonies = self.get_all_by_index(current_empire_index, colonies, empires_index)
                best_colony = tf.boolean_mask(current_colonies, tf.equal(best_colony_power, current_colonies_power))

                new_e = self.broadcastable_where(tf.equal(empires_index, current_empire_index), best_colony, empires)
                new_c = self.broadcastable_where(tf.map_fn(tf.math.reduce_all, tf.equal(best_colony, colonies)),
                                                 current_empire, colonies)
                return new_e, new_c

            def do_nothing():
                return colonies, empires

            current_empire_power = self.get_first_by_index(current_empire_index, empires_power, empires_index)
            current_colonies_power = self.get_all_by_index(current_empire_index, colonies_power, empires_index)
            best_colony_power = tf.reduce_min(current_colonies_power)
            c, e = tf.cond(current_empire_power > best_colony_power, swap, do_nothing)
            return tf.add(1, current_empire_index), c, e, empires_index

        colonies_power = self.evaluate_countries_power(colonies)
        empires_power = self.evaluate_countries_power(empires)
        _, new_colonies, new_empires, _ = tf.while_loop(condition, body,
                                                        (constants.int_zero, empires, colonies, empires_index))
        return new_empires, new_colonies, empires_index

    # @tf.function
    def competition(self, empires, colonies, empires_index: tf.Variable):
        # TODO
        return empires, colonies, empires_index

    # @tf.function
    def elimination(self, empires, colonies, empires_index: tf.Variable):
        # TODO
        return empires, colonies, empires_index

    # @tf.function
    def merging_of_similar(self, empires, colonies, empires_index: tf.Variable):
        # TODO
        return empires, colonies, empires_index

    def start_benchmark(self):
        self.timeit = timeit.default_timer()

    def finish_benchmark(self):
        if self.is_loggable:
            self.evaluation_time = timeit.default_timer() - self.timeit
            print("Evaluation time", self.evaluation_time)

    @staticmethod
    def get_all_by_index(index, tensor, indexes):
        return tf.boolean_mask(tf.squeeze(tensor), tf.equal(index, indexes))

    @staticmethod
    def get_first_by_index(index, tensor, indexes):
        return tf.squeeze(tf.gather(tf.boolean_mask(tf.squeeze(tensor), tf.equal(index, indexes)), 0))

    def delete_rows(self, tensor, index):
        return tf.boolean_mask(tensor,
                               tf.subtract(1, tf.reduce_sum(tf.one_hot(index, self.num_of_countries), 0), tf.bool))

    def evaluate_countries_power(self, countries: tf.Variable):
        return tf.squeeze(tf.transpose(tf.map_fn(self.cost_function.function, countries)))

    def broadcastable_where(self, bool_mask, vector, tensor):
        return tf.where(self.broadcast_boolean_mask(bool_mask, tensor.shape),
                        tf.multiply(tf.ones_like(tensor), vector),
                        tensor)

    def broadcast_boolean_mask(self, bool_vector, shape):
        return tf.reshape(tf.repeat(bool_vector, self.cost_function.dimension), shape)

    def normalize_vectors(self, vector):
        norm = tf.sqrt(tf.reduce_sum(tf.square(vector), keepdims=True))
        return vector / norm

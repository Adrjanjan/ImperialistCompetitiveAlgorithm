import timeit
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import dtypes
from test_functions import CostFunction
import constants


class ICA:

    def __init__(self, cost_function: CostFunction, num_of_countries=100, num_of_imperialist=4, max_iterations=10000,
                 deviation_assimilation=np.pi / 4, direct_assimilation=0.5, avg_colonies_power=0.1, revolution_rate=0.2,
                 close_empires_rating=0.1, log=False, seed=42):
        self.evaluation_time = -1
        self.cost_function = cost_function
        self.dimension = cost_function.dimension
        self.lower = cost_function.lower
        self.upper = cost_function.upper
        self.num_of_countries = num_of_countries
        self.num_of_imperialist = num_of_imperialist
        self.num_of_colonies = num_of_countries - num_of_imperialist
        self.max_iterations = max_iterations
        self.deviation_assimilation = deviation_assimilation
        self.direct_assimilation = direct_assimilation
        self.avg_colonies_power = avg_colonies_power
        self.revolution_rate = revolution_rate
        self.close_empires_rating = close_empires_rating
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
        def body(index, empires, colonies, empires_indexes):
            empires, colonies, empires_indexes = self.assimilation(empires, colonies, empires_indexes)
            empires, colonies, empires_indexes = self.revolution(empires, colonies, empires_indexes)
            empires, colonies, empires_indexes = self.swap_strongest(empires, colonies, empires_indexes)
            empires, colonies, empires_indexes = self.competition(empires, colonies, empires_indexes)
            empires, colonies, empires_indexes = self.merging_of_similar(empires, colonies, empires_indexes)
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
        print(empires.numpy())
        print(colonies.numpy())
        solution = empires[0]
        return solution

    # @tf.function
    def initialize_countries(self):
        return tf.Variable(
            self.calculate_distribution((self.num_of_countries, self.dimension), self.lower, self.upper, self.seed)
        )

    # @tf.function
    def create_empires(self, countries):
        # @tf.function
        def calculate_number_of_colonies(top_powers):
            max_cost = tf.reduce_max(top_powers[:])
            normalised_cost = tf.where(max_cost > 0, tf.subtract(tf.multiply(1.3, max_cost), top_powers),
                                       tf.subtract(tf.multiply(0.7, max_cost), top_powers))
            num_of_colonies_per_empire = tf.round(
                tf.multiply(tf.truediv(normalised_cost, tf.reduce_sum(normalised_cost)), self.num_of_colonies))

            def add_one_to_top_empire(n):
                return tf.where(tf.less(tf.reduce_sum(num_of_colonies_per_empire), self.num_of_colonies),
                                tf.concat([tf.reshape(tf.add(n[0], 1), 1), n[1:]], 0),
                                n)

            return tf.cast(add_one_to_top_empire(num_of_colonies_per_empire), tf.int32)

        def condition(index, *unused):
            return tf.greater_equal(index, constants.int_zero)

        # @tf.function
        def body(index, number_of_colonies, top_indexes, empires, empires_indexes):
            current_num_of_colonies = tf.gather(number_of_colonies, index)
            current_empire_index = tf.gather(top_indexes, index)
            new_empire_matrix = tf.reshape(
                tf.tile(tf.gather(countries, current_empire_index), [current_num_of_colonies]),
                [current_num_of_colonies, self.dimension])
            new_empire_indexes = tf.reshape(tf.tile([index], [current_num_of_colonies]),
                                            [current_num_of_colonies, 1])

            concatenated_empires = tf.concat([empires, new_empire_matrix], 0)
            concatenated_empires_indexes = tf.concat([empires_indexes, new_empire_indexes], 0)
            new_index = tf.subtract(index, 1)
            return new_index, number_of_colonies, top_indexes, concatenated_empires, concatenated_empires_indexes

        power = self.evaluate_countries_power(countries)
        # (-1)*power is important to get lowest power first - lower cost means bigger power
        top_powers, top_indexes = tf.nn.top_k(-power, self.num_of_imperialist)
        top_powers = tf.negative(top_powers)
        index_empires_matrix = (tf.constant(self.num_of_imperialist - 1),
                                calculate_number_of_colonies(top_powers),
                                tf.constant(top_indexes),
                                tf.zeros((0, self.dimension), tf.float64),
                                tf.zeros((0, 1), tf.int32),
                                )

        _, _, _, empires, empires_indexes = tf.while_loop(condition, body, index_empires_matrix)
        colonies = self.delete_rows(countries, top_indexes)
        return empires, colonies, tf.squeeze(empires_indexes)

    # @tf.function
    def assimilation(self, empires, colonies, empires_indexes):
        difference = tf.subtract(empires, colonies)
        uniform = self.calculate_distribution(empires.shape,
                                              constants.zero,
                                              tf.multiply(tf.reduce_sum(difference), self.direct_assimilation)
                                              )
        new_colonies = tf.clip_by_value(
            tf.add(colonies, tf.multiply(uniform, self.normalize_vectors(difference))),
            self.lower,
            self.upper
        )
        return empires, new_colonies, empires_indexes

    # @tf.function
    def revolution(self, empires, colonies, empires_indexes):
        to_revolve = self.calculate_distribution(empires_indexes.shape, constants.zero, constants.one)
        possible_new_positions = self.calculate_distribution(colonies.shape,
                                                             self.lower,
                                                             self.upper
                                                             )
        new_colonies = tf.where(self.broadcast_boolean_mask(tf.less(to_revolve, self.revolution_rate), colonies.shape),
                                possible_new_positions,
                                colonies
                                )
        return empires, new_colonies, empires_indexes

    # @tf.function
    def swap_strongest(self, empires, colonies, empires_indexes):
        def condition(index, *unused):
            return tf.less(index, self.num_of_imperialist)

        def body(current_empire_index, empires, colonies, empires_indexes):
            def swap():
                current_empire = self.get_first_by_index(current_empire_index, empires, empires_indexes)
                current_colonies_power_with_index = self.get_all_by_index(current_empire_index,
                                                                          colonies_power_with_index,
                                                                          empires_indexes)
                current_colonies_power = tf.squeeze(current_colonies_power_with_index[:, :-1])
                current_colonies_power = tf.reshape(current_colonies_power, [-1])
                current_colonies_index = tf.cast(current_colonies_power_with_index[:, -1], tf.int32)

                _, best_colony_index = tf.nn.top_k(-current_colonies_power)
                best_colony = tf.gather(colonies, tf.gather(current_colonies_index, best_colony_index))

                new_e = self.broadcastable_where(tf.equal(empires_indexes, current_empire_index), best_colony, empires)
                new_c = self.broadcastable_where(tf.equal(colonies_indexes, best_colony_index), current_empire,
                                                 colonies)
                return new_e, new_c

            def do_nothing():
                return colonies, empires

            current_empire_power = self.get_first_by_index(current_empire_index, empires_power, empires_indexes)
            current_colonies_power = self.get_all_by_index(current_empire_index, colonies_power, empires_indexes)
            best_colony_power = tf.reduce_min(current_colonies_power)
            swapped_empires, swapped_colonies = tf.cond(current_empire_power > best_colony_power, swap, do_nothing)
            return tf.add(1, current_empire_index), swapped_empires, swapped_colonies, empires_indexes

        colonies_power = self.evaluate_countries_power(colonies)
        colonies_indexes = tf.cast(tf.range(colonies_power.shape[0])[:, None], tf.int32)
        colonies_power_with_index = tf.concat(
            [tf.reshape(colonies_power, [self.num_of_colonies, 1]),
             tf.cast(colonies_indexes, tf.float64)],
            axis=1
        )
        empires_power = self.evaluate_countries_power(empires)
        _, new_empires, new_colonies, _ = tf.while_loop(condition, body,
                                                        (constants.int_zero, empires, colonies, empires_indexes))
        return new_empires, new_colonies, empires_indexes

    # @tf.function
    def competition(self, empires, colonies, empires_indexes):
        # calculate total power
        total_power = self.calculate_empire_total_power(empires, colonies, empires_indexes)

        # select top and worst empire
        top_empire, top_empire_number = self.select_top_empire(empires, empires_indexes, total_power)
        _, worst_empire_index = tf.math.top_k(total_power)
        worst_empire_number = tf.gather(empires_indexes, worst_empire_index)

        # select worst colony from worst empire
        worst_colony_index = self.select_worst_colony_index(worst_empire_number, colonies, empires_indexes)

        # set new empire as empire of worst colony
        new_empires_indexes = tf.squeeze(
            self.replace_row_in_matrix(tf.expand_dims(empires_indexes, 1), top_empire_number, worst_colony_index)
        )
        new_empires = self.replace_row_in_matrix(empires, top_empire, worst_colony_index)

        # reduce num_of_imperialists if it was worst empire last presence
        self.check_number_of_imperialist(new_empires_indexes, worst_empire_number)
        return new_empires, colonies, new_empires_indexes

    def select_worst_colony_index(self, empire_index, colonies, empires_indexes):
        colonies_with_index = tf.concat([colonies, tf.cast(tf.range(colonies.shape[0])[:, None], tf.float64)],
                                        axis=1)
        current_empire_colonies_with_index = self.get_all_by_index(empire_index, colonies_with_index, empires_indexes)
        current_empire_colonies = tf.reshape(current_empire_colonies_with_index[:, :-1],
                                             [current_empire_colonies_with_index.shape[0], self.dimension])
        colonies_power = tf.reshape(self.evaluate_countries_power(current_empire_colonies), [-1])
        _, current_worst_index = tf.math.top_k(colonies_power)
        current_empire_colonies_index = tf.cast(current_empire_colonies_with_index[:, -1], tf.int32)
        worst_colony_index = tf.gather(current_empire_colonies_index, current_worst_index)
        return worst_colony_index

    def select_top_empire(self, empires, empires_indexes, total_power):
        distribution = self.calculate_distribution(total_power.shape, constants.zero, total_power)
        _, top_empire_index = tf.nn.top_k(tf.subtract(total_power, distribution))
        top_empire = tf.gather(empires, top_empire_index)
        top_empire_number = tf.gather(empires_indexes, top_empire_index)
        return top_empire, top_empire_number

    def calculate_empire_total_power(self, empires, colonies, empires_indexes):
        def condition(index, *unused):
            return tf.greater_equal(index, constants.int_zero)

        def evaluate_empire_power(index, total_power):
            empire_number = tf.gather(empire_numbers, index)
            current_empire_colonies_power = tf.reduce_sum(
                self.get_all_by_index(empire_number, colonies_power, empires_indexes)
            )
            empire = self.get_first_by_index(empire_number, empires, empires_indexes)
            empire_power = self.cost_function.function(empire)
            empire_total_power = tf.add(empire_power,
                                        tf.multiply(self.avg_colonies_power, current_empire_colonies_power))
            return tf.subtract(index, 1), self.broadcastable_where(tf.equal(empires_indexes, empire_number),
                                                                   tf.reshape(empire_total_power, [-1]),
                                                                   total_power)

        empire_numbers, _ = tf.unique(empires_indexes)
        colonies_power = self.evaluate_countries_power(colonies)
        _, total_cost = tf.while_loop(condition,
                                      evaluate_empire_power,
                                      (tf.subtract(tf.size(empire_numbers), 1),
                                       tf.ones([self.num_of_colonies, 1], tf.float64))
                                      )
        total_cost = tf.squeeze(total_cost)
        normalised_total_cost = tf.subtract(tf.reduce_max(total_cost), total_cost)
        total_power = tf.abs(tf.truediv(normalised_total_cost, tf.reduce_sum(normalised_total_cost)))
        return total_power

    def check_number_of_imperialist(self, empires_indexes, worst_empire_number):
        self.num_of_imperialist = tf.where(
            tf.equal(0, tf.reduce_sum(tf.cast(tf.equal(empires_indexes, worst_empire_number), tf.int32))),
            tf.subtract(self.num_of_imperialist, 1),
            self.num_of_imperialist
        )

    # @tf.function
    def merging_of_similar(self, empires, colonies, empires_indexes):
        def outer_condition(outer_index, *unused):
            return tf.less(outer_index, self.num_of_imperialist)

        def outer_loop(outer_empire_index, empires, empires_indexes):
            def inner_condition(inner_index, *unused):
                return tf.less(inner_index, self.num_of_imperialist)

            def inner_loop(inner_index, outer_index, outer_empire, empires, empires_indexes):
                inner_empire = self.get_first_by_index(inner_index, empires, empires_indexes)
                new_empires, new_empires_indexes = tf.cond(
                    self.is_empty(inner_empire),
                    lambda: self.merge_empires(outer_index,
                                               inner_index,
                                               outer_empire,
                                               inner_empire,
                                               empires,
                                               empires_indexes),
                    lambda: (empires, empires_indexes)
                )
                return tf.add(1, inner_index), outer_index, outer_empire, new_empires, new_empires_indexes

            outer_empire = self.get_first_by_index(outer_empire_index, empires, empires_indexes)
            _, new_empires, new_empires_indexes = tf.cond(
                self.is_empty(outer_empire),
                lambda: tf.while_loop(inner_condition, inner_loop,
                                      (tf.add(1, outer_empire_index),
                                       outer_empire_index,
                                       outer_empire,
                                       empires,
                                       empires_indexes)
                                      ),
                lambda: (constants.int_zero, empires, empires_indexes)
            )
            return tf.add(1, outer_empire_index), new_empires, new_empires_indexes

        _, new_empires, new_empires_indexes = tf.while_loop(outer_condition, outer_loop,
                                                            (constants.int_zero, empires, empires_indexes))
        return new_empires, colonies, new_empires_indexes

    def merge_empires(self, outer_index, inner_index, outer_empire, inner_empire, empires, empires_indexes):
        def merge():
            condition = tf.equal(empires_indexes, inner_index)
            new_empires = self.broadcastable_where(condition, outer_empire, empires)
            new_empires_indexes = tf.squeeze(
                self.broadcastable_where(condition, outer_index, tf.expand_dims(empires_indexes, 1))
            )
            return new_empires, new_empires_indexes

        return tf.cond(
            self.are_two_empires_close(outer_empire, inner_empire),
            merge,
            lambda: (constants.int_zero, empires, empires_indexes)
        )

    def are_two_empires_close(self, first_empire, second_empire):
        return tf.less(tf.norm(tf.subtract(first_empire, second_empire)), self.close_empires_rating)

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

    def evaluate_countries_power(self, countries):
        return tf.squeeze(tf.transpose(tf.map_fn(self.cost_function.function, countries)))

    def broadcastable_where(self, bool_mask, vector, tensor):
        return tf.where(self.broadcast_boolean_mask(bool_mask, tensor.shape),
                        tf.multiply(tf.ones_like(tensor), vector),
                        tensor)

    def broadcast_boolean_mask(self, bool_vector, shape):
        return tf.reshape(tf.repeat(bool_vector, shape[1]), shape)

    def replace_row_in_matrix(self, tensor, row, index):
        mask = tf.equal(tf.range(tensor.shape[0]), index)
        return self.broadcastable_where(mask, row, tensor)

    @staticmethod
    def normalize_vectors(vector):
        norm = tf.sqrt(tf.reduce_sum(tf.square(vector), keepdims=True))
        return vector / norm

    @staticmethod
    def calculate_distribution(shape, minval, maxval, seed=None):
        return tf.random.uniform(shape=shape,
                                 minval=minval,
                                 maxval=maxval,
                                 dtype=dtypes.float64,
                                 seed=seed
                                 )

    @staticmethod
    def is_empty(tensor):
        return tf.equal(tf.size(tensor), constants.int_zero)

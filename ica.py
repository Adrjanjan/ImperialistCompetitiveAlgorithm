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
        self.deviation_assimilation = tf.constant(deviation_assimilation, dtype=tf.float64)
        self.direct_assimilation = tf.constant(direct_assimilation, dtype=tf.float64)
        self.avg_colonies_power = tf.constant(avg_colonies_power, dtype=tf.float64)
        self.revolution_rate = tf.constant(revolution_rate, dtype=tf.float64)
        self.close_empires_rating = tf.constant(close_empires_rating, dtype=tf.float64)
        self.is_loggable = log
        self.timeit = 0
        self.seed = seed
        self.value_mean = tf.metrics.Mean()

    @tf.function
    def eval(self):
        # @tf.function
        def stop_condition(index, *unused):
            # TODO think about stop condition where there is only one empire
            return tf.greater_equal(index, constants.int_zero)

        @tf.function
        def main_loop(index, empires, colonies, empires_numbers):
            empires, colonies, empires_numbers = self.assimilation(empires, colonies, empires_numbers)
            empires, colonies, empires_numbers = self.revolution(empires, colonies, empires_numbers)
            empires, colonies, empires_numbers = self.swap_strongest(empires, colonies, empires_numbers)
            empires, colonies, empires_numbers = self.competition(empires, colonies, empires_numbers)
            empires, colonies, empires_numbers = self.merging_of_similar(empires, colonies, empires_numbers)
            self.collect_data(empires, colonies, empires_numbers)
            return tf.subtract(index, 1), empires, colonies, empires_numbers

        # country is a row of matrix where each value is parameter of some cost function
        # colonies is matrix, where each row represents colony country, no duplicates
        # empires is matrix, where each row represents empire country, duplicates
        # corresponding row in matrices colonies and empires represents connection
        # between colony belonging to an empire

        self.start_benchmark()
        countries = self.initialize_countries()
        empires, colonies, empires_numbers = self.create_empires(countries)
        # index_and_results =
        _, empires, _, _ = tf.while_loop(stop_condition, main_loop,
                                         (tf.constant(self.max_iterations), empires, colonies, empires_numbers))
        # tf.print(empires.numpy())
        # tf.print(colonies.numpy())
        solution = empires[0]
        self.finish_benchmark(solution)
        return solution

    @tf.function
    def initialize_countries(self):
        return self.calculate_distribution((self.num_of_countries, self.dimension), self.lower, self.upper)

    @tf.function
    def create_empires(self, countries):
        @tf.function
        def condition(index, *unused):
            return tf.greater_equal(index, constants.int_zero)

        @tf.function
        def construct_empires(index, number_of_colonies, top_indexes, empires, empires_numbers, insert_index):
            current_num_of_colonies = tf.gather(number_of_colonies, index)
            current_empire_index = tf.gather(top_indexes, index)
            new_empire_matrix = tf.reshape(
                tf.tile(tf.gather(countries, current_empire_index), [current_num_of_colonies]),
                [current_num_of_colonies, self.dimension])
            new_empire_indexes = tf.reshape(tf.tile([index], [current_num_of_colonies]),
                                            [current_num_of_colonies, 1])

            concatenated_empires = self.insert_in_tensor_from_index(empires, new_empire_matrix,
                                                                    insert_index, current_num_of_colonies)
            concatenated_empires_numbers = self.insert_in_tensor_from_index(empires_numbers, new_empire_indexes,
                                                                            insert_index, current_num_of_colonies)

            new_index = tf.subtract(index, 1)
            return new_index, number_of_colonies, top_indexes, concatenated_empires, concatenated_empires_numbers, \
                   tf.add(insert_index, tf.size(new_empire_indexes))

        power = self.evaluate_countries_power(countries)
        # (-1)*power is important to get lowest power first - lower cost means bigger power
        top_powers, top_indexes = tf.nn.top_k(-power, self.num_of_imperialist)
        top_powers = tf.negative(top_powers)
        number_of_colonies = self.calculate_number_of_colonies(top_powers)
        temp_empires = tf.zeros((self.num_of_colonies, self.dimension), dtype=tf.float64)
        temp_empires_numbers = tf.zeros((self.num_of_colonies, 1), tf.int32)
        iteration_initialization_params = (tf.constant(self.num_of_imperialist - 1),
                                           number_of_colonies,
                                           top_indexes,
                                           temp_empires,
                                           temp_empires_numbers,
                                           tf.constant(0)
                                           )

        _, _, _, empires, empires_numbers, _ = tf.while_loop(condition, construct_empires,
                                                             iteration_initialization_params,
                                                             shape_invariants=(
                                                                 tf.TensorShape(None),
                                                                 tf.TensorShape([None]),
                                                                 tf.TensorShape([None]),
                                                                 tf.TensorShape([None, self.dimension]),
                                                                 tf.TensorShape([None, 1]),
                                                                 tf.TensorShape(None)
                                                             ))
        colonies = self.delete_rows(countries, top_indexes)
        return empires, colonies, tf.squeeze(empires_numbers)

    @tf.function
    def insert_in_tensor_from_index(self, tensor, to_insert, index, insert_size):
        end = tensor[(index + insert_size):]
        result = tf.cond(tf.equal(index, constants.int_zero),
                         lambda: tf.concat([to_insert, end], 0),
                         lambda: tf.concat([tensor[0:index], to_insert, end], 0))
        return result

    @tf.function
    def calculate_number_of_colonies(self, top_powers):
        max_cost = tf.reduce_max(top_powers[:])
        x1 = tf.constant(1.3, dtype=tf.float64)
        x2 = tf.constant(0.7, dtype=tf.float64)
        normalised_cost = tf.where(max_cost > 0, tf.subtract(tf.multiply(x1, max_cost), top_powers),
                                   tf.subtract(tf.multiply(x2, max_cost), top_powers))
        num_of_colonies_per_empire = tf.round(
            tf.multiply(tf.truediv(normalised_cost, tf.reduce_sum(normalised_cost)), self.num_of_colonies))

        return tf.cast(self.add_one_to_top_empire(num_of_colonies_per_empire), tf.int32)

    @tf.function
    def add_one_to_top_empire(self, n):
        more = tf.greater(tf.reduce_sum(n), self.num_of_colonies)
        less = tf.less(tf.reduce_sum(n), self.num_of_colonies)
        return tf.cond(
            less,
            lambda: tf.concat([tf.expand_dims(tf.add(n[0], 1), 0), n[1:]], 0),
            lambda: tf.cond(
                more,
                lambda: tf.concat([tf.expand_dims(tf.subtract(n[0], 1), 0), n[1:]], 0),
                lambda: n
            )

        )

    @tf.function
    def assimilation(self, empires, colonies, empires_numbers):
        difference = tf.subtract(empires, colonies)
        uniform = self.calculate_distribution((self.num_of_colonies, self.dimension),
                                              constants.zero,
                                              tf.multiply(tf.abs(difference), self.direct_assimilation)
                                              )
        new_colonies = tf.clip_by_value(
            tf.add(colonies, tf.multiply(uniform, self.normalize_vectors(difference))),
            self.lower,
            self.upper
        )
        return empires, new_colonies, empires_numbers

    @tf.function
    def revolution(self, empires, colonies, empires_numbers):
        to_revolve = self.calculate_distribution((self.num_of_colonies, 1), constants.zero, constants.one)
        possible_new_positions = self.calculate_distribution((self.num_of_colonies, self.dimension), self.lower,
                                                             self.upper)
        chance_to_revolve = self.broadcast_boolean_mask(tf.less(to_revolve, self.revolution_rate),
                                                        (self.num_of_colonies, self.dimension))

        new_colonies = tf.where(chance_to_revolve, possible_new_positions, colonies)
        return empires, new_colonies, empires_numbers

    @tf.function
    def swap_strongest(self, empires, colonies, empires_numbers):
        @tf.function
        def condition(iterator, empires_numbers_to_check, *unused):
            return tf.less(iterator, tf.size(empires_numbers_to_check))

        @tf.function
        def body(iterator, empires_numbers_to_check, empires, colonies, empires_numbers):
            current_empire_number = tf.gather(empires_numbers_to_check, iterator)
            current_empire_power = self.get_first_by_empire_number(current_empire_number, empires_power,
                                                                   empires_numbers)
            current_colonies_power = self.get_all_by_empire_number(current_empire_number, colonies_power,
                                                                   empires_numbers)
            best_colony_power = tf.reduce_min(current_colonies_power)
            empire_not_exists = self.is_empty(current_empire_power)
            swapped_empires, swapped_colonies = tf.cond(
                empire_not_exists,
                lambda: (empires, colonies),
                lambda: tf.cond(current_empire_power > best_colony_power,
                                lambda: self.swap(current_empire_number, colonies, colonies_indexes, empires,
                                                  empires_numbers, colonies_power_with_index),
                                lambda: (empires, colonies)
                                )
            )

            return tf.add(1, iterator), empires_numbers_to_check, swapped_empires, swapped_colonies, empires_numbers

        colonies_power = self.evaluate_countries_power(colonies)
        colonies_indexes = self.create_index_column(colonies_power)
        colonies_power_with_index = self.concatenate_colonies_power_and_index(colonies_indexes, colonies_power)

        empires_power = self.evaluate_countries_power(empires)
        empires_numbers_to_check, _ = tf.unique(empires_numbers)
        swap_initial_params = (constants.int_zero, empires_numbers_to_check, empires, colonies, empires_numbers)

        _, _, new_empires, new_colonies, _ = tf.while_loop(condition, body, swap_initial_params)
        return new_empires, new_colonies, empires_numbers

    @tf.function
    def swap(self, current_empire_number, colonies, colonies_indexes, empires, empires_numbers,
             colonies_power_with_index):
        current_empire = self.get_first_by_empire_number(current_empire_number, empires, empires_numbers)
        current_colonies_power_with_index = self.get_all_by_empire_number(current_empire_number,
                                                                          colonies_power_with_index,
                                                                          empires_numbers)
        current_colonies_power = tf.reshape(tf.squeeze(current_colonies_power_with_index[:, :-1]), [-1])
        current_colonies_index = tf.cast(current_colonies_power_with_index[:, -1], tf.int32)

        _, best_colony_current_index = tf.nn.top_k(-current_colonies_power)
        best_colony_real_index = tf.gather(current_colonies_index, best_colony_current_index)
        best_colony = tf.gather(colonies, best_colony_real_index)

        new_e = self.broadcastable_where(tf.equal(empires_numbers, current_empire_number), best_colony, empires)
        new_c = self.broadcastable_where(tf.equal(colonies_indexes, best_colony_real_index), current_empire,
                                         colonies)
        return new_e, new_c

    @staticmethod
    @tf.function
    def create_index_column(tensor):
        return tf.cast(tf.range(tensor.shape[0])[:, None], tf.int32)

    @tf.function
    def concatenate_colonies_power_and_index(self, colonies_indexes, colonies_power):
        return tf.concat(
            [tf.reshape(colonies_power, [self.num_of_colonies, 1]),
             tf.cast(colonies_indexes, dtype=tf.float64)],
            axis=1
        )

    @tf.function
    def competition(self, empires, colonies, empires_numbers):
        # calculate total power
        total_power = self.calculate_empire_total_power(empires, colonies, empires_numbers)

        # select top and worst empire
        top_empire, top_empire_number = self.select_top_empire(empires, empires_numbers, total_power)
        _, worst_empire_index = tf.math.top_k(-total_power)
        worst_empire_number = tf.gather(empires_numbers, worst_empire_index)

        # select worst colony from worst empire
        worst_colony_index = self.select_worst_colony_index(worst_empire_number, colonies, empires_numbers)

        # set new empire as empire of worst colony
        new_empires_numbers = tf.squeeze(
            self.replace_row_in_matrix(tf.expand_dims(empires_numbers, 1), top_empire_number, worst_colony_index)
        )
        new_empires = self.replace_row_in_matrix(empires, top_empire, worst_colony_index)

        # reduce num_of_imperialists if it was worst empire last presence
        self.check_number_of_imperialist(new_empires_numbers, worst_empire_number)
        return new_empires, colonies, new_empires_numbers

    @tf.function
    def select_worst_colony_index(self, empire_index, colonies, empires_indexes):
        colonies_with_index = tf.concat([colonies, tf.cast(tf.range(self.num_of_colonies)[:, None], dtype=tf.float64)],
                                        axis=1)
        current_empire_colonies_with_index = self.get_all_by_empire_number(empire_index, colonies_with_index,
                                                                           empires_indexes)
        current_empire_colonies = tf.reshape(current_empire_colonies_with_index[:, :-1],
                                             [current_empire_colonies_with_index.shape[0], self.dimension])
        colonies_power = tf.reshape(self.evaluate_countries_power(current_empire_colonies), [-1])
        _, current_worst_index = tf.math.top_k(colonies_power)
        current_empire_colonies_index = tf.cast(current_empire_colonies_with_index[:, -1], tf.int32)
        worst_colony_index = tf.gather(current_empire_colonies_index, current_worst_index)
        return worst_colony_index

    @tf.function
    def select_top_empire(self, empires, empires_numbers, total_power):
        distribution = self.calculate_distribution((self.num_of_colonies,), constants.zero, total_power)
        _, top_empire_index = tf.nn.top_k(tf.subtract(total_power, distribution))
        top_empire = tf.gather(empires, top_empire_index)
        top_empire_number = tf.gather(empires_numbers, top_empire_index)
        return top_empire, top_empire_number

    @tf.function
    def calculate_empire_total_power(self, empires, colonies, empires_numbers):
        @tf.function
        def condition(index, *unused):
            return tf.greater_equal(index, constants.int_zero)

        @tf.function
        def evaluate_empire_power(index, total_power):
            empire_number = tf.gather(empire_numbers, index)
            empire_colonies_power_sum = tf.reduce_sum(
                self.get_all_by_empire_number(empire_number, colonies_power, empires_numbers))
            empire = self.get_first_by_empire_number(empire_number, empires, empires_numbers)
            empire_power = self.cost_function.function(empire)
            empire_total_power = tf.add(empire_power, tf.multiply(self.avg_colonies_power, empire_colonies_power_sum))
            new_total_power = self.broadcastable_where(tf.equal(empires_numbers, empire_number),
                                                       tf.reshape(empire_total_power, [-1]),
                                                       total_power
                                                       )
            return tf.subtract(index, 1), new_total_power

        empire_numbers, _ = tf.unique(empires_numbers)
        colonies_power = self.evaluate_countries_power(colonies)

        initial_params = (tf.subtract(tf.size(empire_numbers), 1), tf.ones([self.num_of_colonies, 1], dtype=tf.float64))
        _, total_cost = tf.while_loop(condition, evaluate_empire_power, initial_params)

        total_cost = tf.squeeze(total_cost)
        normalised_total_cost = tf.subtract(tf.reduce_max(total_cost), total_cost)
        total_power = tf.abs(tf.truediv(normalised_total_cost, tf.reduce_sum(normalised_total_cost)))
        return total_power

    @tf.function
    def check_number_of_imperialist(self, empires_numbers, worst_empire_number):
        self.num_of_imperialist = tf.where(
            tf.equal(0, tf.reduce_sum(tf.cast(tf.equal(empires_numbers, worst_empire_number), tf.int32))),
            tf.subtract(self.num_of_imperialist, 1),
            self.num_of_imperialist
        )

    @tf.function
    def merging_of_similar(self, empires, colonies, empires_numbers):
        @tf.function
        def outer_condition(_, checked_count, *unused):
            return tf.less(checked_count, self.num_of_imperialist)

        @tf.function
        def outer_loop(outer_empire_number, checked_count, empires, empires_numbers):
            @tf.function
            def inner_condition(inner_index, *unused):
                return tf.less(inner_index, self.num_of_imperialist)

            @tf.function
            def inner_loop(inner_empire_number, outer_empire_number, outer_empire, empires, empires_numbers):
                inner_empire = self.get_first_by_empire_number(inner_empire_number, empires, empires_numbers)
                new_empires, new_empires_numbers = tf.cond(
                    self.is_empty(inner_empire),
                    lambda: (empires, empires_numbers),
                    lambda: self.merge_empires(outer_empire_number,
                                               inner_empire_number,
                                               outer_empire,
                                               inner_empire,
                                               empires,
                                               empires_numbers)
                )
                return tf.add(1, inner_empire_number), outer_empire_number, \
                       outer_empire, new_empires, new_empires_numbers

            outer_empire = self.get_first_by_empire_number(outer_empire_number, empires, empires_numbers)
            inner_initial_params = (tf.add(1, outer_empire_number),
                                    outer_empire_number,
                                    outer_empire,
                                    empires,
                                    empires_numbers)
            empire_with_current_number_exists = self.is_empty(outer_empire)
            _, _, _, new_empires, new_empires_numbers = tf.cond(
                empire_with_current_number_exists,
                lambda: inner_initial_params,
                lambda: tf.while_loop(inner_condition, inner_loop, inner_initial_params)
            )

            new_checked_count = tf.cond(
                empire_with_current_number_exists,
                lambda: checked_count,
                lambda: tf.add(checked_count, 1),
            )
            return tf.add(1, outer_empire_number), new_checked_count, new_empires, new_empires_numbers

        outer_initial_params = (constants.int_zero, constants.int_zero, empires, empires_numbers)
        _, _, new_empires, new_empires_numbers = tf.while_loop(outer_condition, outer_loop, outer_initial_params)
        return new_empires, colonies, new_empires_numbers

    @tf.function
    def merge_empires(self, outer_number, inner_number, outer_empire, inner_empire, empires, empires_numbers):
        @tf.function
        def merge():
            condition = tf.equal(empires_numbers, inner_number)
            new_empires = self.broadcastable_where(condition, outer_empire, empires)
            new_empires_numbers = tf.squeeze(
                self.broadcastable_where(condition, outer_number, tf.expand_dims(empires_numbers, 1)))
            return new_empires, new_empires_numbers

        return tf.cond(
            self.are_two_empires_close(outer_empire, inner_empire),
            merge,
            lambda: (empires, empires_numbers)
        )

    @tf.function
    def are_two_empires_close(self, first_empire, second_empire):
        return tf.less(tf.norm(tf.subtract(first_empire, second_empire)), self.close_empires_rating)

    @tf.function
    def start_benchmark(self):
        tf.print("|----------------- START -----------------|")
        self.timeit = timeit.default_timer()

    @tf.function
    def finish_benchmark(self, result):
        if self.is_loggable:
            self.evaluation_time = timeit.default_timer() - self.timeit
            tf.print(result)
            tf.print(self.cost_function.function(result))
            tf.print("|----------------- PARAMETERS -----------------|")
            tf.print("| Iterations:      ", self.max_iterations)
            tf.print("| Bounds:          ", self.lower, self.upper)
            tf.print("| Dimension:       ", self.dimension)
            tf.print("| Empires number:  ", self.num_of_imperialist)
            tf.print("| Colonies number: ", self.num_of_colonies)
            tf.print("| Evaluation time: ", self.evaluation_time)
            tf.print("| Countries mean value: ", self.value_mean.result())

    @tf.function
    def get_all_by_empire_number(self, empire_number, tensor, empires_numbers):
        return tf.cond(
            self.is_empty(tensor),
            lambda: tensor,
            lambda: tf.boolean_mask(tf.squeeze(tensor), tf.equal(empire_number, empires_numbers))
        )

    @tf.function
    def get_first_by_empire_number(self, empire_number, tensor, empires_numbers):
        result = tf.boolean_mask(tf.squeeze(tensor), tf.equal(empire_number, empires_numbers))
        return tf.cond(
            self.is_empty(tensor),
            lambda: tensor,
            lambda: tf.cond(self.is_empty(result),
                            lambda: result,
                            lambda: tf.squeeze(tf.gather(result, 0))
                            )
        )

    @tf.function
    def delete_rows(self, tensor, index):
        return tf.boolean_mask(tensor, tf.subtract(1.0, tf.reduce_sum(tf.one_hot(index, self.num_of_countries), 0)))

    @tf.function
    def evaluate_countries_power(self, countries):
        return tf.squeeze(tf.transpose(tf.map_fn(self.cost_function.function, countries)))

    @tf.function
    def broadcastable_where(self, bool_mask, vector, tensor):
        return tf.where(self.broadcast_boolean_mask(bool_mask, tensor.shape),
                        tf.multiply(tf.ones_like(tensor), vector),
                        tensor)

    @tf.function
    def broadcast_boolean_mask(self, bool_vector, shape):
        return tf.reshape(tf.repeat(bool_vector, shape[1]), shape)

    @tf.function
    def replace_row_in_matrix(self, tensor, row, index):
        mask = tf.equal(tf.range(tensor.get_shape()[0]), index)
        return self.broadcastable_where(mask, row, tensor)

    @staticmethod
    @tf.function
    def normalize_vectors(vector):
        norm = tf.sqrt(tf.reduce_sum(tf.square(vector), keepdims=True))
        return vector / norm

    @staticmethod
    @tf.function
    def calculate_distribution(shape, minval, maxval, seed=None):
        return tf.random.uniform(shape=shape,
                                 minval=minval,
                                 maxval=maxval,
                                 dtype=tf.float64,
                                 seed=seed
                                 )

    @staticmethod
    @tf.function
    def is_empty(tensor):
        return tf.equal(tf.size(tensor), constants.int_zero)

    def collect_data(self, empires, colonies, empires_numbers):
        self.value_mean(self.evaluate_countries_power(empires))

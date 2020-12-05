import timeit
import tensorflow as tf
import numpy as np
from ica import init, helpers, assimillation, revolution, swap_strongest, competition, merging
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
        # Country is a row (x*) of matrix where each value is parameter of some cost function (F),
        # where F(x*) is value of cost function for country.
        # Colonies is matrix, where each row represents colony country, no duplicates.
        # Empires is matrix, where each row represents empire country, duplicates.
        # Corresponding row in matrices colonies and empires represents connection
        # between colony belonging to an empire. The same relation is mapped in empires_numbers vector.

        self.start_benchmark()
        countries = self.initialize_countries()
        empires, colonies, empires_numbers = self.create_empires(countries)
        loop_params = (tf.constant(self.max_iterations), empires, colonies, empires_numbers)

        _, empires, _, _ = tf.while_loop(self.stop_condition, self.main_loop, loop_params)

        solution = empires[0]
        self.finish_benchmark(solution)
        return solution

    @tf.function
    def create_empires(self, countries):
        return init.create_empires(countries=countries, dimension=self.dimension, num_of_colonies=self.num_of_colonies,
                                   num_of_imperialist=self.num_of_imperialist, cost_function=self.cost_function.function
                                   )

    @tf.function
    def initialize_countries(self):
        return init.initialize_countries(num_of_countries=self.num_of_countries, dimension=self.dimension,
                                         lower=self.lower, upper=self.upper)

    @tf.function
    def stop_condition(self, index, _, __, empires_numbers):
        unique, _ = tf.unique(empires_numbers)
        return tf.logical_and(tf.greater(index, constants.int_zero),
                              tf.logical_not(tf.equal(constants.int_one, tf.size(unique)))
                              )

    @tf.function
    def main_loop(self, index, empires, colonies, empires_numbers):
        colonies = self.assimilation(colonies, empires)
        colonies = self.revolution(colonies)
        empires, colonies = self.swap_strongest(colonies, empires, empires_numbers)
        empires, empires_numbers, self.num_of_imperialist = self.competition(colonies, empires, empires_numbers)
        # empires, empires_numbers = self.merging(empires, empires_numbers)
        self.collect_data(empires, colonies, empires_numbers)
        return tf.subtract(index, 1), empires, colonies, empires_numbers

    @tf.function
    def assimilation(self, colonies, empires):
        return assimillation.assimilation(colonies=colonies, empires=empires, num_of_colonies=self.num_of_colonies,
                                          dimension=self.dimension, lower=self.lower, upper=self.upper,
                                          direct_assimilation=self.direct_assimilation)

    @tf.function
    def revolution(self, colonies):
        return revolution.revolution(colonies=colonies, num_of_colonies=self.num_of_colonies,
                                     dimension=self.dimension, lower=self.lower, upper=self.upper,
                                     revolution_rate=self.revolution_rate)

    @tf.function
    def swap_strongest(self, colonies, empires, empires_numbers):
        return swap_strongest.swap_strongest(colonies=colonies, empires=empires, empires_numbers=empires_numbers,
                                             num_of_colonies=self.num_of_colonies,
                                             cost_function=self.cost_function.function)

    @tf.function
    def competition(self, colonies, empires, empires_numbers):
        return competition.competition(colonies=colonies, empires=empires, empires_numbers=empires_numbers,
                                       num_of_colonies=self.num_of_colonies, num_of_imperialist=self.num_of_imperialist,
                                       dimension=self.dimension, cost_function=self.cost_function.function,
                                       avg_colonies_power=self.avg_colonies_power)

    @tf.function
    def merging(self, empires, empires_numbers):
        return merging.merging_of_similar(empires=empires, empires_numbers=empires_numbers,
                                          num_of_imperialist=self.num_of_imperialist,
                                          close_empires_rating=self.close_empires_rating)

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

    def collect_data(self, empires, colonies, empires_numbers):
        self.value_mean(helpers.evaluate_countries_power(empires, self.cost_function.function))

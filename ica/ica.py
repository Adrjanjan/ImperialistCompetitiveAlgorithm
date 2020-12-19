import timeit
import tensorflow as tf
from ica import init, assimillation, revolution, swap_strongest, competition, helpers
from algorithm_evaluation.test_functions import CostFunction
import constants


class ICA:

    def __init__(self, cost_function: CostFunction, num_of_countries=100, num_of_imperialist=4, max_iterations=10000,
                 direct_assimilation=0.5, avg_colonies_power=0.1, revolution_rate=0.2, log=False, seed=42):
        self.evaluation_time = -1
        self.cost_function = cost_function
        self.dimension = cost_function.dimension
        self.lower = cost_function.lower
        self.upper = cost_function.upper
        self.num_of_countries = num_of_countries
        self.num_of_imperialist = num_of_imperialist
        self.num_of_colonies = num_of_countries - num_of_imperialist
        self.max_iterations = max_iterations
        self.direct_assimilation = tf.constant(direct_assimilation, dtype=tf.float64)
        self.avg_colonies_power = tf.constant(avg_colonies_power, dtype=tf.float64)
        self.revolution_rate = tf.constant(revolution_rate, dtype=tf.float64)
        self.is_loggable = log
        self.timeit = None
        self.seed = seed
        self.final_iteration = None
        self.result = None
        self.reached_minimum = None
        self.top_empire_index = None
        self.lowest_cost_per_iteration = tf.TensorArray(tf.float64, size=0, dynamic_size=True, clear_after_read=False)

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
        loop_params = (constants.int_zero, colonies, empires, empires_numbers)

        self.final_iteration, empires, _, _ = tf.while_loop(self.stop_condition, self.main_loop, loop_params)

        empires_power = helpers.evaluate_countries_power(empires, self.cost_function.function)
        self.reached_minimum, self.top_empire_index = tf.nn.top_k(-empires_power)
        self.reached_minimum = -tf.squeeze(self.reached_minimum)
        self.result = tf.gather(empires, self.top_empire_index)
        self.finish_benchmark()
        return self.result

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
        return tf.logical_and(tf.less(index, self.max_iterations),
                              tf.logical_not(tf.equal(constants.int_one, tf.size(unique)))
                              )

    @tf.function
    def main_loop(self, index, init_colonies, init_empires, init_empires_numbers):
        colonies_assimilation = self.assimilation(colonies=init_colonies, empires=init_empires)
        colonies_revolution = self.revolution(colonies=colonies_assimilation)
        colonies_swap, empires_swap, empires_power_swap, colonies_power_swap = self.swap_strongest(
            colonies=colonies_revolution,
            empires=init_empires,
            empires_numbers=init_empires_numbers
        )
        empires_competition, empires_numbers_competition, empires_power_competition, colonies_power_competition, self.num_of_imperialist = self.competition(
            empires=empires_swap,
            empires_numbers=init_empires_numbers,
            empires_power=empires_power_swap,
            colonies_power=colonies_power_swap
        )
        self.collect_data(index, empires_power_competition)
        return tf.add(index, 1), colonies_swap, empires_competition, empires_numbers_competition

    @tf.function
    def assimilation(self, colonies, empires):
        return assimillation.assimilation(colonies=colonies,
                                          empires=empires,
                                          num_of_colonies=self.num_of_colonies,
                                          dimension=self.dimension,
                                          lower=self.lower,
                                          upper=self.upper,
                                          direct_assimilation=self.direct_assimilation
                                          )

    @tf.function
    def revolution(self, colonies):
        return revolution.revolution(colonies=colonies,
                                     num_of_colonies=self.num_of_colonies,
                                     dimension=self.dimension,
                                     lower=self.lower,
                                     upper=self.upper,
                                     revolution_rate=self.revolution_rate
                                     )

    @tf.function
    def swap_strongest(self, colonies, empires, empires_numbers):
        return swap_strongest.swap_strongest(colonies=colonies,
                                             empires=empires,
                                             empires_numbers=empires_numbers,
                                             num_of_colonies=self.num_of_colonies,
                                             cost_function=self.cost_function.function
                                             )

    @tf.function
    def competition(self, empires, empires_numbers, empires_power, colonies_power):
        return competition.competition(empires=empires,
                                       empires_numbers=empires_numbers,
                                       num_of_colonies=self.num_of_colonies,
                                       num_of_imperialist=self.num_of_imperialist,
                                       dimension=self.dimension,
                                       avg_colonies_power=self.avg_colonies_power,
                                       empires_power=empires_power,
                                       colonies_power=colonies_power
                                       )

    @tf.function
    def start_benchmark(self):
        if self.is_loggable:
            tf.print("|----------------- START -----------------|")
        self.timeit = timeit.default_timer()

    @tf.function
    def finish_benchmark(self):
        self.evaluation_time = timeit.default_timer() - self.timeit
        if self.is_loggable:
            tf.print("|----------------- PARAMETERS -----------------|")
            tf.print("| Result value:    ", self.cost_function.function(self.result))
            tf.print("| Iterations:      ", self.max_iterations)
            tf.print("| Bounds:          ", self.lower, self.upper)
            tf.print("| Dimension:       ", self.dimension)
            tf.print("| Empires number:  ", self.num_of_imperialist)
            tf.print("| Colonies number: ", self.num_of_colonies)
            tf.print("| Evaluation time: ", self.evaluation_time)

    def collect_data(self, index, empires_power):
        self.lowest_cost_per_iteration = self.lowest_cost_per_iteration.write(index, tf.reduce_min(empires_power))

    def get_evaluation_data(self):
        return {
            "evaluation_time": self.evaluation_time,
            "reached_minimum": self.reached_minimum.numpy(),
            "final_iteration": self.final_iteration.numpy(),
            "max_iterations": self.max_iterations,
            "empires_number": self.num_of_imperialist.numpy(),
            "colonies_number": self.num_of_colonies,
            "direct_assimilation": self.direct_assimilation.numpy(),
            "avg_colonies_power": self.avg_colonies_power.numpy(),
            "revolution_rate": self.revolution_rate.numpy(),
            "lowest_cost_per_iteration": self.lowest_cost_per_iteration.stack(),
            "solution_error": tf.truediv(
                tf.abs(tf.reduce_sum(self.result - self.cost_function.o_vector[:self.dimension])),
                self.dimension).numpy(),
            "solution_distance": tf.sqrt(
                tf.reduce_sum(tf.square(self.result - self.cost_function.o_vector[:self.dimension]))).numpy(),
        }

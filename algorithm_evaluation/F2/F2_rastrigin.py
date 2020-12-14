import tensorflow as tf
from algorithm_evaluation.evaluation_helper import *
from algorithm_evaluation.test_functions import CostFunction


class F2_Rastrigin(CostFunction):

    def __init__(self, upper=100, lower=-100, dimension=1000, o_vector="resources/F2-xopt.txt"):
        super().__init__(self.rastrigin, upper, lower, dimension, o_vector)

    @tf.function
    def rastrigin(self, vector):
        z = vector - self.o_vector
        return self.rastrigin_func(z)


tf.config.run_functions_eagerly(True)

params = {
    "num_of_countries": [500, 1000],
    "num_of_imperialist": [5, 10],
    "max_iterations": [2000],
    "deviation_assimilation": [3.14 / 4.0],
    "direct_assimilation": [0.2, 0.7],
    "avg_colonies_power": [0.1],
    "revolution_rate": [0.2, 0.5],
    "close_empires_rating": [0.1],
    "seed": [420]
}

iterations_results = gridsearch(F2_Rastrigin(), params)

print(iterations_results)

file_path = "results/"
create_and_save_plots_to_file(iterations_results, file_path, "F2")
print(create_and_save_params_grid_as_latex_table(iterations_results, file_path, "F2"))
import tensorflow as tf
from algorithm_evaluation.evaluation_helper import *
from test_functions import CostFunction


class F1_Elliptic(CostFunction):

    def __init__(self, upper=100, lower=-100, dimension=1000, o_vector="resources/F1-xopt.txt"):
        super().__init__(self.elliptic, upper, lower, dimension, o_vector)

    @tf.function
    def elliptic(self, vector):
        z = vector - self.o_vector
        return self.elliptic_func(z)


tf.config.run_functions_eagerly(True)

params = {
    "num_of_countries": [100],
    "num_of_imperialist": [5],
    "max_iterations": [10, 20, 30, 100, 200],
    "deviation_assimilation": [3.14 / 4.0],
    "direct_assimilation": [0.2],
    "avg_colonies_power": [0.1],
    "revolution_rate": [0.2],
    "close_empires_rating": [0.1],
    "seed": [420]
}

iterations_results = gridsearch(F1_Elliptic(), params)

print(iterations_results)

file_path = "results/"
create_and_save_plots_to_file(iterations_results, file_path, "F1")
print(create_and_save_params_grid_as_latex_table(iterations_results, file_path, "F1"))

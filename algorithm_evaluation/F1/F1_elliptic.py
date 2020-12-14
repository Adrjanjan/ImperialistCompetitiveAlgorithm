from algorithm_evaluation.evaluation_helper import *
from algorithm_evaluation.test_functions import CostFunction


class F1_Elliptic(CostFunction):

    def __init__(self, upper=100, lower=-100, dimension=1000,
                 o_vector="algorithm_evaluation/F1/resources/F1-xopt.txt"
                 ):
        super().__init__(self.elliptic, upper, lower, dimension, o_vector)

    @tf.function
    def elliptic(self, vector):
        z = vector - self.o_vector
        return self.elliptic_func(z)


tf.config.run_functions_eagerly(True)

params = {
    "num_of_countries": [500, 1000],
    "num_of_imperialist": [5, 10],
    "max_iterations": [4000],
    "deviation_assimilation": [3.14 / 4.0],
    "direct_assimilation": [0.2, 0.7],
    "avg_colonies_power": [0.1],
    "revolution_rate": [0.2, 0.5],
    "close_empires_rating": [0.1],
    "seed": [420]
}
file_path = "algorithm_evaluation/F1/results/"
iterations_results = gridsearch(F1_Elliptic(), params, file_path)

print(iterations_results)

create_and_save_plots_to_file(iterations_results, file_path, "F1")
print(create_and_save_params_grid_as_latex_table(iterations_results, file_path, "F1"))

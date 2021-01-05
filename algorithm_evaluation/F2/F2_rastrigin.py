from algorithm_evaluation.evaluation_helper import *
from algorithm_evaluation.test_functions import CostFunction


class F2_Rastrigin(CostFunction):

    def __init__(self, upper=5, lower=-5, dimension=1000,
                 o_vector="algorithm_evaluation/F2/resources/F2-xopt.txt"
                 ):
        super().__init__(self.rastrigin, upper, lower, dimension, o_vector)

    @tf.function
    def rastrigin(self, matrix):
        z = matrix - self.o_vector
        return self.rastrigin_func(z)


tf.config.run_functions_eagerly(True)

params = {
    "num_of_countries": [4000, 2000],
    "num_of_imperialist": [10],
    "max_iterations": [10000],
    "direct_assimilation": [0.7, 1.1],
    "avg_colonies_power": [0.1],
    "revolution_rate": [0.0, 0.0001],
    "seed": [None]
}

results_path = "algorithm_evaluation/F2/results/"
iterations_results = gridsearch(F2_Rastrigin(), params, results_path)
create_and_save_plots_to_file(iterations_results, results_path, "F2")
latex_table = create_and_save_params_grid_as_latex_table(iterations_results, results_path, "F2")

print(iterations_results)
print(latex_table)

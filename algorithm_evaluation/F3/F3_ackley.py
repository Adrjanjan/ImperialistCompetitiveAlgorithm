from algorithm_evaluation.evaluation_helper import *
from algorithm_evaluation.test_functions import CostFunction


class F3_Ackley(CostFunction):

    def __init__(self, upper=32, lower=-32, dimension=1000,
                 o_vector="algorithm_evaluation/F3/resources/F3-xopt.txt"
                 ):
        super().__init__(self.ackley, upper, lower, dimension, o_vector)

    @tf.function
    def ackley(self, matrix):
        z = matrix - self.o_vector
        return self.ackley_func(z)


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

result_path = "algorithm_evaluation/F3/results/"
iterations_results = gridsearch(F3_Ackley(), params, result_path)

print(iterations_results)

create_and_save_plots_to_file(iterations_results, result_path, "F3")
print(create_and_save_params_grid_as_latex_table(iterations_results, result_path, "F3"))

from algorithm_evaluation.evaluation_helper import *
from algorithm_evaluation.test_functions import CostFunction


class F9_Rastrigin(CostFunction):

    def __init__(self, upper=100, lower=-100, dimension=1000,
                 o_vector="algorithm_evaluation/F9/resources/F9-xopt.txt",
                 p_vector="algorithm_evaluation/F9/resources/F9-p.txt",
                 r_25="algorithm_evaluation/F9/resources/F9-R25.txt",
                 r_50="algorithm_evaluation/F9/resources/F9-R50.txt",
                 r_100="algorithm_evaluation/F9/resources/F9-R100.txt",
                 s="algorithm_evaluation/F9/resources/F9-s.txt",
                 w="algorithm_evaluation/F9/resources/F9-w.txt"):
        super().__init__(self.rastrigin, upper, lower, dimension, o_vector, p_vector, r_25, r_50, r_100, s, w)
        self.s_size = 20

    @tf.function
    def rastrigin(self, matrix):
        # 1
        z = matrix - self.o_vector
        # 2
        rotated = tf.transpose(self.rotation_matrix.matmul(tf.transpose(matrix)))
        return self.rastrigin_func(rotated)


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

result_path = "algorithm_evaluation/F9/results/"
iterations_results = gridsearch(F9_Rastrigin(), params, result_path)

print(iterations_results)

create_and_save_plots_to_file(iterations_results, result_path, "F9")
print(create_and_save_params_grid_as_latex_table(iterations_results, result_path, "F9"))

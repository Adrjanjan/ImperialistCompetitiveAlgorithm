from algorithm_evaluation.evaluation_helper import *
from algorithm_evaluation.test_functions import CostFunction


class F5_Rastrigin(CostFunction):

    def __init__(self, upper=100, lower=-100, dimension=1000,
                 o_vector="algorithm_evaluation/F5/resources/F5-xopt.txt",
                 p_vector="algorithm_evaluation/F5/resources/F5-p.txt",
                 r_25="algorithm_evaluation/F5/resources/F5-R25.txt",
                 r_50="algorithm_evaluation/F5/resources/F5-R50.txt",
                 r_100="algorithm_evaluation/F5/resources/F5-R100.txt",
                 s="algorithm_evaluation/F5/resources/F5-s.txt",
                 w="algorithm_evaluation/F5/resources/F5-w.txt"):
        super().__init__(self.rastrigin, upper, lower, dimension, o_vector, p_vector, r_25, r_50, r_100, s, w)
        self.s_size = 7

    @tf.function
    def rastrigin(self, matrix):
        # 1
        z = matrix - self.o_vector
        # 2
        rotated = tf.transpose(self.rotation_matrix.matmul(tf.transpose(matrix)))

        # 3
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

result_path = "algorithm_evaluation/F5/results/"
iterations_results = gridsearch(F5_Rastrigin(), params, result_path)

print(iterations_results)

create_and_save_plots_to_file(iterations_results, result_path, "F5")
print(create_and_save_params_grid_as_latex_table(iterations_results, result_path, "F5"))

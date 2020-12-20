from algorithm_evaluation.evaluation_helper import *
from algorithm_evaluation.test_functions import CostFunction


class F14_Schwefel(CostFunction):

    def __init__(self, upper=100, lower=-100, dimension=905,
                 o_vector="algorithm_evaluation/F14/resources/F14-xopt.txt",
                 p_vector="algorithm_evaluation/F14/resources/F14-p.txt",
                 r_25="algorithm_evaluation/F14/resources/F14-R25.txt",
                 r_50="algorithm_evaluation/F14/resources/F14-R50.txt",
                 r_100="algorithm_evaluation/F14/resources/F14-R100.txt",
                 s="algorithm_evaluation/F14/resources/F14-s.txt",
                 w="algorithm_evaluation/F14/resources/F14-w.txt"):
        super().__init__(self.schwefel, upper, lower, dimension, o_vector, p_vector, r_25, r_50, r_100, s, w)
        self.s_size = 20
        self.overlap = 5

    @tf.function
    def schwefel(self, matrix):
        # 2
        result = 0
        start = 0
        for i in range(self.s_size):
            result, start = self.calculate_partial_rotation(i, result, start, matrix)
        return result

    @tf.function
    def calculate_partial_rotation(self, index, result, start, vector):
        result = result + tf.reduce_sum(
            self.w[index] * self.schwefel_func(self.rotate_vector_conflict(vector, start, self.s[index], index)))
        return result, start + self.s[index]


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

result_path = "algorithm_evaluation/F14/results/"
iterations_results = gridsearch(F14_Schwefel(), params, result_path)

print(iterations_results)

create_and_save_plots_to_file(iterations_results, result_path, "F14")
print(create_and_save_params_grid_as_latex_table(iterations_results, result_path, "F14"))

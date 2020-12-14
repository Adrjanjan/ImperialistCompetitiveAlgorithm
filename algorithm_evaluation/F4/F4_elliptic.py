from algorithm_evaluation.evaluation_helper import *
from algorithm_evaluation.test_functions import CostFunction


class F4_Elliptic(CostFunction):

    def __init__(self, upper=100, lower=-100, dimension=1000,
                 o_vector="algorithm_evaluation/F4/resources/F4-xopt.txt",
                 p_vector="algorithm_evaluation/F4/resources/F4-p.txt",
                 r_25="algorithm_evaluation/F4/resources/F4-R25.txt",
                 r_50="algorithm_evaluation/F4/resources/F4-R50.txt",
                 r_100="algorithm_evaluation/F4/resources/F4-R100.txt",
                 s="algorithm_evaluation/F4/resources/F4-s.txt",
                 w="algorithm_evaluation/F4/resources/F4-w.txt"):
        super().__init__(self.elliptic, upper, lower, dimension, o_vector, p_vector, r_25, r_50, r_100, s, w)
        self.s_size = 7

    @tf.function
    def elliptic(self, vector):
        # 1
        z = vector - self.o_vector
        # 2
        result = 0
        i = 0
        start = 0
        result, i, start = self.calculate_partial_rotation(i, result, start, vector)
        result, i, start = self.calculate_partial_rotation(i, result, start, vector)
        result, i, start = self.calculate_partial_rotation(i, result, start, vector)
        result, i, start = self.calculate_partial_rotation(i, result, start, vector)
        result, i, start = self.calculate_partial_rotation(i, result, start, vector)
        result, i, start = self.calculate_partial_rotation(i, result, start, vector)
        result, i, start = self.calculate_partial_rotation(i, result, start, vector)
        # 3
        return result + tf.reduce_sum(self.elliptic_func(tf.gather(z, self.p_vector[start:]), start, self.dimension))

    @tf.function
    def calculate_partial_rotation(self, index, result, start, vector):
        result = result + tf.reduce_sum(
            self.w[index] * self.elliptic_func(self.rotate_vector(vector, start, self.s[index]), start,
                                               start + self.s[index]))
        return result, index + 1, start + self.s[index]

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

result_path = "algorithm_evaluation/F1/results/"
iterations_results = gridsearch(F4_Elliptic(), params, result_path)

print(iterations_results)

create_and_save_plots_to_file(iterations_results, result_path, "F4")
print(create_and_save_params_grid_as_latex_table(iterations_results, result_path, "F4"))

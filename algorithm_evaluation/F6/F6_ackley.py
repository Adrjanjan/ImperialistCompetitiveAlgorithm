from algorithm_evaluation.evaluation_helper import *
from algorithm_evaluation.test_functions import CostFunction


class F6_Ackley(CostFunction):

    def __init__(self, upper=100, lower=-100, dimension=1000,
                 o_vector="algorithm_evaluation/F6/resources/F6-xopt.txt",
                 p_vector="algorithm_evaluation/F6/resources/F6-p.txt",
                 r_25="algorithm_evaluation/F6/resources/F6-R25.txt",
                 r_50="algorithm_evaluation/F6/resources/F6-R50.txt",
                 r_100="algorithm_evaluation/F6/resources/F6-R100.txt",
                 s="algorithm_evaluation/F6/resources/F6-s.txt",
                 w="algorithm_evaluation/F6/resources/F6-w.txt"):
        super().__init__(self.ackley, upper, lower, dimension, o_vector, p_vector, r_25, r_50, r_100, s, w)
        self.s_size = 7

    @tf.function
    def ackley(self, vector):
        # 1
        z = vector - self.o_vector
        # 2
        result = 0
        i = 0
        start = 0
        result, i, start = self.calculate_partial_rotation(i, result, start, z)
        result, i, start = self.calculate_partial_rotation(i, result, start, z)
        result, i, start = self.calculate_partial_rotation(i, result, start, z)
        result, i, start = self.calculate_partial_rotation(i, result, start, z)
        result, i, start = self.calculate_partial_rotation(i, result, start, z)
        result, i, start = self.calculate_partial_rotation(i, result, start, z)
        result, i, start = self.calculate_partial_rotation(i, result, start, z)
        # 3
        return result + tf.reduce_sum(self.ackley_func(tf.gather(z, self.p_vector[start:]), start, self.dimension))

    @tf.function
    def calculate_partial_rotation(self, index, result, start, vector):
        result = result + tf.reduce_sum(
            self.w[index] * self.ackley_func(self.rotate_vector(vector, start, self.s[index])))
        return result, index + 1, start + self.s[index]

tf.config.run_functions_eagerly(True)

params = {
    "num_of_countries": [500, 1000],
    "num_of_imperialist": [5, 10],
    "max_iterations": [4000],
    "direct_assimilation": [0.2, 0.7],
    "avg_colonies_power": [0.1],
    "revolution_rate": [0.2, 0.5],
    "seed": [420]
}

result_path = "algorithm_evaluation/F6/results/"
iterations_results = gridsearch(F6_Ackley(), params, result_path)

print(iterations_results)

create_and_save_plots_to_file(iterations_results, result_path, "F6")
print(create_and_save_params_grid_as_latex_table(iterations_results, result_path, "F6"))

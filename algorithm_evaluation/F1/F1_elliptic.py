import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import ParameterGrid

from ica.ica import ICA
from test_functions import CostFunction

tf.config.run_functions_eagerly(True)


class F1_Elliptic(CostFunction):

    def __init__(self, upper=100, lower=-100, dimension=1000, o_vector="resources/F1-xopt.txt"):
        super().__init__(self.elliptic, upper, lower, dimension, o_vector)

    @tf.function
    def elliptic(self, vector):
        z = vector - self.o_vector
        return self.elliptic_func(z)


def create_and_save_plots_to_file(iterations_results, file_path):
    # TODO
    # min_vals = []
    # for index_params in iterations_results:
    #     index, params = index_params
    #     min_vals.append(params["lowest_cost_per_iteration"])
    # str(params["max_iterations"]),

    # x = metadata[0]
    # y = metadata[1]
    # plt.scatter(x, y, marker='v', color='r')
    # plt.title('Scatter Plot Example')
    # plt.show()
    # plt.savefig(file_path + 'foo.png', bbox_inches='tight')
    pass


def create_and_save_params_grid_as_latex_table(iterations_results, file_path):
    sep = " & "
    endline = r"\\\hline" + "\n"

    begin = r"\begin{tabular}{|c|c|c|c|c|c|c|}" + "\n" \
            + sep.join(["Nr. wykresu",
                        "Czas ewaluacji",
                        "Wartość osiągniętego minimum",
                        "Ostatnia iteracja",
                        "Maksymalna liczba iteracji",
                        "Liczba imperiów",
                        "Liczba kolonii",
                        r"Bezpośrednia asymilacja - $\beta$",
                        "Współczynnik rewolucji",
                        r"Współczynnik siły koloni $\xi$"]) \
            + endline

    middle = ""
    for index_params in iterations_results:
        index, params = index_params

        middle = middle + sep.join([
            str(index),
            str(params["evaluation_time"]),
            str(params["reached_minimum"]),
            str(params["final_iteration"]),
            str(params["max_iterations"]),
            str(params["empires_number"]),
            str(params["colonies_number"]),
            str(params["direct_assimilation"]),
            str(params["avg_colonies_power"]),
            str(params["revolution_rate"])
        ]) + endline

    end = r"\end{tabular}"
    return begin + middle + end


iterations_results = []

param_grid = {
    "num_of_countries": [100],
    "num_of_imperialist": [5],
    "max_iterations": [100, 500],
    "deviation_assimilation": [3.14 / 4.0],
    "direct_assimilation": [0.2],
    "avg_colonies_power": [0.1],
    "revolution_rate": [0.2],
    "close_empires_rating": [0.1],
    "seed": [420]
}

grid = ParameterGrid(param_grid)
function = F1_Elliptic()

for index, params in enumerate(grid):
    ica = ICA(cost_function=function,
              num_of_countries=params["num_of_countries"],
              num_of_imperialist=params["num_of_imperialist"],
              max_iterations=params["max_iterations"],
              deviation_assimilation=params["deviation_assimilation"],
              direct_assimilation=params["direct_assimilation"],
              avg_colonies_power=params["avg_colonies_power"],
              revolution_rate=params["revolution_rate"],
              close_empires_rating=params["close_empires_rating"],
              seed=params["seed"]
              )
    result = ica.eval()
    metadata = ica.get_evaluation_data()
    iterations_results.append((index, metadata))

print(iterations_results)

file_path = "results"
create_and_save_plots_to_file(iterations_results, file_path)
print(create_and_save_params_grid_as_latex_table(iterations_results, file_path))

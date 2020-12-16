import os

import tensorflow as tf
from ica.ica import ICA
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt


def create_and_save_plots_to_file(iterations_results, file_path, function_name):
    if not os.path.isdir(file_path):
        raise RuntimeError()
    for index_params in iterations_results:
        index, params = index_params
        y = params["lowest_cost_per_iteration"]
        x = range(len(y))
        plt.plot(x, y, label=str(index))
        plt.yscale("log")
    plt.legend()
    plt.savefig(file_path + function_name + '.png', bbox_inches='tight')
    plt.clf()


def create_and_save_params_grid_as_latex_table(iterations_results, file_path, function_name):
    sep = " & "
    endline = r"\\\hline" + "\n"

    begin = r"\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|}" + "\n" \
            + sep.join(["Nr. wykresu",
                        "Czas ewaluacji",
                        "Wartość osiągniętego minimum",
                        "Błąd osiagniętego minimum",
                        "Ostatnia iteracja",
                        "Maksymalna liczba iteracji",
                        "Liczba imperiów",
                        "Liczba kolonii",
                        r"Bezpośrednia asymilacja - $\beta$",
                        "Współczynnik rewolucji",
                        r"Współczynnik siły koloni $\xi$",
                        "Odległość minimum znalezionego od oczekiwanego"
                        ]) \
            + endline

    middle = ""
    for index_params in iterations_results:
        index, params = index_params

        middle = middle + sep.join([
            str(index),
            str(params["evaluation_time"]),
            str(params["reached_minimum"]),
            str(params["solution_error"]),
            str(params["final_iteration"]),
            str(params["max_iterations"]),
            str(params["empires_number"]),
            str(params["colonies_number"]),
            str(params["direct_assimilation"]),
            str(params["avg_colonies_power"]),
            str(params["revolution_rate"]),
            str(params["solution_distance"])
        ]) + endline

    end = r"\end{tabular}"
    result = begin + middle + end
    with open(file_path + function_name + ".txt", "w") as text_file:
        text_file.write(result)
    return result


def gridsearch(function, params, results_path):
    param_grid = ParameterGrid(params)
    iterations_results = []
    for index, params in enumerate(param_grid):
        print(params)
        ica = ICA(cost_function=function,
                  num_of_countries=params["num_of_countries"],
                  num_of_imperialist=params["num_of_imperialist"],
                  max_iterations=params["max_iterations"],
                  direct_assimilation=params["direct_assimilation"],
                  avg_colonies_power=params["avg_colonies_power"],
                  revolution_rate=params["revolution_rate"],
                  seed=params["seed"]
                  )
        try:
            ica.eval()
            metadata = ica.get_evaluation_data()
            iterations_results.append((index, metadata))
            print(metadata)
            save_metadata_per_iteration(index, metadata, results_path)
            save_result(index, ica.result.numpy(), results_path)
            create_and_save_plots_to_file(iterations_results, results_path, str(index))
        except tf.errors.InvalidArgumentError:
            continue

    return iterations_results


def save_metadata_per_iteration(index, metadata, file_path):
    with open(file_path + "Metadata" + str(index) + ".txt", "w") as text_file:
        text_file.write(",".join([str(index),
                                  str(metadata["evaluation_time"]),
                                  str(metadata["reached_minimum"]),
                                  str(metadata["solution_error"]),
                                  str(metadata["final_iteration"]),
                                  str(metadata["max_iterations"]),
                                  str(metadata["empires_number"]),
                                  str(metadata["colonies_number"]),
                                  str(metadata["direct_assimilation"]),
                                  str(metadata["avg_colonies_power"]),
                                  str(metadata["revolution_rate"]),
                                  str(metadata["solution_distance"])
                                  ])
                        )


def save_result(index, result, file_path):
    with open(file_path + "Result" + str(index) + ".txt", "w") as text_file:
        text_file.write(str(result))

from cec2013lsgo import cec2013
import timeit
import numpy as np
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import dtypes

benchmark = cec2013.Benchmark()
fn = benchmark.get_function(1)
benchmark.get_info(1)


class CostFunction:
    def __init__(self, function, upper, lower, dimension):
        self.function = function
        self.upper = upper
        self.lower = lower
        self.dimension = dimension


class ICA:

    def __init__(self, cost_function: CostFunction, num_of_countries=100, num_of_imperialist=4, max_iterations=10000,
                 deviation_assimilation=np.pi / 4, direct_assimilation=0.5, avg_colonies_power=0.1, revolution_rate=0.2,
                 log=False):
        self.cost_function = cost_function
        self.num_of_countries = num_of_countries
        self.num_of_imperialist = num_of_imperialist
        self.max_iterations = max_iterations
        self.deviation_assimilation = deviation_assimilation
        self.direct_assimilation = direct_assimilation
        self.avg_colonies_power = avg_colonies_power
        self.revolution_rate = revolution_rate
        self.is_loggable = log
        self.timeit = 0

    def eval(self):
        self.start_benchmark()

        # country is a row of matrix where each value is parameter of some cost function
        # colonies is matrix, where each row represents country, no duplicates
        # empires is matrix, where each row represents country, duplicates
        # corresponding row in matrices colonies and empires represents connection
        # between colony belonging to empire

        countries = self.initialize_countries()
        empires, colonies = self.create_empires(countries)

        for _ in range(self.max_iterations):
            empires, colonies = self.assimilation(empires, colonies)  # move to empire
            empires, colonies = self.revolution(empires, colonies)  # move from empire
            empires, colonies = self.swapStrongest(empires, colonies)  # case for inside ond between empires
            empires, colonies = self.competition(empires, colonies)
            empires, colonies = self.elimination(empires, colonies)
        # solution = colonies.gather_nd(0)
        self.finish_benchmark()
        return empires

    @tf.function
    def initialize_countries(self):
        return tf.random.uniform(shape=(self.num_of_countries, self.cost_function.dimension),
                                 minval=self.cost_function.lower,
                                 maxval=self.cost_function.upper,
                                 dtype=dtypes.float64,
                                 seed=0
                                 )

    # @tf.function
    def create_empires(self, countries):
        imperialists_indexes = tf.argsort(self.cost_function.function(countries))[-self.num_of_imperialist:]
        # wybranie imperiów
        # potrzebny będzie jeszcze wektor kosztów dla coloni i imperiów
        # obliczenie siły


        # tf.argsortpi
        # indices = result^
        # pass
        return countries, countries

    @tf.function
    def assimilation(self, empires, colonies):
        return empires, colonies

    @tf.function
    def swap_countries(self, first_index, second_index):
        pass

    @tf.function
    def revolution(self, empires, colonies):
        return empires, colonies

    @tf.function
    def swapStrongest(self, empires, colonies):
        return empires, colonies

    @tf.function
    def competition(self, empires, colonies):
        return empires, colonies

    @tf.function
    def elimination(self, empires, colonies):
        return empires, colonies

    def start_benchmark(self):
        self.timeit = timeit.default_timer()

    def finish_benchmark(self):
        if self.is_loggable:
            print(timeit.default_timer() - self.timeit)


# rzutowanie na prymityw obiektu Imperium


ica = ICA(CostFunction([], -10, 10, 1000))
result = ica.eval()
print(result)

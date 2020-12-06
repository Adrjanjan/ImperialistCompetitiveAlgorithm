import tensorflow as tf
from test_functions import CostFunction
from ica.ica import ICA


class F1_Elliptic(CostFunction):

    def __init__(self, upper=100, lower=-100, dimension=1000):
        function = self.elliptic
        o_vector = "resources/F1-xopt.txt"
        super().__init__(function, upper, lower, dimension, o_vector)

    @tf.function
    def elliptic(self, vector):
        z = self.transform_asy(self.M1(vector - self.o_vector))
        return self.elliptic_func(z)


# tutaj gridsearch
params = {"p1": [1, 2, 3, 4], "p2": [1, 2, 3, 4]}

function = F1_Elliptic()
ica = ICA(function, )

# pamiętaj o zbieraniu parametrów obliczeń do jakiegoś wektora
# twórz odpowiednie wykresy i zapisuj je do pliku
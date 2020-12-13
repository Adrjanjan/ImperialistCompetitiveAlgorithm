from ica.ica import ICA
from algorithm_evaluation.test_functions import CostFunction
import tensorflow as tf

# tf.config.run_functions_eagerly(False)
tf.config.run_functions_eagerly(True)
tf.debugging.set_log_device_placement(True)

square = CostFunction(lambda x: tf.reduce_sum(tf.square(x)), 100.0, -100.0, 1000)
ica = ICA(square, 1000, 5, 100, log=True, revolution_rate=0.01)
result = ica.eval()

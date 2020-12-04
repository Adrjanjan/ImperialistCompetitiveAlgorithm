import tensorflow as tf
from ica import helpers
import constants


@tf.function
def revolution(colonies, num_of_colonies, dimension, lower, upper, revolution_rate):
    to_revolve = helpers.calculate_distribution((num_of_colonies, 1), constants.zero, constants.one)
    possible_new_positions = helpers.calculate_distribution((num_of_colonies, dimension), lower, upper)
    chance_to_revolve = helpers.broadcast_boolean_mask(tf.less(to_revolve, revolution_rate),
                                                       (num_of_colonies, dimension))

    new_colonies = tf.where(chance_to_revolve, possible_new_positions, colonies)
    return new_colonies

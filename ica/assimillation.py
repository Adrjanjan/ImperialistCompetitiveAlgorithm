import tensorflow as tf
from ica import helpers
import constants


@tf.function
def normalize_vectors(vector):
    norm = tf.sqrt(tf.reduce_sum(tf.square(vector), keepdims=True))
    return vector / norm


@tf.function
def assimilation(colonies, empires, num_of_colonies, dimension, direct_assimilation, lower, upper):
    # TODO implement different variants, test them
    difference = tf.subtract(empires, colonies)
    uniform = helpers.calculate_distribution(
        (num_of_colonies, dimension),
        constants.zero,
        tf.multiply(tf.abs(difference), direct_assimilation)
    )

    new_colonies = tf.clip_by_value(
        tf.add(colonies, tf.multiply(uniform, normalize_vectors(difference))),
        lower,
        upper
    )
    return new_colonies


@tf.function
def assimilation_v2(colonies, empires, num_of_colonies, dimension, direct_assimilation, lower, upper):
    # TODO implement different variants, test them
    difference = tf.subtract(empires, colonies)
    beta = helpers.calculate_distribution((1,), constants.zero, direct_assimilation)
    new_colonies = tf.clip_by_value(
        tf.add(colonies, tf.multiply(beta, difference)),
        lower,
        upper
    )

    return new_colonies

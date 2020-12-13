import tensorflow as tf
from ica import helpers
import constants


@tf.function
def initialize_countries(num_of_countries, dimension, lower, upper):
    return helpers.calculate_distribution((num_of_countries, dimension), lower, upper)


@tf.function
def add_one_to_top_empire(num_of_colonies_per_empire, num_of_colonies):
    more = tf.greater(tf.reduce_sum(num_of_colonies_per_empire), num_of_colonies)
    less = tf.less(tf.reduce_sum(num_of_colonies_per_empire), num_of_colonies)
    return tf.cond(
        less,
        lambda: tf.concat(
            [tf.expand_dims(tf.add(num_of_colonies_per_empire[0], 1), 0), num_of_colonies_per_empire[1:]], 0),
        lambda: tf.cond(
            more,
            lambda: tf.concat(
                [tf.expand_dims(tf.subtract(num_of_colonies_per_empire[0], 1), 0), num_of_colonies_per_empire[1:]], 0),
            lambda: num_of_colonies_per_empire
        )
    )


@tf.function
def calculate_number_of_colonies(top_powers, num_of_colonies):
    max_cost = tf.reduce_max(top_powers[:])
    x1 = tf.constant(1.3, dtype=tf.float64)
    x2 = tf.constant(0.7, dtype=tf.float64)
    normalised_cost = tf.where(max_cost > 0, tf.subtract(tf.multiply(x1, max_cost), top_powers),
                               tf.subtract(tf.multiply(x2, max_cost), top_powers))
    num_of_colonies_per_empire = tf.round(
        tf.multiply(tf.truediv(normalised_cost, tf.reduce_sum(normalised_cost)), num_of_colonies))

    return tf.cast(add_one_to_top_empire(num_of_colonies_per_empire, num_of_colonies), tf.int32)


@tf.function
def condition(index, *unused):
    return tf.greater_equal(index, constants.int_zero)


@tf.function
def construct_empires(index, number_of_colonies, top_indexes, empires, empires_numbers, insert_index, countries,
                      dimension):
    current_num_of_colonies = tf.gather(number_of_colonies, index)
    current_empire_index = tf.gather(top_indexes, index)
    new_empire_matrix = tf.reshape(
        tf.tile(tf.gather(countries, current_empire_index), [current_num_of_colonies]),
        [current_num_of_colonies, dimension])
    new_empire_indexes = tf.reshape(tf.tile([index], [current_num_of_colonies]),
                                    [current_num_of_colonies, 1])

    concatenated_empires = helpers.insert_in_tensor_from_index(empires, new_empire_matrix,
                                                               insert_index, current_num_of_colonies)
    concatenated_empires_numbers = helpers.insert_in_tensor_from_index(empires_numbers, new_empire_indexes,
                                                                       insert_index, current_num_of_colonies)

    new_index = tf.subtract(index, 1)
    return new_index, number_of_colonies, top_indexes, concatenated_empires, concatenated_empires_numbers, \
           tf.add(insert_index, tf.size(new_empire_indexes)), countries, dimension


@tf.function
def create_empires(countries, dimension, num_of_colonies, num_of_imperialist, cost_function):
    power = helpers.evaluate_countries_power(countries, cost_function)
    # (-1)*power is important to get lowest power first - lower cost means bigger power
    top_powers, top_indexes = tf.nn.top_k(-power, num_of_imperialist)
    top_powers = tf.negative(top_powers)
    num_of_colonies_per_empire = calculate_number_of_colonies(top_powers, num_of_colonies)
    temp_empires = tf.zeros((num_of_colonies, dimension), dtype=tf.float64)
    temp_empires_numbers = tf.zeros((num_of_colonies, 1), tf.int32)
    iteration_initialization_params = (tf.constant(num_of_imperialist - 1),
                                       num_of_colonies_per_empire,
                                       top_indexes,
                                       temp_empires,
                                       temp_empires_numbers,
                                       tf.constant(0),
                                       countries,
                                       dimension
                                       )

    _, _, _, empires, empires_numbers, _, _, _ = tf.while_loop(condition, construct_empires,
                                                               iteration_initialization_params,
                                                               shape_invariants=(
                                                                   tf.TensorShape(None),
                                                                   tf.TensorShape([None]),
                                                                   tf.TensorShape([None]),
                                                                   tf.TensorShape([None, dimension]),
                                                                   tf.TensorShape([None, 1]),
                                                                   tf.TensorShape(None),
                                                                   tf.TensorShape([None, dimension]),
                                                                   tf.TensorShape(None),
                                                               ))
    colonies = helpers.delete_rows(countries, top_indexes, num_of_colonies + num_of_imperialist)
    return empires, colonies, tf.squeeze(empires_numbers)

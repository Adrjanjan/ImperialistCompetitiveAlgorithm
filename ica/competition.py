import tensorflow as tf
from ica import helpers
import constants


@tf.function
def select_worst_colony_index(empire_index, empires_indexes, num_of_colonies, dimension, colonies_power):
    colonies_power_with_index = tf.concat([tf.expand_dims(colonies_power, axis=1),
                                           tf.cast(tf.range(num_of_colonies)[:, None], dtype=tf.float64)],
                                          axis=1)

    current_empire_colonies_power_with_index = helpers.get_all_by_empire_number(empire_index, colonies_power_with_index,
                                                                                empires_indexes)
    colonies_power = tf.reshape(current_empire_colonies_power_with_index[:, :-1],
                                (current_empire_colonies_power_with_index.shape[0],))

    _, current_worst_index = tf.math.top_k(colonies_power)
    current_empire_colonies_index = tf.cast(current_empire_colonies_power_with_index[:, -1], tf.int32)
    worst_colony_index = tf.gather(current_empire_colonies_index, current_worst_index)
    return worst_colony_index


@tf.function
def select_top_empire(empires, empires_numbers, total_power, num_of_colonies):
    distribution = helpers.calculate_distribution((num_of_colonies,), constants.zero, total_power)
    _, top_empire_index = tf.nn.top_k(tf.subtract(total_power, distribution))
    top_empire = tf.gather(empires, top_empire_index)
    top_empire_number = tf.gather(empires_numbers, top_empire_index)
    return top_empire, top_empire_number


@tf.function
def condition_evaluate_total_power(index, *unused):
    return tf.greater_equal(index, constants.int_zero)


@tf.function
def evaluate_empire_power(index, total_power, empires_power, colonies_power, avg_colonies_power, empire_numbers,
                          empires_numbers):
    empire_number = tf.gather(empire_numbers, index)
    empire_colonies_power_sum = tf.reduce_sum(
        helpers.get_all_by_empire_number(empire_number, colonies_power, empires_numbers))
    empire_power = helpers.get_first_by_empire_number(empire_number, empires_power, empires_numbers)
    empire_total_power = tf.add(empire_power, tf.multiply(avg_colonies_power, empire_colonies_power_sum))
    new_total_power = helpers.broadcastable_where(tf.equal(empires_numbers, empire_number),
                                                  tf.reshape(empire_total_power, [-1]),
                                                  total_power
                                                  )
    return tf.subtract(index, 1), new_total_power, empires_power, colonies_power, avg_colonies_power, empire_numbers, \
           empires_numbers


@tf.function
def calculate_empires_total_power(empires_numbers, avg_colonies_power,
                                  num_of_colonies, empires_power, colonies_power):
    empire_numbers, _ = tf.unique(empires_numbers)

    initial_params = (tf.subtract(tf.size(empire_numbers), 1), tf.ones([num_of_colonies, 1], dtype=tf.float64),
                      empires_power, colonies_power, avg_colonies_power, empire_numbers, empires_numbers)
    _, total_cost, _, _, _, _, _ = tf.while_loop(condition_evaluate_total_power, evaluate_empire_power, initial_params)

    total_cost = tf.squeeze(total_cost)
    normalised_total_cost = tf.subtract(tf.reduce_max(total_cost), total_cost)
    total_power = tf.abs(tf.truediv(normalised_total_cost, tf.reduce_sum(normalised_total_cost)))
    return total_power


@tf.function
def check_number_of_imperialist(empires_numbers, worst_empire_number, num_of_imperialist):
    return tf.where(
        tf.equal(0, tf.reduce_sum(tf.cast(tf.equal(empires_numbers, worst_empire_number), tf.int32))),
        tf.subtract(num_of_imperialist, 1),
        num_of_imperialist
    )


@tf.function
def competition(empires, empires_numbers, num_of_colonies, num_of_imperialist, avg_colonies_power, dimension,
                empires_power, colonies_power):
    # calculate total power
    total_power = calculate_empires_total_power(empires_numbers, avg_colonies_power, num_of_colonies,
                                                empires_power, colonies_power)

    # select top and worst empire
    top_empire, top_empire_number = select_top_empire(empires, empires_numbers, total_power, num_of_colonies)
    _, worst_empire_index = tf.math.top_k(-total_power)
    worst_empire_number = tf.gather(empires_numbers, worst_empire_index)

    # select worst colony from worst empire
    worst_colony_index = select_worst_colony_index(worst_empire_number, empires_numbers, num_of_colonies,
                                                   dimension, colonies_power)

    # set new empire as empire of worst colony
    new_empires_numbers = tf.squeeze(
        helpers.replace_row_in_matrix(tf.expand_dims(empires_numbers, 1), top_empire_number, worst_colony_index)
    )
    new_empires = helpers.replace_row_in_matrix(empires, top_empire, worst_colony_index)
    top_empire_power = helpers.get_first_by_empire_number(top_empire_number, empires_power, empires_numbers)
    new_empires_power = tf.squeeze(
        helpers.replace_row_in_matrix(tf.expand_dims(empires_power, 1), top_empire_power, worst_colony_index)
    )

    # reduce num_of_imperialists if it was worst empire last presence
    new_num_of_imperialist = check_number_of_imperialist(new_empires_numbers, worst_empire_number, num_of_imperialist)

    return new_empires, new_empires_numbers, new_empires_power, colonies_power, new_num_of_imperialist

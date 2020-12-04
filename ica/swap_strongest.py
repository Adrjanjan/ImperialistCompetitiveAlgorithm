import tensorflow as tf
from ica import helpers
import constants


@tf.function
def swap(current_empire_number, colonies, colonies_indexes, empires, empires_numbers,
         colonies_power_with_index):
    current_empire = helpers.get_first_by_empire_number(current_empire_number, empires, empires_numbers)
    current_colonies_power_with_index = helpers.get_all_by_empire_number(current_empire_number,
                                                                         colonies_power_with_index,
                                                                         empires_numbers)
    current_colonies_power = tf.reshape(tf.squeeze(current_colonies_power_with_index[:, :-1]), [-1])
    current_colonies_index = tf.cast(current_colonies_power_with_index[:, -1], tf.int32)

    _, best_colony_current_index = tf.nn.top_k(-current_colonies_power)
    best_colony_real_index = tf.gather(current_colonies_index, best_colony_current_index)
    best_colony = tf.gather(colonies, best_colony_real_index)

    new_e = helpers.broadcastable_where(tf.equal(empires_numbers, current_empire_number), best_colony, empires)
    new_c = helpers.broadcastable_where(tf.equal(colonies_indexes, best_colony_real_index), current_empire,
                                        colonies)
    return new_e, new_c


@tf.function
def swap_strongest(empires, colonies, empires_numbers, num_of_colonies, cost_function):
    @tf.function
    def condition(iterator, empires_numbers_to_check, *unused):
        return tf.less(iterator, tf.size(empires_numbers_to_check))

    @tf.function
    def body(iterator, empires_numbers_to_check, empires, colonies, empires_numbers):
        current_empire_number = tf.gather(empires_numbers_to_check, iterator)
        current_empire_power = helpers.get_first_by_empire_number(current_empire_number, empires_power,
                                                                  empires_numbers)
        current_colonies_power = helpers.get_all_by_empire_number(current_empire_number, colonies_power,
                                                                  empires_numbers)
        best_colony_power = tf.reduce_min(current_colonies_power)
        empire_not_exists = helpers.is_empty(current_empire_power)
        swapped_empires, swapped_colonies = tf.cond(
            empire_not_exists,
            lambda: (empires, colonies),
            lambda: tf.cond(current_empire_power > best_colony_power,
                            lambda: swap(current_empire_number, colonies, colonies_indexes, empires,
                                         empires_numbers, colonies_power_with_index),
                            lambda: (empires, colonies)
                            )
        )

        return tf.add(1, iterator), empires_numbers_to_check, swapped_empires, swapped_colonies, empires_numbers

    colonies_power = helpers.evaluate_countries_power(colonies, cost_function)
    colonies_indexes = helpers.create_index_column(colonies_power)
    colonies_power_with_index = helpers.concatenate_tensor_with_index(colonies_power, colonies_indexes, num_of_colonies)

    empires_power = helpers.evaluate_countries_power(empires, cost_function)
    empires_numbers_to_check, _ = tf.unique(empires_numbers)
    swap_initial_params = (constants.int_zero, empires_numbers_to_check, empires, colonies, empires_numbers)

    _, _, new_empires, new_colonies, _ = tf.while_loop(condition, body, swap_initial_params)
    return new_empires, new_colonies

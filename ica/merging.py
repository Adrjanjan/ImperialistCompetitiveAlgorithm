import tensorflow as tf
from ica import helpers
import constants


@tf.function
def are_two_empires_close(first_empire, second_empire, close_empires_rating):
    return tf.less(tf.norm(tf.subtract(first_empire, second_empire)), close_empires_rating)


@tf.function
def merge_empires(outer_number, inner_number, outer_empire, inner_empire, empires, empires_numbers,
                  close_empires_rating):
    @tf.function
    def merge():
        condition = tf.equal(empires_numbers, inner_number)
        new_empires = helpers.broadcastable_where(condition, outer_empire, empires)
        new_empires_numbers = tf.squeeze(
            helpers.broadcastable_where(condition, outer_number, tf.expand_dims(empires_numbers, 1)))
        return new_empires, new_empires_numbers

    return tf.cond(
        are_two_empires_close(outer_empire, inner_empire, close_empires_rating),
        merge,
        lambda: (empires, empires_numbers)
    )


@tf.function
def merging_of_similar(empires, empires_numbers, num_of_imperialist, close_empires_rating):
    @tf.function
    def outer_condition(_, checked_count, *unused):
        return tf.less(checked_count, num_of_imperialist)

    @tf.function
    def outer_loop(outer_empire_number, checked_count, empires, empires_numbers):
        @tf.function
        def inner_condition(inner_index, *unused):
            return tf.less(inner_index, num_of_imperialist)

        @tf.function
        def inner_loop(inner_empire_number, outer_empire_number, outer_empire, empires, empires_numbers):
            inner_empire = helpers.get_first_by_empire_number(inner_empire_number, empires, empires_numbers)
            new_empires, new_empires_numbers = tf.cond(
                helpers.is_empty(inner_empire),
                lambda: (empires, empires_numbers),
                lambda: merge_empires(outer_empire_number,
                                      inner_empire_number,
                                      outer_empire,
                                      inner_empire,
                                      empires,
                                      empires_numbers,
                                      close_empires_rating)
            )
            return tf.add(1, inner_empire_number), outer_empire_number, \
                   outer_empire, new_empires, new_empires_numbers

        outer_empire = helpers.get_first_by_empire_number(outer_empire_number, empires, empires_numbers)
        inner_initial_params = (tf.add(1, outer_empire_number),
                                outer_empire_number,
                                outer_empire,
                                empires,
                                empires_numbers)
        empire_with_current_number_not_exists = helpers.is_empty(outer_empire)
        _, _, _, new_empires, new_empires_numbers = tf.cond(
            empire_with_current_number_not_exists,
            lambda: inner_initial_params,
            lambda: tf.while_loop(inner_condition, inner_loop, inner_initial_params)
        )

        new_checked_count = tf.cond(
            empire_with_current_number_not_exists,
            lambda: checked_count,
            lambda: tf.add(checked_count, 1),
        )
        return tf.add(1, outer_empire_number), new_checked_count, new_empires, new_empires_numbers

    outer_initial_params = (constants.int_zero, constants.int_zero, empires, empires_numbers)
    _, _, new_empires, new_empires_numbers = tf.while_loop(outer_condition, outer_loop, outer_initial_params)
    return new_empires, new_empires_numbers

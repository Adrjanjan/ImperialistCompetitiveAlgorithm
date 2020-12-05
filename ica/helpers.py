import tensorflow as tf
import constants


@tf.function
def calculate_distribution(shape, minval, maxval, seed=None):
    return tf.random.uniform(shape=shape,
                             minval=minval,
                             maxval=maxval,
                             dtype=tf.float64,
                             seed=seed
                             )


@tf.function
def insert_in_tensor_from_index(tensor, to_insert, index, insert_size):
    end = tensor[(index + insert_size):]
    result = tf.cond(tf.equal(index, constants.int_zero),
                     lambda: tf.concat([to_insert, end], 0),
                     lambda: tf.concat([tensor[0:index], to_insert, end], 0))
    return result


@tf.function
def delete_rows(tensor, index, length):
    return tf.boolean_mask(tensor, tf.subtract(1.0, tf.reduce_sum(tf.one_hot(index, length), 0)))


@tf.function
def evaluate_countries_power(countries, cost_function):
    return tf.squeeze(tf.transpose(tf.map_fn(cost_function, countries)))


@tf.function
def broadcast_boolean_mask(bool_vector, shape):
    return tf.reshape(tf.repeat(bool_vector, shape[1]), shape)


@tf.function
def create_index_column(tensor):
    return tf.cast(tf.range(tensor.shape[0])[:, None], tf.int32)


@tf.function
def concatenate_tensor_with_index(tensor, index, tensor_length):
    return tf.concat(
        [tf.reshape(tensor, [tensor_length, 1]),
         tf.cast(index, dtype=tf.float64)],
        axis=1
    )


@tf.function
def is_empty(tensor):
    return tf.equal(tf.size(tensor), constants.int_zero)


@tf.function
def get_all_by_empire_number(empire_number, tensor, empires_numbers):
    return tf.cond(
        is_empty(tensor),
        lambda: tensor,
        lambda: tf.boolean_mask(tf.squeeze(tensor), tf.equal(empire_number, empires_numbers))
    )


@tf.function
def get_first_by_empire_number(empire_number, tensor, empires_numbers):
    result = tf.boolean_mask(tf.squeeze(tensor), tf.equal(empire_number, empires_numbers))
    return tf.cond(
        is_empty(tensor),
        lambda: tensor,
        lambda: tf.cond(is_empty(result),
                        lambda: result,
                        lambda: tf.squeeze(tf.gather(result, 0))
                        )
    )


@tf.function
def broadcast_boolean_mask(bool_vector, shape):
    return tf.broadcast_to(bool_vector, shape)


@tf.function
def broadcastable_where(bool_mask, vector, tensor):
    return tf.where(broadcast_boolean_mask(bool_mask, tensor.shape),
                    tf.broadcast_to(vector, tensor.get_shape()),
                    tensor)


@tf.function
def replace_row_in_matrix(tensor, row, index):
    mask = tf.equal(tf.range(tensor.get_shape()[0]), index)
    return broadcastable_where(mask, row, tensor)

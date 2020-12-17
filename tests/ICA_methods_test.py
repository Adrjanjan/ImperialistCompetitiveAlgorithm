import unittest
import tensorflow as tf
from ica.ica import ICA
from algorithm_evaluation.test_functions import CostFunction
from numpy.testing import assert_equal

# tf.config.run_functions_eagerly(False)
tf.config.run_functions_eagerly(True)


class MyTestCase(unittest.TestCase):

    def test_initialize_countries(self):
        # given
        test_function = CostFunction(lambda x: tf.reduce_sum(tf.square(x), 1), 10.0, -10.0, 2)
        ica = ICA(test_function, 10, 3, 1, log=True, seed=42)
        # when
        countries = ica.initialize_countries()
        # then
        self.assertTrue(tf.reduce_all(tf.greater(countries, -10.0).numpy()))
        self.assertTrue(tf.reduce_all(tf.less(countries, 10.0).numpy()))

    def test_create_empires(self):
        # given
        test_function = CostFunction(lambda x: tf.reduce_sum(tf.square(x), 1), 10.0, -10.0, 2)
        ica = ICA(test_function, 10, 3, 1, log=True, seed=42)
        countries = ica.initialize_countries()
        # when
        empires, colonies, empires_numbers = ica.create_empires(countries)
        # then
        unique, _ = tf.unique(empires_numbers)
        self.assertEqual(3, tf.size(unique))
        self.assertEqual([7, 2], colonies.shape)
        self.assertEqual([7, 2], empires.shape)

    def test_assimilation(self):
        # given
        test_function = CostFunction(lambda x: tf.reduce_sum(tf.square(x), 1), 10.0, -10.0, 2)
        ica = ICA(test_function, 5, 2, 1, log=True, seed=42)
        empires = tf.constant([[3., 3., ],
                               [3., 3., ],
                               [2., 4., ]
                               ], tf.float64)
        colonies = tf.constant([[1., 9., ],
                                [5., 3., ],
                                [1., 7., ]
                                ], tf.float64)
        empires_numbers = tf.constant([1, 1, 0])
        # when
        new_colonies = ica.assimilation(empires, colonies)
        # then
        distance_before = tf.map_fn(tf.norm, tf.subtract(empires, colonies))
        distance_after = tf.map_fn(tf.norm, tf.subtract(empires, new_colonies))

        self.assertTrue(tf.reduce_all(tf.less(distance_after, distance_before)).numpy())

    def test_revolution(self):
        pass

    def test_swap_strongest_one_colony_to_swap(self):
        # given
        test_function = CostFunction(lambda x: tf.reduce_sum(tf.square(x), 1), 10.0, -10.0, 2)
        ica = ICA(test_function, 5, 2, 1, log=True)
        empires = tf.constant([[3., 3., ],
                               [3., 3., ],
                               [2., 4., ]
                               ], tf.float64)
        colonies = tf.constant([[1., 1., ],
                                [5., 3., ],
                                [1., 7., ]
                                ], tf.float64)
        empires_numbers = tf.constant([1, 1, 0])
        # when
        new_colonies, new_empires, _, _ = ica.swap_strongest(colonies=colonies,
                                                             empires=empires,
                                                             empires_numbers=empires_numbers
                                                             )
        # then
        assert_equal(new_empires.numpy(), [[1., 1., ],
                                           [1., 1., ],
                                           [2., 4., ]])
        assert_equal(new_colonies.numpy(), [[3., 3., ],
                                            [5., 3., ],
                                            [1., 7., ]])

    def test_swap_strongest_two_better_colonies_swap_first(self):
        # given
        test_function = CostFunction(lambda x: tf.reduce_sum(tf.square(x), 1), 10.0, -10.0, 2)
        ica = ICA(test_function, 5, 2, 1, log=True, seed=42)
        empires = tf.constant([[3., 3., ],
                               [3., 3., ],
                               [2., 4., ]
                               ], tf.float64)
        colonies = tf.constant([[1., 1., ],
                                [1., 1., ],
                                [1., 7., ]
                                ], tf.float64)
        empires_numbers = tf.constant([1, 1, 0])
        # when
        new_empires, new_colonies = ica.swap_strongest(empires, colonies, empires_numbers)
        # then
        assert_equal(new_empires.numpy(), [[1., 1., ],
                                           [1., 1., ],
                                           [2., 4., ]])
        assert_equal(new_colonies.numpy(), [[3., 3., ],
                                            [1., 1., ],
                                            [1., 7., ]])

    def test_swap_strongest_none_to_swap(self):
        # given
        test_function = CostFunction(lambda x: tf.reduce_sum(tf.square(x), 1), 10.0, -10.0, 2)
        ica = ICA(test_function, 5, 2, 1, log=True, seed=42)
        empires = tf.constant([[3., 3., ],
                               [3., 3., ],
                               [2., 4., ]
                               ], tf.float64)
        colonies = tf.constant([[4., 4., ],
                                [5., 5., ],
                                [1., 7., ]
                                ], tf.float64)
        empires_numbers = tf.constant([1, 1, 0])
        # when
        new_empires, new_colonies = ica.swap_strongest(empires, colonies, empires_numbers)
        # then
        assert_equal(new_empires.numpy(), [[3., 3., ],
                                           [3., 3., ],
                                           [2., 4., ]])
        assert_equal(new_colonies.numpy(), [[4., 4., ],
                                            [5., 5., ],
                                            [1., 7., ]])

    def test_competition_one_is_better(self):
        # given
        test_function = CostFunction(lambda x: tf.reduce_sum(tf.square(x), 1), 10.0, -10.0, 2)
        ica = ICA(test_function, 6, 2, 1, log=True, seed=42)
        empires = tf.constant([[3., 3., ],
                               [3., 3., ],
                               [2., 4., ],
                               [2., 4., ]
                               ], tf.float64)
        colonies = tf.constant([[4., 4., ],
                                [5., 5., ],
                                [1., 7., ],
                                [9., 9., ]
                                ], tf.float64)
        empires_numbers = tf.constant([1, 1, 0, 0])
        # when
        new_empires, new_empire_numbers, _ = ica.competition(empires, colonies, empires_numbers)
        # then
        assert_equal(new_empire_numbers.numpy(), [1, 1, 0, 1])
        assert_equal(new_empires.numpy(), [[3., 3., ],
                                           [3., 3., ],
                                           [2., 4., ],
                                           [3., 3., ]])

    def test_competition_only_one_empire_no_changes(self):
        # given
        test_function = CostFunction(lambda x: tf.reduce_sum(tf.square(x), 1), 10.0, -10.0, 2)
        ica = ICA(test_function, 6, 2, 1, log=True, seed=42)
        empires = tf.constant([[3., 3., ],
                               [3., 3., ],
                               [3., 3., ],
                               [3., 3., ]
                               ], tf.float64)
        colonies = tf.constant([[4., 4., ],
                                [5., 5., ],
                                [1., 7., ],
                                [9., 9., ]
                                ], tf.float64)
        empires_numbers = tf.constant([1, 1, 1, 1])
        # when
        new_empires, new_empire_numbers, _ = ica.competition(empires, colonies, empires_numbers)
        # then
        assert_equal(new_empire_numbers.numpy(), empires_numbers.numpy())
        assert_equal(new_empires.numpy(), empires.numpy())

    def test_merging_of_similar(self):
        pass


if __name__ == '__main__':
    unittest.main()

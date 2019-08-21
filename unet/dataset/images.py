import pathlib

import tensorflow as tf


def load_image(path, num_channels):
    """
    Given a path to an image, loads the image data and sets the shape of the result tensor.
    :param path: Path to the image.
    :param num_channels: Number of channels expected in the image.
    :return: The loaded image.
    """
    image = tf.io.decode_image(tf.io.read_file(path), channels=num_channels)
    image.set_shape((None, None, num_channels))
    return image


def load_images(paths, num_channels):
    return tf.nest.map_structure(load_image, paths, num_channels)

import pathlib
import tensorflow as tf
from collections import namedtuple


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
    """
    Loads the images given by `paths` with the number of channels given by channels.
    :param paths: A (nested) structure of string tensors describing image paths.
    :param num_channels: A (nested) structure of ints describing the numbers of channels in each image.
    TODO allow num_channels to be of int type and broadcast!
    :return: A (nested) structure of (tf.uint8) image tensors.
    """
    return tf.nest.map_structure(load_image, paths, num_channels)


def filename_transformer(in_pattern: str, out_pattern: str):
    """
    Returns a function that calculates transformations on a file name. To that end, to python format strings are
    specified as patterns. One is used to parse the original filename, and the second to construct the derived filename.

    Example:
    >>> f = filename_transformer("{:d}.png", "{:d}.jpg")
    >>> f("5.png")
    5.jpg

    :param in_pattern: The format pattern based on which the input file name is parsed.
    :param out_pattern: The format string for the output filename.
    :return: A function that takes a file name and returns the transformed file name.
    """
    from parse import parse

    def transform(file_name):
        num = parse(in_pattern, file_name)
        if num is None:
            raise ValueError("Input file name '%s' does not match pattern '%s'" % (file_name, in_pattern))
        return out_pattern.format(*num.fixed, **num.named)
    return transform


def make_image_dataset(image_paths, channels, shuffle=True, max_buffer: int = 1000, cache: bool = None):
    """
    Given a structure of images, makes a `tf.data.Dataset` out of these.
    If there are less than `max_buffer` elements, than the entire dataset will be cached in memory,
    and if `shuffle == True`, the shuffle buffer will be as big as the entire dataset.
    :param image_paths: A (nested) structure of paths to images, or a dataset of paths to images.
    :param channels: A (nested) structure of integers with the number of channels in the images.
    :param shuffle: Whether to shuffle the dataset.
    :param max_buffer: Maximum number of elements to use in a buffer.
    :param cache: Whether to cache the dataset in memory. If not explicitly set to True or False, caching is determined
    by the number of elements in the dataset.
    :return: A dataset of loaded images (uint8).
    """
    if not isinstance(image_paths, tf.data.Dataset):
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    else:
        dataset = image_paths

    # load_images expects two arguments, but dataset.map unpacks tuples, so we need to step in here
    # TODO does this work for dicts and single element images?
    def load_images_mappable(*args):
        return load_images(args, channels)
    dataset = dataset.map(load_images_mappable, num_parallel_calls=4)

    if cache is True or (tf.data.experimental.cardinality(dataset) < max_buffer and cache is None):
        dataset = dataset.cache()

    if shuffle:
        # cardinality should always be known, as from_tensor_slices produces a deterministic number of elements.
        return dataset.shuffle(tf.minimum(tf.data.experimental.cardinality(dataset), max_buffer))
    else:
        return dataset


# TODO derive from namedtuple so that data cannot be changed anymore
class ImageSetSpec:
    def __init__(self, directory, pattern, channels):
        self.directory = directory
        self.pattern = pattern
        self.channels = channels
        self._image_list = None

    @property
    def image_files(self):
        """
        :return: A list with paths to all the image files.
        """
        if self._image_list is None:
            self._image_list = sorted(self.directory.glob(self.pattern))

        return self._image_list

    @property
    def as_dict(self):
        return {"directory": self.directory, "pattern": self.pattern, "channels": self.channels}

    @staticmethod
    def from_dict(data):
        return ImageSetSpec(**data)


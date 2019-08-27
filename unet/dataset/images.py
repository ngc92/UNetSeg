import pathlib
import tensorflow as tf
import logging
import json
from collections import namedtuple

_logger = logging.getLogger(__name__)


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
        if isinstance(image_paths, list):
            raise ValueError("Please supply the images as a tuple containing lists of image paths, "
                             "not as a list of tuples.")
        dataset = tf.data.Dataset.from_tensor_slices(tf.nest.map_structure(str, image_paths))
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
    def __init__(self, directory, pattern, channels, source_pattern=None):
        self.directory = pathlib.Path(directory)
        self.pattern = pattern
        self.channels = channels
        self.source_pattern = source_pattern
        if source_pattern is not None:
            self.transform = filename_transformer(source_pattern, pattern)
        else:
            self.transform = lambda x: x
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
        return {"directory": self.directory, "pattern": self.pattern, "channels": self.channels,
                "source_pattern": self.source_pattern}

    def get_matching(self, source_image: pathlib.Path):
        source_name = source_image.name
        target_name = self.transform(source_name)
        return self.directory / target_name

    @staticmethod
    def from_dict(data, root=None):
        if root is not None:
            data["directory"] = root / pathlib.Path(data["directory"])
        return ImageSetSpec(**data)

    def __repr__(self):
        return "ImageSetSpec(%r, %r, %r, %r)" % (str(self.directory), self.pattern, self.channels, self.source_pattern)


def pic2pic_matching_files(source_spec: ImageSetSpec, target_spec: ImageSetSpec):
    """
    Given two image specs, returns matching files as a tuple of lists. The struct of array approach was chosen over
    an array of struct approaches, so that it can be used to create a `tf.data.Dataset` of tuples instead of a dataset
    containing a single, multi-element tensor.
    :param source_spec: Image spec for the source images.
    :param target_spec: Image spec for the target images. Either `pattern` and `source_pattern` need to be set, or
    these images are expected to have the same file name as the source images.
    :return: A pair of a lists.
    """
    sources = []
    targets = []

    source_images = source_spec.image_files
    if len(source_images) == 0:
        _logger.warning("Could not find any images for %s", source_spec)
    for source in source_images:
        seg = target_spec.get_matching(source)
        if not seg.exists():
            _logger.warning("Could not find matching file '%s' for source '%s'", source, seg)
            continue
        sources.append(source)
        targets.append(seg)

    if len(sources) != len(source_images):
        _logger.warning("Found %d source images but only %d matching targets", len(source_images), len(sources))

    _logger.info("Found %d image pairs", len(sources))

    return sources, targets


class SegmentationDataset:
    def __init__(self, source_spec: ImageSetSpec, seg_spec: ImageSetSpec):
        # TODO support for masks
        self.source_spec = source_spec
        self.seg_spec = seg_spec

    def make_dataset(self, shuffle=True):
        """
        Makes a `tf.data.Dataset` consisting of single pairs of images (no batching!)
        :return:
        """
        image_paths = pic2pic_matching_files(source_spec=self.source_spec, target_spec=self.seg_spec)
        if len(image_paths[0]) == 0:
            raise ValueError("Could not find any images for the dataset")

        images = make_image_dataset(image_paths, (self.source_spec.channels, self.seg_spec.channels), shuffle=shuffle)
        return images

    @staticmethod
    def from_dict(data):
        if "directory" in data:
            root = pathlib.Path(data["directory"])
        else:
            root = None

        source_spec = ImageSetSpec.from_dict(data["source"], root=root)
        target_spec = ImageSetSpec.from_dict(data["target"], root=root)
        return SegmentationDataset(source_spec, target_spec)

    @staticmethod
    def from_json(path):
        path = pathlib.Path(path)
        return SegmentationDataset.from_dict(json.loads(path.read_text()))

import tensorflow as tf
import imageio
import pathlib
import logging
import numpy as np

from unet.dataset.images import load_images

_logger = logging.getLogger(__name__)


"""
We consider the following types of augmentations, that arise from transformation properties of the segmentation process.
For utility, for an input image `x` we call the corresponding ground truth `y=g(x)`.
1) Invariances. A transformation `f` such that `g(x) = g(f(x))`. Examples are noise modifications, brightness and 
contrast changes. 
"""


class TransformationProperty:
    """
    This class encodes certain transformation properties of the segmentation problem. Let `x` be the input image, and
    `y = g(x)` the ground truth segmentation. A transformation property is some form of invariance given by
    `g(f(x; t)) = h(g(x); t)` with a parameter `t`. By convention, a parameter of `t=0` indicates that `g, h` are
    identity mappings. We can classify the transformations based on the possible values of `t`. It is either a
    continuous symmetry, when t is from the interval `[0, 1]`, or a discrete symmetry, where t is from `{0, 1, ..., k}`.
    For implementation purposes, we allow "random" operations, which get an additional integer seed passed, which is the
    same for the image and the segmentation transformation.

    A further problem arises is the mapping `h` is not bijective. This happens e.g. for rotations that are not multiples
    of 90°. In that case a binary mask is needed to determine which pixels are valid and which are artifacts. For that
    reason, and additional `mask_transform` function needs to be defined. It is assumed that masking does not depend on
    the actual contents of `image` and `segmentation`, but only on the shape.
    """
    def __init__(self, num_discrete: int):
        """
        :param num_discrete: The number of discrete transformations represented with this property. Set to 0 for
        continuous transformations.
        """
        self._num_discrete = num_discrete

    def image_transform(self, image: tf.Tensor, argument: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()

    def segmentation_transform(self, segmentation: tf.Tensor, argument: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()

    def mask_transform(self, mask: tf.Tensor, argument: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()

    def inverse_segmentation_transform(self, segmentation: tf.Tensor, argument: tf.Tensor,
                                       seed: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()

    def inverse_mask_transform(self, mask: tf.Tensor, argument: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()

    def augment(self, source: tf.Tensor, segmentation: tf.Tensor, mask: tf.Tensor):
        # verify inputs
        source.shape.assert_has_rank(3)

        choice = self._get_argument()
        seed = tf.random.uniform((2,), 0, tf.int32.max, dtype=tf.int32)

        return (self.image_transform(source, argument=choice, seed=seed),
                self.segmentation_transform(segmentation, argument=choice, seed=seed),
                self.mask_transform(mask, argument=choice, seed=seed))

    def boost(self, source: tf.Tensor, mask: tf.Tensor):
        # verify inputs
        source.shape.assert_has_rank(3)

        choice = self._get_argument()
        seed = tf.random.uniform((), 0, tf.int32.max, dtype=tf.int32)

        def back_trafo(segmentation, mask):
            return (self.segmentation_transform(segmentation, argument=choice, seed=seed),
                    self.mask_transform(mask, argument=choice, seed=seed))

        return (self.image_transform(source, argument=choice, seed=seed),
                self.mask_transform(mask, argument=choice, seed=seed),
                back_trafo)

    def _get_argument(self):
        if self._num_discrete > 0:
            choice = tf.random.uniform((), 0, self._num_discrete + 1, dtype=tf.int32)
        else:
            choice = tf.random.uniform((), 0.0, 1.0)
        return choice

# TODO figure out how we can make zoom/crop work sensibly.


class AugmentationPipeline:
    def __init__(self, transformations: list):
        self._transformations = transformations or []

    def add_transformation(self, trafo):
        self._transformations.append(trafo)

    @tf.function
    def augment(self, source: tf.Tensor, segmentation: tf.Tensor, mask: tf.Tensor = None):
        assert source.dtype.is_floating, source.dtype
        assert segmentation.dtype.is_floating, segmentation.dtype

        if mask is None:
            mask = tf.ones_like(segmentation)

        for trafo in self._transformations:
            source, segmentation, mask = trafo.augment(source, segmentation, mask)
        return source, segmentation, mask

    @tf.function
    def augment_batch(self, source: tf.Tensor, segmentation: tf.Tensor, mask: tf.Tensor = None):
        if mask is None:
            mask = tf.ones_like(segmentation)

        return tf.map_fn(lambda x: self.augment(*x), (source, segmentation, mask), back_prop=False)

    def test_for_image(self, source, segmentation, target_folder, num_samples: int, mask=None):
        if not isinstance(target_folder, pathlib.Path):
            target_folder = pathlib.Path(target_folder)

        if isinstance(source, (str, pathlib.Path)):
            source = tf.io.decode_image(tf.io.read_file(str(source)), dtype=tf.float32)
        if isinstance(segmentation, (str, pathlib.Path)):
            segmentation = tf.io.decode_image(tf.io.read_file(str(segmentation)), dtype=tf.float32)

        if not target_folder.exists():
            target_folder.mkdir()

        for i in range(num_samples):
            img, seg, msk = self.augment(source, segmentation, mask)
            imageio.imwrite(target_folder / ("%03d-image.png" % i), img.numpy())
            imageio.imwrite(target_folder / ("%03d-segmentation.png" % i), seg.numpy())
            imageio.imwrite(target_folder / ("%03d-mask.png" % i), msk.numpy())

    def augmented_dataset(self, source_images, segmentation_images):
        dataset = self.images_from_list(source_images, segmentation_images, shuffle=True).repeat()
        return self.augment_dataset(dataset)

    @staticmethod
    def images_from_list(source_images, segmentation_images, shuffle=True, channels_in=3, channels_out=3):
        """
        Given a list of source images and a list of corresponding ground truth segmentations,
        builds a `tf.data.Dataset` out of these image pairs.
        :param source_images: List of paths to the source images.
        :param segmentation_images: List of paths to the ground truth segmentations.
        :param shuffle: Whether to shuffle the dataset.
        :param channels_in: Number of channels in the input image.
        :param channels_out: Number of channels in the segmentation.
        :return: A dataset of pairs `(image, ground_truth)`.
        """
        source_images = list(map(str, source_images))
        segmentation_images = list(map(str, segmentation_images))

        assert len(source_images) == len(segmentation_images)
        assert len(source_images) > 0, "No images supplied"
        _logger.info("Creating dataset with %d images", len(source_images))

        dataset = tf.data.Dataset.from_tensor_slices((source_images, segmentation_images))
        dataset = dataset.map(lambda x, y: load_images((x, y), (channels_in, channels_out)), num_parallel_calls=4).cache()
        if shuffle:
            return dataset.shuffle(len(source_images))
        else:
            return dataset

    @staticmethod
    def images_from_directories(source_dir, segmentation_dir, shuffle=True, pattern="*.png",
                                channels_in=3, channels_out=3, name_transform=None):
        """
        Given a source an a segmentations folder, this function returns a `tf.data.Dataset` containing
        all pairs of images. This assumes that the filename in `source_dir` is the same as the
        corresponding filename in the `segmentation_dir`. Source files for which no segmentation file
        does not exist wil be ignored.
        :param source_dir: Directory with the input files.
        :param segmentation_dir: Directory with the ground truth segmentations.
        :param shuffle: Whether to shuffle the image. Otherwise they are sorted by file name.
        :param pattern: File name pattern to select image files. Defaults to `*.png` for png files.
        :param channels_in: Number of channels in the input image.
        :param channels_out: Number of channels in the segmentation.
        :param name_transform: Function that returns the name of the segmentation file based on the original file name.
        :return: A dataset of image pairs, as per `images_from_list`.
        """
        source_dir = pathlib.Path(source_dir)
        seg_dir = pathlib.Path(segmentation_dir)

        sources = []
        segmentations = []
        source_images = sorted(source_dir.glob(pattern))
        for source in source_images:
            seg_name = source.name
            if name_transform:
                seg_name = name_transform(seg_name)
            segmentation_image = (seg_dir / seg_name)
            if not segmentation_image.exists():
                continue
            sources.append(source)
            segmentations.append(segmentation_image)

        if len(sources) != len(source_images):
            _logger.warn("Found %d source images but only %d matching segmentations", len(source_images), len(sources))

        return AugmentationPipeline.images_from_list(sources, segmentations, shuffle=shuffle,
                                                     channels_in=channels_in, channels_out=channels_out)

    def augment_dataset(self, dataset: tf.data.Dataset):
        """
        Takes as input an image dataset consists of tuples (image, segmentation)
        (the image tensors have rank 3) or tuples (image, segmentation, mask).
        :param dataset: The dataset containing augmented images.
        :return:
        """
        def process_images(img, seg, mask=None):
            img = tf.image.convert_image_dtype(img, tf.float32)
            seg = tf.image.convert_image_dtype(seg, tf.float32)
            if mask is not None:
                mask = 1.0 - tf.cast(tf.equal(mask, 0), tf.float32)
            return self.augment(img, seg, mask)
        return dataset.map(process_images, num_parallel_calls=4)

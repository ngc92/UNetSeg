from typing import Tuple

import tensorflow as tf
import imageio
import pathlib
import logging
import numpy as np

from unet.dataset.images import load_images, make_image_dataset

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
        """
        Performs a random augmentation on the given input data.
        :param source: The source image that should be augmented.
        :param segmentation: The corresponding ground truth segmentation, that might also be transformed due to the
        augmentation.
        :param mask: The input mask. Some augmentations might make parts of the input invalid, which will be masked.
        :return: New source, segmentation, and mask.
        """
        # verify inputs
        source.shape.assert_has_rank(3)

        choice = self._get_argument()
        seed = tf.random.uniform((2,), 0, tf.int32.max, dtype=tf.int32)

        return (self.image_transform(source, argument=choice, seed=seed),
                self.segmentation_transform(segmentation, argument=choice, seed=seed),
                self.mask_transform(mask, argument=choice, seed=seed))

    def boost(self, source: tf.Tensor, mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, callable]:
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

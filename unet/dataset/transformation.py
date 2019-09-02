from abc import abstractmethod
from typing import Tuple

import tensorflow as tf


class ArgSpec:
    @abstractmethod
    def __call__(self):
        raise NotImplementedError()


class DiscreteArg(ArgSpec):
    def __init__(self, num_states):
        self.num_states = num_states

    def __call__(self):
        return tf.random.uniform((), 0, self.num_states + 1, dtype=tf.int32)


def ContinuousArg(ArgSpec):
    def __init__(self, lower=None, upper=None):
        if lower is None:
            if upper is None:
                lower = 0.0
                upper = 1.0
            else:
                upper = lower
                lower = 0.0

        # lower not None, upper None

class TransformationProperty:
    """
    This class encodes certain transformation properties of the segmentation problem. Let `x` be the input image, and
    `y = g(x)` the ground truth segmentation. A transformation property is some form of invariance given by
    `g(f(x; t)) = h(g(x); t)` with a parameter `t`. By convention, a parameter of `t=0` indicates that `g, h` are
    identity mappings. We can classify the transformations based on the possible values of `t`. It is either a
    continuous symmetry, when `t` is from the interval `[0, 1]`, or a discrete symmetry, where t is from `{0, 1, ..., k}`.
    For implementation purposes, we allow "random" operations, which get an additional integer seed passed, which is the
    same for the image and the segmentation transformation. We also allow `t` to be
    a structured data type, i.e. being a tuple/dict with values as above.

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
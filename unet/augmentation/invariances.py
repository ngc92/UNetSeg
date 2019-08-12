from typing import Tuple

import tensorflow as tf

from unet.augmentation.augmentation import TransformationProperty


class Invariance(TransformationProperty):
    """
    An invariance is a `TransformationProperty` where the ground truth is not influenced by the image transformation.
    It is expected to leave the masks unchanged.
    """
    def __init__(self, num_discrete: int):
        super().__init__(num_discrete=num_discrete)

    def image_transform(self, image: tf.Tensor, argument: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        return self.transform(image, argument, seed)

    def segmentation_transform(self, segmentation: tf.Tensor, argument: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        return segmentation

    def inverse_segmentation_transform(self, segmentation: tf.Tensor, argument: tf.Tensor,
                                       seed: tf.Tensor) -> tf.Tensor:
        return segmentation

    def mask_transform(self, mask: tf.Tensor, argument: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        return mask

    def inverse_mask_transform(self, mask: tf.Tensor, argument: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        return mask

    def transform(self, image, k, seed):
        raise NotImplementedError()


class ContrastInvariance(Invariance):
    def __init__(self, min_factor: float, max_factor: float):
        super().__init__(num_discrete=0)
        self.min_factor = min_factor
        self.max_factor = max_factor

    def transform(self, image, factor, seed):
        return tf.image.adjust_contrast(image, self.min_factor + factor * (self.max_factor - self.min_factor))


class BrightnessInvariance(Invariance):
    def __init__(self, max_delta: float):
        super().__init__(num_discrete=0)
        self.max_delta = max_delta

    def transform(self, image, factor, seed):
        return tf.image.adjust_brightness(image, (2 * factor - 1) * self.max_delta)


class NoiseInvariance(Invariance):
    def __init__(self, max_strength: float):
        super().__init__(num_discrete=0)
        self.max_strength = max_strength

    def transform(self, image, factor, seed):
        noise_pattern = tf.random.stateless_uniform(tf.shape(image), minval=-1, maxval=1, seed=seed)
        offset = noise_pattern * factor * self.max_strength
        return tf.clip_by_value(image + offset, 0.0, 1.0)

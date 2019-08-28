from typing import Tuple

import tensorflow as tf

from unet.dataset.augmentation import TransformationProperty
from unet.layers import BlurLayer
from unet.ops import make_random_field, occlude_image


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


class LocalContrastInvariance(Invariance):
    def __init__(self, contrast_factor: float, min_segments: int = 4,
                 min_segment_size: float = 16, blur_size: int = 3):
        super().__init__(num_discrete=0)
        self.contrast_factor = contrast_factor
        self.min_segments = min_segments
        self.min_segment_size = min_segment_size
        self.blur_size = blur_size
        self._blur = BlurLayer(self.blur_size)

    def transform(self, image, factor, seed):
        low_contrast_image = tf.image.adjust_contrast(image, self.contrast_factor)

        # individual noise pattern
        noise_pattern = make_random_field(image, 0.5, self.min_segments, self.min_segment_size, 1, self._blur, seed)[0]
        noise_pattern = (0.5 + noise_pattern) * factor
        result = image * (1.0 - noise_pattern) + low_contrast_image * noise_pattern
        return result


class LocalBrightnessInvariance(Invariance):
    def __init__(self, brightness_change: float, min_segments: int = 4,
                 min_segment_size: float = 16, blur_size: int = 3):
        super().__init__(num_discrete=0)
        self.brightness_change = brightness_change
        self.min_segments = min_segments
        self.min_segment_size = min_segment_size
        self.blur_size = blur_size
        self._blur = BlurLayer(self.blur_size)

    def transform(self, image, factor, seed):
        # individual noise pattern
        noise_pattern = make_random_field(image, self.brightness_change, self.min_segments,
                                          self.min_segment_size, 1, self._blur, seed)[0]
        return tf.clip_by_value(image + noise_pattern * factor, 0.0, 1.0)


class OcclusionInvariance(Invariance):
    def __init__(self, min_size: int, max_size: int, max_occlusions: int):
        super().__init__(num_discrete=max_occlusions)
        self.min_size = min_size
        self.max_size = max_size

    def transform(self, image, k, seed):
        r = tf.random.stateless_uniform((4*k,), seed, 0.0, 1.0)
        for i in range(k):
            print(i)
            height = tf.cast(r[4*i + 0] * (self.max_size - self.min_size) + self.min_size, tf.int32)
            width = tf.cast(r[4*i + 1] * (self.max_size - self.min_size) + self.min_size, tf.int32)

            # TODO randomly decide noise strength in mask?
            image = occlude_image(image, height, width, r[4*i + 2], r[4*i + 3], 0)

        return image

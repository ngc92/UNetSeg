import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from unet.augmentation.augmentation import TransformationProperty
from unet.layers import BlurLayer
from unet.ops import make_random_field


class MaybeTransform:
    def __init__(self, op):
        self.op = op

    def __call__(self, image, k):
        return tf.cond(tf.equal(k, 0), lambda: image, lambda: self.op(image))


class GeometryEquivariance(TransformationProperty):
    def image_transform(self, image: tf.Tensor, argument: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        return self.transform(image, argument, seed)

    def segmentation_transform(self, segmentation: tf.Tensor, argument: tf.Tensor,  seed: tf.Tensor) -> tf.Tensor:
        return self.transform(segmentation, argument, seed)

    def mask_transform(self, mask: tf.Tensor, argument: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        return self.transform(mask, argument, seed)

    def inverse_segmentation_transform(self, segmentation: tf.Tensor, argument: tf.Tensor,
                                       seed: tf.Tensor) -> tf.Tensor:
        return self.inverse(segmentation, argument, seed)

    def inverse_mask_transform(self, mask: tf.Tensor, argument: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        return self.inverse(mask, argument, seed)

    def transform(self, image: tf.Tensor, k: tf.Tensor, seed: tf.Tensor):
        raise NotImplementedError()

    def inverse(self, image: tf.Tensor, k: tf.Tensor, seed: tf.Tensor):
        raise NotImplementedError()


class HorizontalFlips(GeometryEquivariance):
    def __init__(self):
        super().__init__(num_discrete=1)
        self._trafo = MaybeTransform(tf.image.flip_left_right)

    def transform(self, image, k, seed):
        return self._trafo(image, k)

    def inverse(self, image, k, seed):
        return self._trafo(image, k)


class VerticalFlips(GeometryEquivariance):
    def __init__(self):
        super().__init__(num_discrete=1)
        self._trafo = MaybeTransform(tf.image.flip_up_down)

    def transform(self, image, k, seed):
        return self._trafo(image, k)

    def inverse(self, image, k, seed):
        return self._trafo(image, k)


class Rotation90Degrees(GeometryEquivariance):
    def __init__(self):
        super().__init__(num_discrete=3)

    def transform(self, image, k, seed):
        return tf.image.rot90(image, k)

    def inverse(self, image, k, seed):
        return tf.image.rot90(image, 4 - k)


class FreeRotation(GeometryEquivariance):
    def __init__(self):
        super().__init__(num_discrete=0)

    def transform(self, image: tf.Tensor, angle: tf.Tensor, seed) -> tf.Tensor:
        img = tfa.image.rotate(image, angle * 2 * np.pi)
        return img

    def inverse(self, image: tf.Tensor, angle: tf.Tensor, seed) -> tf.Tensor:
        img = tfa.image.rotate(image, -angle * 2 * np.pi)
        return img


class Warp(GeometryEquivariance):
    def __init__(self, min_strength: float, max_strength: float, min_flow_segments: int = 4,
                 min_flow_segment_size: float = 16, blur_size: int = 3):
        super().__init__(num_discrete=0)
        self.min_strength = min_strength
        self.max_strength = max_strength
        self.min_flow_segments = min_flow_segments
        self.min_flow_segment_size = min_flow_segment_size
        self.blur_size = blur_size
        self._blur_flow_field = BlurLayer(self.blur_size)

    def transform(self, image: tf.Tensor, strength: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        flow_field = self.make_flow_field(image, strength, seed)
        warped = tfa.image.dense_image_warp(image[tf.newaxis, ...], flow_field)[0]
        warped.set_shape(image.shape)
        return warped

    def inverse(self, image: tf.Tensor, strength: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        flow_field = self.make_flow_field(image, strength, seed)
        warped = tfa.image.dense_image_warp(image[tf.newaxis, ...], -flow_field)[0]
        warped.set_shape(image.shape)
        return warped

    def make_flow_field(self, image, strength, seed):
        magnitude = self.min_strength + strength * (self.max_strength - self.min_strength)
        return make_random_field(image, magnitude, self.min_flow_segments,
                                 self.min_flow_segment_size, 2, self._blur_flow_field, seed)


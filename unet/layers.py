import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

from unet.ops import GaussKernelInitializer


class Upscale2DConv(keras.layers.Layer):
    def __init__(self, filters, name=None, activation="relu", **kwargs):
        """
        An upscaling layer that does an upsampling followed by a single convolution.
        :param filters: Number of filters in the convolutional layer.
        :param activation: Activation function to use after the convolution.
        :param name: Name of this block. Optional.
        """
        super().__init__(name=name, **kwargs)
        self.upsample = keras.layers.UpSampling2D()
        self.up_conv = keras.layers.Conv2D(filters, kernel_size=3, activation=activation)

    def call(self, inputs, **kwargs):
        return self.upsample(inputs)


class BlurLayer(keras.layers.Layer):
    def __init__(self, size: int, padding="VALID", **kwargs):
        """
        A layer that applies guassian blur to a given image.
        :param size: The size of the blur filter. The amount of neighbouring pixels taken into account when calculating
        the blurred image. The standard deviation of the gaussian is chosen to be half that distance.
        :param padding: Whether to include padding such that the resulting image has the same dimensions as the input.
        """
        super().__init__(**kwargs)
        self._size = size
        self._padding = padding

        self.kernel = None

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(2 * self._size + 1, 2 * self._size + 1, 1, 1),
            initializer=GaussKernelInitializer(self._size, self._size / 2))

    def call(self, inputs: tf.Tensor, **kwargs):
        if inputs.shape.rank == 3:
            return self._blur_by_convolution(inputs[None, ...], self._padding)[0]
        else:
            return self._blur_by_convolution(inputs, self._padding)

    def _blur_by_convolution(self, x, padding):
        """
        maps `_blur_single_channel` over the input channels if necessary, and ensures correct normalization for
        padded blurs. Expects a batch of images as input.
        """
        if padding:
            # calculate the weights for each position and normalize the result correspondingly
            mask = tf.ones(shape=tf.concat([tf.shape(x)[:-1], [1]], axis=0))
            blurred_mask = self._blur_single_channel(mask)
            return self._blur_by_convolution(x, False) / blurred_mask

        if x.shape[-1] != 1:
            # put channel in batch dimension
            transposed = tf.transpose(x, perm=(3, 0, 1, 2))
            blurred = tf.map_fn(lambda y: self._blur_single_channel(y[..., tf.newaxis]), transposed)[..., 0]
            return tf.transpose(blurred, perm=(1, 2, 3, 0))
        else:
            return self._blur_single_channel(x)

    def _blur_single_channel(self, x):
        """
        Performs the blurring for a single input/output channel.
        """
        return tf.nn.conv2d(x, self.kernel, strides=[1, 1, 1, 1], padding=self._padding)

    def get_config(self):
        return {**super().get_config(), "size": self._size, "padding": self._padding}

import tensorflow as tf
from tensorflow import keras

from unet.layers import Upscale2DConv
from unet.ops import crop_and_concat


class DownBlock(keras.layers.Layer):
    def __init__(self, filters, name=None, activation="relu", **kwargs):
        """
        Downward block of the U-Net architecture. Two convolutions followed by a max pooling with stride two.
        Therefore, the incoming image size is reduced from `s` to `(s-4) / 2`. Therefore, input images are required
        to have sizes divisible by two. Calling the layer with an incompatible size causes an assertion to be raised.
        :param filters: Number of filters for the convolutional layers. Both layers will use the same amount of filters.
        :param activation: Activation function to use after the convolutions.
        :param name: Name of this block. Optional.
        """
        super().__init__(name=name, **kwargs)
        self.conv1 = keras.layers.Conv2D(filters, kernel_size=3, activation=activation)
        self.conv2 = keras.layers.Conv2D(filters, kernel_size=3, activation=activation)
        self.pool = keras.layers.MaxPool2D(pool_size=2, strides=(2, 2))

    def call(self, x):
        tf.debugging.Assert(tf.equal(tf.shape(x)[1] % 2, 0), data=("Input size is not a multiple of two", tf.shape(x), tf.shape(x)[1]), summarize=5)
        tf.debugging.Assert(tf.equal(tf.shape(x)[2] % 2, 0), data=("Input size is not a multiple of two", tf.shape(x), tf.shape(x)[2]), summarize=5)
        x = self.conv1(x)
        x = self.conv2(x)
        y = self.pool(x)
        return y, x


class UpBlock(keras.layers.Layer):
    def __init__(self, filters, use_upscaling=False, name=None, activation="relu", **kwargs):
        """
        The upwards block of the U-Net. This layer expects a tuple as input, where the first element comes from the
        sequential part of the model and the second entry from the skip connections. Given an input of size `s` this
        layer produces an output of size `(s-4) * 2`. This means, in particular, that
        :param filters: Number of filters in the convolutional layers.
        :param use_upscaling: Set to true to use an upsampling followed by a convolution, and to false (default) to use
        a deconvolution as the resolution increasing operation.
        :param activation: Activation function to use after the convolutions.
        :param name: Name of this block. Optional.
        """
        super().__init__(name=name, **kwargs)
        self.conv1 = keras.layers.Conv2D(filters, kernel_size=3, activation=activation)
        self.conv2 = keras.layers.Conv2D(filters, kernel_size=3, activation=activation)
        if use_upscaling:
            self.upsample = Upscale2DConv(filters)
        else:
            self.upsample = keras.layers.Conv2DTranspose(filters, kernel_size=3, strides=2,
                                                         padding="same", activation=activation)

    def call(self, inputs):
        x, y = inputs
        x = crop_and_concat(x, y)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.upsample(x)


class Bottleneck(keras.layers.Layer):
    def __init__(self, filters, use_upscaling=False, name="bottleneck", activation="relu", **kwargs):
        """
        The middle part of the U-Net. Like the upsampling layer, but performs dropout before
        increasing the resolution.
        :param filters: Number of filters in the convolutional layers.
        :param use_upscaling: Whether to use upsampling or deconvolution.
        :param activation: Activation function to use after the convolutions.
        :param name: Name of this block.
        """
        super().__init__(name=name, **kwargs)
        self.conv1 = keras.layers.Conv2D(filters, kernel_size=3, activation=activation)
        self.conv2 = keras.layers.Conv2D(filters, kernel_size=3, activation=activation)
        self.drop = keras.layers.Dropout(0.5)
        if use_upscaling:
            self.upsample = Upscale2DConv(filters)
        else:
            self.upsample = keras.layers.Conv2DTranspose(filters, kernel_size=3, strides=2,
                                                         padding="same", activation=activation)

    def call(self, x, training=None):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop(x, training=training)
        return self.upsample(x)


class OutputBlock(keras.layers.Layer):
    def __init__(self, filters, n_channels, name="output", activation="relu", **kwargs):
        """
        Output block of the U-Net. Performs three convolutions, where the last one produces `n_classes` many channels
        and has no nonlinearity. This means that this layer is intended to output logits.
        :param filters: Number of filters in the first two convolutional layers.
        :param activation: Activation function to use after the non-output convolutions.
        :param n_channels: Numer of output channels.
        :param name: Name of this block.
        """
        super().__init__(name, **kwargs)
        self.conv1 = keras.layers.Conv2D(filters, kernel_size=3, activation=activation)
        self.conv2 = keras.layers.Conv2D(filters, kernel_size=3, activation=activation)
        self.conv_out = keras.layers.Conv2D(n_channels, kernel_size=1)

    def call(self, inputs):
        x, y = inputs
        x = crop_and_concat(x, y)
        return self.conv_out(self.conv2(self.conv1(x)))

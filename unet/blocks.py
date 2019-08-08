import tensorflow as tf
import tensorflow.keras as keras


def _crop_and_concat(inputs, residual_input):
    aw = inputs.shape[1]
    bw = residual_input.shape[1]
    surplus = bw - aw

    bbox_begin = tf.stack([0, surplus // 2, surplus // 2, 0])
    bbox_size = tf.stack([-1, aw, aw, -1])

    cropped = tf.slice(residual_input, bbox_begin, bbox_size)
    cropped.set_shape([None, aw, aw, residual_input.get_shape()[3]])

    return tf.concat([inputs, cropped], axis=-1)



class DownBlock(keras.layers.Layer):
    def __init__(self, filters, name=None, **kwargs):
        """
        Downward block of the U-Net architecture. Two convolutions followed by a max pooling with stride two.
        :param filters: Number of filters for the convolutional layers. Both layers will use the same amount of filters.
        :param name: Name of this block. Optional.
        """
        super().__init__(name=name, **kwargs)
        self.conv1 = keras.layers.Conv2D(filters, kernel_size=3, activation="relu")
        self.conv2 = keras.layers.Conv2D(filters, kernel_size=3, activation="relu")
        self.pool = keras.layers.MaxPool2D(pool_size=2, strides=(2, 2))

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        y = self.pool(x)
        return y, x


class Upscale2DConv(keras.layers.Layer):
    def __init__(self, filters, name=None, **kwargs):
        """
        An upscaling layer that does an upsampling followed by a single convolution.
        :param filters: Number of filters in the convolutional layer.
        :param name: Name of this block. Optional.
        """
        super().__init__(name=name, **kwargs)
        self.upsample = keras.layers.UpSampling2D()
        self.up_conv = keras.layers.Conv2D(filters, kernel_size=3, activation="relu")

    def call(self, inputs, **kwargs):
        return self.upsample(inputs)


class UpBlock(keras.layers.Layer):
    def __init__(self, filters, use_upscaling=False, name=None, **kwargs):
        """
        The upwards block of the U-Net. This layer expects a tuple as input, where the first element comes from the
        sequential part of the model and the second entry from the skip connections.
        :param filters: Number of filters in the convolutional layers.
        :param use_upscaling: Set to true to use an upsampling followed by a convolution, and to false (default) to use
        a deconvolution as the resolution increasing operation.
        :param name: Name of this block. Optional.
        """
        super().__init__(name=name, **kwargs)
        self.conv1 = keras.layers.Conv2D(filters, kernel_size=3, activation="relu")
        self.conv2 = keras.layers.Conv2D(filters, kernel_size=3, activation="relu")
        if use_upscaling:
            self.upsample = Upscale2DConv(filters)
        else:
            self.upsample = keras.layers.Conv2DTranspose(filters, kernel_size=3, strides=2,
                                                         padding="same", activation="relu")

    def call(self, inputs):
        x, y = inputs
        x = _crop_and_concat(x, y)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.upsample(x)


class Bottleneck(keras.layers.Layer):
    def __init__(self, filters, use_upscaling=False, name="bottleneck", **kwargs):
        """
        The middle part of the U-Net. Like the upsampling layer, but performs dropout before
        increasing the resolution.
        :param filters: Number of filters in the convolutional layers.
        :param use_upscaling: Whether to use upsampling or deconvolution.
        :param name: Name of this block.
        """
        super().__init__(name=name, **kwargs)
        self.conv1 = keras.layers.Conv2D(filters, kernel_size=3, activation="relu")
        self.conv2 = keras.layers.Conv2D(filters, kernel_size=3, activation="relu")
        self.drop = keras.layers.Dropout(0.5)
        if use_upscaling:
            self.upsample = Upscale2DConv(filters)
        else:
            self.upsample = keras.layers.Conv2DTranspose(filters, kernel_size=3, strides=2,
                                                         padding="same", activation="relu")

    def call(self, x, training=None):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop(x, training=training)
        return self.upsample(x)


class OutputBlock(keras.layers.Layer):
    def __init__(self, filters, n_classes, name="output", **kwargs):
        """
        Output block of the U-Net. Performs three convolutions, where the last one produces `n_classes` many channels
        and has a sigmoid nonlinearity.
        :param filters: Number of filters in the first two convolutional layers.
        :param n_classes: Numer of output channels.
        :param name: Name of this block.
        """
        super().__init__(name, **kwargs)
        self.conv1 = keras.layers.Conv2D(filters, kernel_size=3, activation="relu")
        self.conv2 = keras.layers.Conv2D(filters, kernel_size=3, activation="relu")
        self.conv_out = keras.layers.Conv2D(n_classes, kernel_size=1, activation="sigmoid")

    def call(self, inputs):
        x, y = inputs
        x = _crop_and_concat(x, y)
        return self.conv_out(self.conv2(self.conv1(x)))

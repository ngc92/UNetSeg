import numpy as np
import tensorflow as tf
from tensorflow import keras

from unet.blocks import DownBlock, Bottleneck, UpBlock, OutputBlock
from unet.model.segmentation import SegmentationModel


class UNetMapping(keras.Model):
    """
    This class implements a mapping that corresponds to the unet structure. Input images of shape `w x h x c` are mapped
    to new images `(w-b) x (h-b) x o` for a specified number `o` of output channels. There is expected to be some
    correspondence between the pixel locations in the input image and in the output image.
    """
    def __init__(self, out_channels: int = 1, filters: int = 64, depth: int = 4, use_upscaling: bool = False):
        if not isinstance(out_channels, int) or out_channels < 1:
            raise ValueError("Invalid number of output channels")

        self._down_blocks = [DownBlock(filters=filters, name="input")]
        self._up_blocks = []

        for num_filters in [2**k * filters for k in range(1, depth)]:
            down_block = DownBlock(filters=num_filters, name="down_%d" % num_filters)
            self._down_blocks.append(down_block)

        self._bottleneck = Bottleneck(filters=2**depth * filters, use_upscaling=use_upscaling)

        for num_filters in [2**k * filters for k in reversed(range(1, depth))]:
            self._up_blocks.append(UpBlock(filters=num_filters, name="up_%d" % num_filters, use_upscaling=use_upscaling))

        self._out_block = OutputBlock(filters=filters, n_channels=out_channels)
        self._depth = depth

    def call(self, inputs, training=None):
        """
        Applies the U-Net to the input image..
        :param inputs: A batch of images
        :param training: Whether to operate in training or inference mode. Activates dropout in the bottleneck layer.
        :return: The segmented image. Note that this is smaller than the input image.
        """
        skip_connections = []
        x = inputs
        for block in self._down_blocks:
            x, skip = block(x)
            skip_connections.append(skip)

        x = self._bottleneck(x, training=training)
        for block in self._up_blocks:
            x = block((x, skip_connections.pop()))

        return self._out_block((x, skip_connections.pop()))

    @property
    def depth(self):
        """The depth of this UNet: Number of downward blocks."""
        return self._depth

    @property
    def border_width(self):
        """
        The size of the border of this UNet. This is the number of pixels that the output image is smaller
        (symmetrically) than the input image.
        :return:
        """
        return _get_border_size(self.depth)

    def is_valid_input_size(self, input_size):
        """
        Checks whether the given shape is a valid input size for this network.
        :param input_size: A 2-tuple containing the height and width of the input image.
        :return: True, if the network can process the given input.
        """
        try:
            output = self.output_size(input_size)
            return True
        except AssertionError:
            return False

    def output_size(self, input_size):
        """The size of the output image, given an input of shape `image_size`"""
        return int(_get_output_size(input_size, self.depth))

    def input_size_for_output(self, output_size):
        return _get_input_size(output_size + 2 * self.border_width, self.depth, crop=False)


class UNetModel(SegmentationModel):
    def __init__(self, n_channels, filters=64, depth=4, use_upscaling=False, symmetries=None, normalize_input=True,
                 **kwargs):
        """
        A U-Net as in []. While the model is fully convolutional and thus not limited to a fixed input size,
        the structure of the convolution makes only certain input sizes possible. The call method of this model
        simply applies to U-Net to the input as-is, and will produce an error. For training, it is recommended
        that you use a fixed image size (e.g. by taking fixed sizes crops). For inference on (almost -- the size
        must be divisible by two) arbitrarily sizes images, the predict function can be used.
        :param n_channels: Number of channels in the output image.
        :param filters: Multiplier for the number of filters used. Each downward block doubles the number of filters.
        :param use_upscaling: Whether to use deconvolutions (False) or upscaling to increase the image size.
        """
        super().__init__(n_channels, normalize_input=normalize_input, **kwargs)

        self._mapping = UNetMapping(n_channels, filters, depth, use_upscaling)
        self._symmetries = symmetries or None

    @property
    def depth(self):
        return self._mapping.depth

    @property
    def border_width(self):
        return self._mapping.border_width

    def is_valid_input_size(self, input_size):
        """
        Checks whether the given shape is a valid input size for this network.
        :param input_size: A 2-tuple containing the height and width of the input image.
        :return: True, if the network can process the given input.
        """
        return self._mapping.is_valid_input_size(input_size)

    @property
    def symmetries(self):
        return self._symmetries

    def logits(self, inputs, training=None):
        """
        Applies the U-Net to the input image and returns the resulting logits.
        :param inputs: A batch of images
        :param training: Whether to operate in training or inference mode. Activates dropout in the bottleneck layer.
        :return: The segmented image. Note that this is smaller than the input image.
        """
        return self._mapping(inputs, training=training)

    def input_mask_to_output_mask(self, input_mask: tf.Tensor):
        # if mask is 0, 1 - mask is 1 and all pixels touched by this input will be masked.
        return 1.0 - keras.layers.MaxPooling2D(pool_size=1+2*self.border_width, strides=1)(1.0 - input_mask)

    def predict(self, image, padding=False):
        """
        Perform segmentation for the given image, which need not have a compatible shape. If padding is set, the image
        will simple be padded with zeros to the next valid shape. If padding is not set, an incompatible image will be
        split into four (overlapping) patches of compatible sizes, which will be processes individually and then
        combined. If padding is set, the image is padded with enough zeros to ensure that the segmentation has the same
        shape as the input image.
        :param image: An image or a batch of images. They need not have a shape that is compatible with the U-Net.
        :param padding: Whether to use zero-padding or
        :return: The segmented image.
        """
        depth = len(self._down_blocks)
        # check for batch dimension
        has_batch = True
        if len(image.shape) == 3:
            has_batch = False
            image = image[None, ...]

        # add "border size" many pixels so input and output can have same shape
        if padding:
            bs = self.border_width
            image = tf.pad(image, [[0, 0], [bs, bs], [bs, bs], [0, 0]])

        h, w, c = image.shape[1:]

        # check if the image size is natively supported
        if _get_input_size((h, w), depth) == (h, w):
            prediction = self(image)
        else:
            # if we allow padding, the image is extended further until the new size is valid
            if padding:
                nh, nw = _get_input_size((h, w), depth, crop=False)
                assert (nh - h) % 2 == 0
                assert (nw - w) % 2 == 0
                bh = int((nh - h) // 2)
                bw = int((nw - w) // 2)
                padded_image = tf.pad(image, [[0, 0], [bh, bh], [bw, bw], [0, 0]])
                prediction = self(padded_image)
                prediction = prediction[:, bh:-bh, bw:-bw, :]
            else:
                prediction = self._tiled_prediction(image)

        if has_batch:
            return prediction
        else:
            return prediction[0, ...]

    def _tiled_prediction(self, image):
        depth = len(self._down_blocks)
        h, w, c = image.shape[1:]

        bs = _get_border_size(depth)
        # try to cut the image into four tiles, ideally [0, h/2 + border], [h/2 - border, h]
        qh, qw = _get_input_size((h // 2 + bs, w // 2 + bs), depth, crop=False)
        # we need to do tiling.
        tile_11 = image[:, 0:qh, 0:qw, :]
        tile_12 = image[:, 0:qh, -qw:, :]
        tile_21 = image[:, -qh:, 0:qw, :]
        tile_22 = image[:, -qh:, -qw:, :]
        joined_batch = tf.concat([tile_11, tile_12, tile_21, tile_22], axis=0)
        joined_prediction = self(joined_batch)
        batch_size = tf.shape(image)[0]
        # undo the batching from before
        p_11, p_12, p_21, p_22 = (joined_prediction[n * batch_size:(n + 1) * batch_size] for n in range(4))

        # padding to get the partial predictions to have the correct shape
        ph = h - 2*bs - p_11.shape[1]
        pw = w - 2*bs - p_11.shape[2]

        weight = tf.ones_like(p_11)
        p_11 = tf.pad(p_11, [[0, 0], [0, ph], [0, pw], [0, 0]])
        w_11 = tf.pad(weight, [[0, 0], [0, ph], [0, pw], [0, 0]])
        # [0:sw], [pw:]
        p_12 = tf.pad(p_12, [[0, 0], [0, ph], [pw, 0], [0, 0]])
        w_12 = tf.pad(weight, [[0, 0], [0, ph], [pw, 0], [0, 0]])

        p_21 = tf.pad(p_21, [[0, 0], [ph, 0], [0, pw], [0, 0]])
        w_21 = tf.pad(weight, [[0, 0], [ph, 0], [0, pw], [0, 0]])
        p_22 = tf.pad(p_22, [[0, 0], [ph, 0], [pw, 0], [0, 0]])
        w_22 = tf.pad(weight, [[0, 0], [ph, 0], [pw, 0], [0, 0]])

        return tf.add_n([p_11, p_12, p_21, p_22]) / tf.add_n([w_11, w_12, w_21, w_22])

    def output_size(self, input_size):
        return self._mapping.output_size(input_size)

    def input_size_for_output(self, output_size):
        return self._mapping.input_size_for_output(output_size)


def _get_input_size(image_size, depth, crop=True):
    # two convolutions: size - 4, max-pool: size / 2
    # given a depth of `k` layers, and a size of `m` at the (input of) the bottleneck layer,
    # the network has an input size of `2**k (m+4) - 4`. Therefore, theoretically, the U-Net
    # could process any image of size `2**k m + 2**(k+2) - 4`.
    if isinstance(image_size, tuple):
        return tuple(_get_input_size(x, depth, crop) for x in image_size)

    m = _get_bottleneck_size(image_size, depth)
    if crop:
        m = int(np.floor(m))
    else:
        m = int(np.ceil(m))
    return 2**depth * (m + 4) - 4


def _get_output_size(image_size, depth):
    # two convolutions: size - 4, upsample: size * 2
    # s[k+1] = (s[k] - 4) * 2 = 2 s[k] - 8
    # s[n] = 2^n s - 8(2^n - 1)
    if isinstance(image_size, tuple):
        return (_get_output_size(x, depth) for x in image_size)

    m = _get_bottleneck_size(image_size, depth)
    assert m == int(m), "invalid input size"

    # -4, due to output block convolutions
    return 2**depth * m - 8*(2**depth - 1) - 4


def _get_border_size(depth):
    m = 16
    in_size = 2**depth * (m + 4) - 4
    out_size = _get_output_size(in_size, depth)
    assert (in_size - out_size) % 2 == 0
    return int((in_size - out_size) // 2)


def _get_bottleneck_size(image_size, depth):
    return (image_size + 4) / (2**depth) - 4

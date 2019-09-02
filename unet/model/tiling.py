from typing import Union, Tuple, TYPE_CHECKING

import tensorflow as tf

if TYPE_CHECKING:
    from unet.model.segmentation import SegmentationModel


class TiledPredictor:
    """
    A tiled predictor takes an existing predictor (currently: a SegmentationModel instance) and allows it to be allowed
    to arbitrarily shaped input images. This is implemented by taking overlapping (according to the base predictor
    border) patches of the image and predicting these separately, then stitching together the result. If the image size
    is not exactly compatible with this tiling, the last (in each dimension) tile has a bigger overlap with the
    other tiles, and the overlapping area in the result is averaged.
    """
    def __init__(self, base_predictor: "SegmentationModel", patch_shape: Union[int, Tuple[int, int]]):
        """
        :param base_predictor: The underlying predictor that is used to make predictions on a single patch.
        :param patch_shape: The shape of the patches in which the image will be cut. Can be either a tuple `(h, w)` or
        an integer for square patches.
        """
        self.base_predictor = base_predictor
        self.channels = base_predictor.channels
        self.border_width = base_predictor.border_width

        if isinstance(patch_shape, int):
            self.tile_size = (patch_shape, patch_shape)
        else:
            self.tile_size = tuple(patch_shape)

        if self.tile_size[0] <= 2 * self.border_width or self.tile_size[1] <= self.border_width:
            raise ValueError("Patch shape `%s` must be bigger than image border `%s`" % (patch_shape, self.border_width))

        if not self.base_predictor.is_valid_input_size(self.tile_size):
            raise ValueError("Patch shape `%s` must be a valid input shape" % patch_shape)

    def __call__(self, input):
        return self._apply_to_single_image(input)

    def _apply_to_single_image(self, image, return_weight=False):
        # TODO figure out how to apply this to a batch of images.
        border = 2 * self.border_width
        h, w, c = image.shape

        value = tf.Variable(tf.zeros((h - border, w - border, self.channels)))
        weight = tf.Variable(tf.zeros((h - border, w - border)))

        for y, a in strided_iteration(h, self.tile_size[0] - border, self.tile_size[0]):
            for x, b in strided_iteration(w, self.tile_size[1] - border, self.tile_size[1]):
                patch = image[tf.newaxis, y:a, x:b, :]
                pred = self.base_predictor(patch)[0]
                mask = self.base_predictor.input_mask_to_output_mask(tf.ones((1, a-y, b-x, 1)))[0]

                indices = make_update_indices((y, a - border), (x, b - border))
                value.scatter_nd_add(indices, updates=pred * mask)
                weight.scatter_nd_add(indices, updates=mask[..., 0])

        weight = weight.value()[..., tf.newaxis]
        # for debugging, we might want to get access to the weights!
        if return_weight:
            return value.value(), weight

        return value.value() / weight


def make_update_indices(x, y):
    """
    Makes the indices Tensor for an update of a patch given by x and y.
    :param x: A 2-tuple containing the lower and upper bound of the first coordinate.
    :param y: A 2-tuple containing lower and upper bound of the second coordinate.
    :return: A indexing tensor to we used with scatter_nd.
    """
    x = tf.range(*x)
    y = tf.range(*y)
    p, q = tf.meshgrid(x, y, indexing="ij")
    indices = tf.stack([p, q], axis=-1)
    return indices


def strided_iteration(size, stride, width):
    """
    Iterates from 0 to `size` with stride `stride`,
    and returns intervals of width `width`. If the last interval would exceed `size`, an interval `[size-width, size]`
    is returned instead.
    :param size: The size of the entire interval, i.e. the maximum value the upper bound of an interval can take.
    :param stride: The stride, i.e. how much to shift the lower bound of the interval each step.
    :param width: The width of the yielded intervals.
    :return: A generator that yields pairs.
    """
    x = 0
    while True:
        if x + width < size:
            yield x, x + width
        elif x + width == size:
            yield x, x + width
            return
        else:
            yield size - width, size
            return
        x += stride

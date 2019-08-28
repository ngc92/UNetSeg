import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp


def crop_and_concat(inputs, residual_input):
    aw = inputs.shape[1]
    bw = residual_input.shape[1]
    surplus = bw - aw

    bbox_begin = tf.stack([0, surplus // 2, surplus // 2, 0])
    bbox_size = tf.stack([-1, aw, aw, -1])

    cropped = tf.slice(residual_input, bbox_begin, bbox_size)
    cropped.set_shape([None, aw, aw, residual_input.get_shape()[3]])

    return tf.concat([inputs, cropped], axis=-1)


# https://stackoverflow.com/questions/52012657/how-to-make-a-2d-gaussian-filter-in-tensorflow
def gaussian_kernel(size: int, mean: float, std: float):
    """Makes 2D gaussian Kernel for convolution."""
    tfd = tfp.distributions

    # make 1-D distribution by sampling the pdf of a normal distribution
    d = tfd.Normal(mean, std)
    vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))

    gauss_kernel = tf.einsum('i,j->ij', vals, vals)
    return gauss_kernel / tf.reduce_sum(gauss_kernel)


class GaussKernelInitializer(keras.initializers.Initializer):
    """
    An initializer that produces a Gaussian kernel with given size and standard deviation. The kernel is
    generated over the support [-size, size]².
    """
    def __init__(self, size: int, std: float):
        """
        :param size: The size of the kernel. The number of entries that should be taken into account in either direction.
        For a given size, the resulting kernel will be an `(2*size + 1) x (2*size+1)` matrix.
        :param std: The standard deviation of the kernel.
        """
        self.size = size
        self.std = std

    def __call__(self, shape, dtype=None):
        gauss_kernel = gaussian_kernel(self.size, 0.0, self.std)
        gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
        assert shape == gauss_kernel.shape
        return gauss_kernel

    def get_config(self):
        return {"size": self.size, "std": self.std}


def random_crop_stateless(value, size, seed, name=None):
    """Randomly crops a tensor to a given size.

    Slices a shape `size` portion out of `value` at a uniformly chosen offset.
    Requires `value.shape >= size`.

    If a dimension should not be cropped, pass the full size of that dimension.
    For example, RGB images can be cropped with
    `size = [crop_height, crop_width, 3]`.

    Args:
    value: Input tensor to crop.
    size: 1-D tensor with size the rank of `value`.
    seed: An integral tensor used for reproducible crops.
    name: A name for this operation (optional).

    Returns:
    A cropped tensor of the same rank as `value` and shape `size`.
    """
    # TODO(shlens): Implement edge case to guarantee output size dimensions.
    # If size > value.shape, zero pad the result so that it always has shape
    # exactly size.
    with tf.name_scope(name or "random_crop") as name:
        value = tf.convert_to_tensor(value, name="value")
        size = tf.convert_to_tensor(size, dtype=tf.int32, name="size")
        shape = tf.shape(value)
        check = tf.Assert(
            tf.reduce_all(shape >= size),
            ["Need value.shape >= size, got ", shape, size],
            summarize=1000)
        with tf.control_dependencies([check]):
            limit = shape - size + 1
            offset = tf.random.stateless_uniform(
                tf.shape(shape),
                dtype=size.dtype,
                maxval=size.dtype.max,
                seed=seed) % limit
            return tf.slice(value, offset, size, name=name)


def segmentation_error_visualization(ground_truth: tf.Tensor, segmentation: tf.Tensor,
                                     mask: tf.Tensor = None, channel: int = 0):
    ground_truth = ground_truth[..., channel]
    segmentation = segmentation[..., channel]

    red = ground_truth - segmentation
    green = (1 - tf.abs(ground_truth - segmentation)) * ground_truth
    blue = 2 * (segmentation - ground_truth)

    result = tf.stack([red, green, blue], axis=-1)
    if mask is not None:
        result = result * mask
    return tf.clip_by_value(result, 0.0, 1.0)


def make_random_field(image, magnitude, min_segments, min_segment_size, channels, blur, seed):
    h, w = tf.shape(image)[0], tf.shape(image)[1]

    lb = tf.convert_to_tensor(min_segments, dtype=tf.float32)
    ux = tf.cast(h, tf.float32) / tf.convert_to_tensor(min_segment_size, dtype=tf.float32)
    uy = tf.cast(w, tf.float32) / tf.convert_to_tensor(min_segment_size, dtype=tf.float32)

    scaling = tf.random.stateless_uniform((2,), seed=seed, minval=0.0, maxval=1.0, dtype=tf.float32)
    scale_x = tf.cast(lb + (ux - lb) * scaling[0], tf.int32)
    scale_y = tf.cast(lb + (uy - lb) * scaling[1], tf.int32)

    flow_field = tf.random.stateless_uniform((1, scale_x, scale_y, channels), seed=seed, minval=-magnitude, maxval=magnitude)
    flow_field = tf.image.resize(flow_field, (h + 2 * blur.kernel_size, w + 2 * blur.kernel_size))

    with tf.device("/cpu:0"):
        flow_field = blur(flow_field)

    return flow_field


def occlude_image(image, height, width, y, x, new_value):
    """
    Takes a single(!) image and overpaints it with a `height x width` rectangle that is places at coordinates `y, x`.
    :param image: The source image.
    :param height: Height of the painting rectangle.
    :param width: Width of the painting rectangle.
    :param y: Y coordinate of the upper left corner of the rectangle, as a float in range [0, 1]
    :param x: X coordinate of the upper left corner of the rectangle, as a float in range [0, 1]
    :param new_value: The value with which the rect will be filled.
    :return: The overpainted image.
    """
    shape = tf.shape(image)
    rh = shape[0] - height
    rw = shape[1] - width

    upper = tf.cast(tf.cast(rh, tf.float32) * y, tf.int32)
    left = tf.cast(tf.cast(rw, tf.float32) * x, tf.int32)

    mask = tf.ones((height, width, 1))
    mask = tf.pad(mask, [[upper, rh - upper], [left, rw - left], [0, 0]])

    # TODO randomly decide noise strength in mask?
    return image * (1.0 - mask) + mask * new_value

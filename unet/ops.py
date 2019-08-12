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

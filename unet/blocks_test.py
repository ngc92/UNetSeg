import pytest
import tensorflow as tf
from unet.blocks import UpBlock, DownBlock


def test_upscale_deconv_shape():
    deconv = UpBlock(4)
    upscale = UpBlock(4, use_upscaling=True)

    input_x = tf.zeros((1, 8, 8, 1))
    input_y = tf.zeros((1, 8, 8, 1))

    ds = deconv((input_x, input_y))
    us = upscale((input_x, input_y))

    print(ds.shape)
    print(us.shape)
    assert ds.shape == [1, 16, 16, 4]
    assert us.shape == [1, 16, 16, 4]


def test_downward_size_validation():
    block = DownBlock(4)
    x = tf.zeros((1, 8, 7, 1))

    @tf.function
    def block_fn():
        return block(x)

    with pytest.raises(tf.errors.InvalidArgumentError):
        block(x)

    with pytest.raises(tf.errors.InvalidArgumentError):
        block_fn()

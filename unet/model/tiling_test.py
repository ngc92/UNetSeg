import pytest
from unet.model.tiling import *


@pytest.mark.parametrize("stride", [1, 2, 3, 7, 10, 11])
@pytest.mark.parametrize("width", [1, 2, 3, 7, 10, 11])
def test_strided_iteration_generic(stride, width):
    counter = 0
    # test width of elements
    for a, b in strided_iteration(100, stride, width):
        if counter == 0:
            assert a == 0

        assert b == a + width
        counter += 1
        assert counter <= 100

    # test final element
    assert b == 100


def test_strided_iteration_example():
    expected = [(0, 3), (2, 5), (4, 7), (6, 9), (7, 10)]
    assert list(strided_iteration(10, 2, 3)) == expected


########################################################################################################################
#   Tests for StridedPredictor
########################################################################################################################


def make_mock_predictor(channels, border, use_conv=False):
    class OnePredictor:
        def __init__(self):
            self.channels = channels
            self.border_width = border
            if use_conv:
                self.cv = tf.keras.layers.Conv2D(channels, 2 * border + 1)

        def input_mask_to_output_mask(self, x):
            pool_size = 1 + 2 * border
            return 1.0 - tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=1)(1.0 - x)

        def is_valid_input_size(self, input_size):
            return True

        def __call__(self, x):
            b, w, h, c = x.shape
            if use_conv:
                return self.cv(x)
            else:
                return tf.ones((b, w-2*border, h-2*border, channels))

    return OnePredictor()


def test_tiling_invalid_size():
    base = make_mock_predictor(3, 2)
    with pytest.raises(ValueError):
        TiledPredictor(base, 4)

    base.is_valid_input_size = lambda x: False
    with pytest.raises(ValueError):
        TiledPredictor(base, 12)


@pytest.mark.parametrize("border", [0, 1])
@pytest.mark.parametrize("height", [9, 12])
def test_tiling_shape(border, height):
    base = make_mock_predictor(3, border)
    predictor = TiledPredictor(base, 4)
    source = tf.zeros((height, 9, 1))
    prediction = predictor(source)

    assert prediction.shape == (height - 2*border, 9 - 2*border, 3)


@pytest.mark.parametrize("patch", [6, (6, 5)])
@pytest.mark.parametrize("border", [0, 1, 2])
def test_tiling_ones(patch, border):
    base = make_mock_predictor(3, border)
    predictor = TiledPredictor(base, patch_shape=patch)
    source = tf.zeros((13, 9, 1))
    prediction = predictor(source)

    assert pytest.approx(prediction.numpy()) == 1.0


def test_tiling_conv():
    """
    Use a simple convolutional model here for testing. This will hopefully show if the stitching togther
    does not work correctly.
    """
    base = make_mock_predictor(3, 1, use_conv=True)
    predictor = TiledPredictor(base, 6)
    source = tf.zeros((13, 9, 1))

    prediction = predictor(source)
    reference = base.cv(source[None, ...])[0]

    assert prediction.shape == reference.shape
    assert pytest.approx(prediction.numpy()) == reference.numpy()


def test_weighting_pattern():
    """
    If there is no border, and the image size is a multiple of the patch size,
    the weighting should be a constant.
    """
    base = make_mock_predictor(1, 0)
    predictor = TiledPredictor(base, 6)
    source = tf.zeros((12, 12, 1))

    _, w = predictor._apply_to_single_image(source, return_weight=True)

    assert w.shape == (12, 12, 1)
    assert pytest.approx(w.numpy()) == 1.0


# TODO implement and test batching

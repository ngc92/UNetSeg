import numpy as np
import tensorflow as tf
import pytest
from unet.model import _get_input_size, _get_output_size, _get_border_size, UNetModel


def test_size_determination():
    assert _get_input_size(572, 4) == 572
    assert _get_input_size(573, 4) == 572
    assert _get_input_size(571, 4, crop=False) == 572

    assert _get_output_size(572, 4) == 388

    assert _get_border_size(4) == (572 - 388) / 2


def test_predict_sizes():
    model = UNetModel(1, 1)

    # check default size
    result = model.predict(tf.zeros((1, 572, 572, 1)), False)
    assert result.shape == (1, 388, 388, 1)

    # check batching
    result = model.predict(tf.zeros((572, 572, 1)), False)
    assert result.shape == (388, 388, 1)

    # check padding in trivial setting
    result = model.predict(tf.zeros((1, 388, 388, 1)), True)
    assert result.shape == (1, 388, 388, 1)

    # check non-trivial padding
    result = model.predict(tf.zeros((1, 386, 386, 1)), True)
    assert result.shape == (1, 386, 386, 1), result.shape

    # check tiling
    result = model.predict(tf.zeros((1, 574, 574, 1)), False)
    assert result.shape == (1, 390, 390, 1), result.shape


def test_tiling_consistency():
    model = UNetModel(1, 1)
    inputs = tf.random.uniform((1, 316, 316, 1))
    reference = model(inputs).numpy()[0, :, :, 0]
    tiled = model._tiled_prediction(inputs).numpy()[0, :, :, 0]

    differences = np.abs(reference - tiled) > np.abs(reference) * 1e-3
    assert len(differences) == 0

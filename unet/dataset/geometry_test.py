from unittest import mock

import pytest
import tensorflow as tf

from unet.dataset.geometry import MaybeTransform, GeometryEquivariance


def test_maybe_apply():
    mt = MaybeTransform(tf.ones_like)

    assert mt(tf.zeros(0), 0).numpy() == pytest.approx(0)
    assert mt(tf.zeros(0), 1).numpy() == pytest.approx(1)


def test_equivariance():
    class MockEquiv(GeometryEquivariance):
        def __init__(self):
            super().__init__(1)
            self.transform = mock.Mock()
            self.inverse = mock.Mock()

    me = MockEquiv()
    image = mock.Mock()
    me.image_transform(image, 1, 5)
    me.transform.assert_called_with(image, 1, 5)

    seg = mock.Mock()
    me.segmentation_transform(seg, 1, 5)
    me.transform.assert_called_with(seg, 1, 5)
    me.inverse_segmentation_transform(seg, 1, 5)
    me.inverse.assert_called_with(seg, 1, 5)

    mask = mock.Mock()
    me.segmentation_transform(mask, 1, 5)
    me.transform.assert_called_with(mask, 1, 5)
    me.inverse_mask_transform(mask, 1, 5)
    me.inverse.assert_called_with(mask, 1, 5)

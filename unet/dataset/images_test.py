import pytest
from unittest import mock
import unet.dataset.images as images


def test_load_images():
    source = ("a", "b", "c")
    channels = (3, 2, 1)

    with mock.patch("unet.dataset.images.load_image") as li:
        images.load_images(source, channels)

    assert li.call_args_list[0] == mock.call('a', 3)
    assert li.call_args_list[1] == mock.call('b', 2)
    assert li.call_args_list[2] == mock.call('c', 1)


def test_filename_transformer():
    # check positional
    f = images.filename_transformer("{:d}.png", "{:d}.jpg")
    assert f("5.png") == "5.jpg"

    # check combining positional and named
    f = images.filename_transformer("{name}_{:d}.png", "{:d}_{name}.jpg")
    assert f("test_5.png") == "5_test.jpg"

    # check ignore
    f = images.filename_transformer("{name}_{:d}.png", "{:d}.jpg")
    assert f("test_5.png") == "5.jpg"

    # check non-match
    with pytest.raises(ValueError):
        f("nonumber.png")

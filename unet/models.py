import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

from unet.blocks import DownBlock, Bottleneck, UpBlock, OutputBlock


class UNetModel(keras.Model):
    def __init__(self, n_classes, filters=64, use_upscaling=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._down_blocks = [DownBlock(filters=filters, name="input")]
        self._up_blocks = []

        # FUCKING PYTHON SCOPING: do !NOT! call the loop variable filters!
        for num_filters in [2*filters, 4*filters, 8*filters]:
            down_block = DownBlock(filters=num_filters, name="down_%d" % num_filters)
            self._down_blocks.append(down_block)

        self._bottleneck = Bottleneck(filters=16*filters, use_upscaling=use_upscaling)

        for num_filters in [8*filters, 4*filters, 2*filters]:
            self._up_blocks.append(UpBlock(filters=num_filters, name="up_%d" % num_filters, use_upscaling=use_upscaling))

        self._out_block = OutputBlock(filters=filters, n_classes=n_classes)

        # field of view size
        self.fov_size = 572

        # segmentation size
        self.seg_size = 388

    def call(self, inputs, training=None, mask=None):
        skip_connections = []
        x = inputs
        for block in self._down_blocks:
            x, skip = block(x)
            skip_connections.append(skip)

        x = self._bottleneck(x, training=training)
        for block in self._up_blocks:
            x = block((x, skip_connections.pop()))

        return self._out_block((x, skip_connections.pop()))

    def predict(self, image, padding=False):
        return predict_tiled(self, image, self.fov_size, self.seg_size, padding=padding)


def predict_tiled(model, image, fov_size, pred_size, padding=False):
    border = (fov_size - pred_size) // 2

    # do we need to resize?
    min_size = min(image.shape[0], image.shape[1])
    if padding:
        min_size += border

    if min_size < fov_size:
        if padding:
            resize_factor = (fov_size - border) / (min_size - border)
        else:
            resize_factor = fov_size / min_size

        nw, nh = int(np.ceil(image.shape[0] * resize_factor)), int(np.ceil(image.shape[1] * resize_factor))
        image = tf.image.resize(image, (nw, nh))

    if padding:
        image = tf.pad(image, [[border, border], [border, border], [0, 0]])
    width, height = image.shape[0], image.shape[1]

    assert width >= fov_size
    assert height >= fov_size

    segmentation = np.zeros((width, height))
    x = 0
    while x + fov_size <= width:
        y = 0
        while y + fov_size <= height:
            glimpse = image[x:x+fov_size, y:y+fov_size]
            seg = model(glimpse[None, ...])[0]
            segmentation[x+border:x+border+pred_size, y+border:y+border+pred_size] = seg[..., 0]
            if y + fov_size == height:
                break
            else:
                y += pred_size

            if y + fov_size > height:
                y = height - fov_size

        if x + fov_size == width:
            break
        else:
            x += pred_size

        if x + fov_size > width:
            x = width - fov_size

    return segmentation[border:-border, border:-border]

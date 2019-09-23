import tensorflow as tf
from tensorflow import keras

from unet.dataset import *
from unet.dataset.images import SegmentationDataset
from unet.model import UNetModel
from unet.training import default_unet_trainer
from unet.tools.inspect_ui import InspectUI


model = UNetModel(1, use_upscaling=True)

SETTING = "one"

trainer = default_unet_trainer(model, SETTING)
trainer.restore()


def prep(img):
    size = 572
    return tf.image.convert_image_dtype(
        tf.image.resize_with_crop_or_pad(img, size, size),
        dtype=tf.float32)


def data_loader(path):
    image = load_image(path, 1)
    return prep(image)


InspectUI.ui_main_loop(model, data_loader)

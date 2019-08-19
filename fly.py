import tensorflow as tf
from tensorflow import keras

from unet.augmentation import *
from unet.model import UNetModel
from unet.training import default_unet_trainer


model = UNetModel(1, use_upscaling=True)

trainer = default_unet_trainer(model)

ckp = tf.train.Checkpoint(trainer=trainer)
ckp.restore("ckp/trainer-1")

#pipeline = AugmentationPipeline([HorizontalFlips(), VerticalFlips(), Rotation90Degrees(), FreeRotation(),
#                              Warp(1.0, 10.0, blur_size=5),
#                              NoiseInvariance(0.2), ContrastInvariance(0.5, 1.1), BrightnessInvariance(0.2)])
#pipeline.test_for_image("data/train/original/000.png", "data/train/segmentation/000.png", "/tmp/test", 10)
#exit()

dataset = AugmentationPipeline.images_from_directories(
    "data/train/original",
    "data/train/segmentation",
    channels_in=1, channels_out=1
)


def padding(image, segmentation):
    mask = tf.ones_like(image)
    segmentation = keras.layers.MaxPool2D(3, strides=1, padding="same")(segmentation[None, ...])[0]
    return (tf.image.resize_with_crop_or_pad(image, 588, 588),
            tf.image.resize_with_crop_or_pad(segmentation, 588, 588),
            tf.image.resize_with_crop_or_pad(mask, 588, 588))


dataset = dataset.map(padding)
for _ in range(10):
    trainer.train_epoch(dataset)
ckp.save("ckp/trainer")

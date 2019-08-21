import tensorflow as tf
from tensorflow import keras

from unet.dataset import *
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


def prep(img):
    size = 572  # model.input_size_for_output(512)
    return tf.image.convert_image_dtype(
        tf.image.resize_with_crop_or_pad(img, size, size),
        dtype=tf.float32)


def padding(image, segmentation):
    mask = tf.ones_like(image, dtype=tf.float32)
    segmentation = keras.layers.MaxPool2D(3, strides=1, padding="same")(segmentation[None, ...])[0]
    return prep(image), prep(segmentation), prep(mask)


dataset = AugmentationPipeline.images_from_directories(
    "data/train/original",
    "data/train/segmentation",
    channels_in=1, channels_out=1
)
dataset = dataset.map(padding)
eval_data = AugmentationPipeline.images_from_directories(
    "data/eval/original",
    "data/eval/segmentation",
    channels_in=1, channels_out=1
)
eval_data = eval_data.map(padding)


for _ in range(10):
    trainer.train_epoch(dataset)
    trainer.evaluate(eval_data)
    print(_)
ckp.save("ckp/trainer")

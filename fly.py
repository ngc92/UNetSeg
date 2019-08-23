import tensorflow as tf
from tensorflow import keras
from easydict import EasyDict

from unet.dataset import *
from unet.dataset.images import filename_transformer
from unet.model import UNetModel
from unet.training import default_unet_trainer


model = UNetModel(1, use_upscaling=True)

SETTING = "one"

trainer = default_unet_trainer(model, SETTING)
trainer.restore()


def get_dataset(spec):
    spec = EasyDict(spec)
    return AugmentationPipeline.images_from_directories(
        source_dir=spec.original.directory,
        channels_in=spec.original.channels,
        segmentation_dir=spec.segmentation.directory,
        channels_out=spec.segmentation.channels,
        pattern=spec.pattern
    )


def prep(img):
    size = 572  # model.input_size_for_output(512)
    return tf.image.convert_image_dtype(
        tf.image.resize_with_crop_or_pad(img, size, size),
        dtype=tf.float32)


def padding(image, segmentation):
    mask = tf.ones_like(image, dtype=tf.float32)
    segmentation = keras.layers.MaxPool2D(3, strides=1, padding="same")(segmentation[None, ...])[0]
    return prep(image), prep(segmentation), prep(mask)


pattern = "000.png" if SETTING == "one" else "*.png"
dataset = AugmentationPipeline.images_from_directories(
    "data/train/original",
    "data/train/segmentation",
    pattern=pattern,
    channels_in=1, channels_out=1
)
dataset = dataset.map(padding)
eval_data = AugmentationPipeline.images_from_directories(
    "data/eval/original",
    "data/eval/segmentation",
    channels_in=1, channels_out=1
)
eval_data = eval_data.map(padding)

wingdisk_mapping = filename_transformer("wingdisk_grey_{:03d}.png", "wingdisk_seg_{:03d}.png")
test_data = AugmentationPipeline.images_from_directories(
    "data/DahmannGroup/Wingdisc/wingdisk_org/wingdisk_grey",
    "data/DahmannGroup/Wingdisc/wingdisk_seg/wingdisk_seg_grey",
    channels_in=1, channels_out=1, name_transform=wingdisk_mapping
)
test_data = test_data.map(padding)


# data for unsupervised training
def prep_unsup(x):
    image = load_image(x, 1)
    mask = tf.ones_like(image, dtype=tf.float32)
    return prep(image), prep(mask)


if SETTING == "unsupervised":
    unsup_data = tf.data.Dataset.list_files("data/DahmannGroup/Wingdisc/wingdisk_org/wingdisk_grey/*.png").map(prep_unsup)
else:
    unsup_data = None


if SETTING == "one":
    dataset = dataset.repeat(144)


while trainer.epoch < 15:
    trainer.train_epoch(dataset, unsupervised_data=unsup_data)
    trainer.evaluate(eval_data)
    trainer.evaluate(test_data, tag="test")
    print(trainer.summary_dict)
    trainer.save()

trainer._evaluate(test_data)
print(trainer.summary_dict)

tf.keras.Model
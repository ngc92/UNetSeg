import tensorflow as tf

from unet.augmentation import AugmentationPipeline
from unet.model import UNetModel
from unet.training import default_unet_trainer


model = UNetModel(1, use_upscaling=True)

trainer = default_unet_trainer(model)

dataset = AugmentationPipeline.images_from_directories(
    "data/train/original",
    "data/train/segmentation",
    channels_in=1, channels_out=1
)


def padding(image, segmentation):
    mask = tf.ones_like(image)
    return (tf.image.resize_with_crop_or_pad(image, 588, 588),
            tf.image.resize_with_crop_or_pad(segmentation, 588, 588),
            tf.image.resize_with_crop_or_pad(mask, 588, 588))


dataset = dataset.map(padding)
for _ in range(10):
    trainer.train_epoch(dataset)

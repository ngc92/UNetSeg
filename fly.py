import tensorflow as tf

from unet.augmentation import AugmentationPipeline
from unet.model import UNetModel
from unet.training import default_unet_trainer


model = UNetModel(1, use_upscaling=True)

trainer = default_unet_trainer(model)

dataset = AugmentationPipeline.images_from_directories(
    "/home/erik/Desktop/DahmannGroup/Wingdisc/wingdisk_org/wingdisk_grey",
    "/home/erik/Desktop/DahmannGroup/Wingdisc/wingdisk_seg/wingdisk_seg_grey",
    channels_in=1, channels_out=1
)


def padding(image, segmentation):
    return tf.image.resize_with_crop_or_pad(image, 588, 588), tf.image.resize_with_crop_or_pad(segmentation, 588, 588)


dataset = dataset.map(padding)
trainer.train_epoch(dataset)

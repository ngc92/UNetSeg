from abc import ABCMeta, abstractmethod

import tensorflow as tf
from tensorflow import keras


class SegmentationModel(keras.Model, metaclass=ABCMeta):
    def __init__(self, channels, normalize_input=True, *args, **kwargs):
        """
        :param channels: Number of channels in the output image.
        :param args: Arguments passed on to `keras.Model`.
        :param kwargs: Arguments passed on to `keras.Model`.
        """
        super().__init__(*args, **kwargs)
        self._n_channels = channels
        self._normalize_input = normalize_input
    
    @property
    def channels(self):
        return self._n_channels

    @abstractmethod
    def logits(self, inputs, training):
        """
        Applies the U-Net to the input image and returns the resulting logits.
        :param inputs: A batch of images
        :param training: Whether to operate in training or inference mode. Activates dropout in the bottleneck layer.
        :param mask: TODO would this even work?
        :return: The segmented image. Note that this is smaller than the input image.
        """
        pass

    def logits_to_prediction(self, logits):
        if self._n_channels == 1:
            return keras.activations.sigmoid(logits)
        else:
            return keras.activations.softmax(logits)

    def call(self, inputs, training=None):
        """
        Applies the U-Net to the input image.
        :param inputs: A batch of images
        :param training: Whether to operate in training or inference mode. Activates dropout in the bottleneck layer.
        :return: The segmented image. Note that this is smaller than the input image.
        """
        if self._normalize_input:
            inputs = tf.image.per_image_standardization(inputs)

        logits = self.logits(inputs, training=training)
        return self.logits_to_prediction(logits)

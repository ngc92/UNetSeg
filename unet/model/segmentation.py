from abc import ABCMeta, abstractmethod

import tensorflow as tf
from tensorflow import keras


class SegmentationModel(keras.Model, metaclass=ABCMeta):
    """
    Base class for segmentation models. These are image-to-image mappings
    where the last layer is a classification. The network may accept inputs
    only for specific sizes, and the size of the output image may be different
    from the input image. In particular, an input image `[0, h]×[0, w]` can be
    mapped to a segmentation `[b, h-b]×[b, w-b]`. In this case `b` is given
    by `SegmentationModel.border_width`.
    """

    def __init__(self, channels, normalize_input=True, *args, **kwargs):
        """
        :param channels: Number of channels in the output image.
        :param normalize_input: Whether input images will be normalized before being passed to the network.
        :param args: Arguments passed on to `keras.Model`.
        :param kwargs: Arguments passed on to `keras.Model`.
        """
        super().__init__(*args, **kwargs)
        self._n_channels = channels
        self._normalize_input = normalize_input
    
    @property
    def channels(self) -> int:
        """Number of output channels"""
        return self._n_channels

    @property
    def border_width(self) -> int:
        """
        Size of the border in the input image for which no output will be produced. If the segmentation
        has the same scale as the input, then a border of `b` means that for an input image defined on `[0, h] x [0, w]`
        the output is only defined on `[b, h-b] x [b, w-b]`.
        Such a situation can easily arise if the network uses convolutions with `"VALID"` padding.
        :return: Border width in pixels.
        """
        raise NotImplementedError()

    @abstractmethod
    def is_valid_input_size(self, input_size):
        """
        Checks whether the given shape is a valid input size for this network.
        :param input_size: A 2-tuple containing the height and width of the input image.
        :return: True, if the network can process the given input.
        """
        raise NotImplementedError()

    @abstractmethod
    def logits(self, inputs, training):
        """
        Applies the U-Net to the input image and returns the resulting logits.
        :param inputs: A batch of images
        :param training: Whether to operate in training or inference mode. Activates dropout in the bottleneck layer.
        :return: The segmented image. Note that this is smaller than the input image.
        """
        pass

    @abstractmethod
    def input_mask_to_output_mask(self, input_mask: tf.Tensor):
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

import tensorflow as tf
import imageio
import pathlib
import logging

_logger = logging.getLogger(__name__)

# TODO figure out how we can make zoom/crop work sensibly.


class AugmentationPipeline:
    def __init__(self, transformations: list):
        self._transformations = transformations or []

    def add_transformation(self, trafo):
        self._transformations.append(trafo)

    @tf.function
    def augment(self, source: tf.Tensor, segmentation: tf.Tensor, mask: tf.Tensor = None):
        assert source.dtype.is_floating, source.dtype
        assert segmentation.dtype.is_floating, segmentation.dtype

        if mask is None:
            mask = tf.ones_like(segmentation)

        for trafo in self._transformations:
            source, segmentation, mask = trafo.augment(source, segmentation, mask)
        return source, segmentation, mask

    @tf.function
    def augment_batch(self, source: tf.Tensor, segmentation: tf.Tensor, mask: tf.Tensor = None):
        if mask is None:
            mask = tf.ones_like(segmentation)

        return tf.map_fn(lambda x: self.augment(*x), (source, segmentation, mask), back_prop=False)

    def test_for_image(self, source, segmentation, target_folder, num_samples: int, mask=None):
        if not isinstance(target_folder, pathlib.Path):
            target_folder = pathlib.Path(target_folder)

        if isinstance(source, (str, pathlib.Path)):
            source = tf.io.decode_image(tf.io.read_file(str(source)), dtype=tf.float32)
        if isinstance(segmentation, (str, pathlib.Path)):
            segmentation = tf.io.decode_image(tf.io.read_file(str(segmentation)), dtype=tf.float32)

        if not target_folder.exists():
            target_folder.mkdir()

        for i in range(num_samples):
            img, seg, msk = self.augment(source, segmentation, mask)
            imageio.imwrite(target_folder / ("%03d-image.png" % i), img.numpy())
            imageio.imwrite(target_folder / ("%03d-segmentation.png" % i), seg.numpy())
            imageio.imwrite(target_folder / ("%03d-mask.png" % i), msk.numpy())

    def augment_dataset(self, dataset: tf.data.Dataset):
        """
        Takes as input an image dataset consists of tuples (image, segmentation)
        (the image tensors have rank 3) or tuples (image, segmentation, mask).
        :param dataset: The dataset containing augmented images.
        :return:
        """
        def process_images(img, seg, mask=None):
            img = tf.image.convert_image_dtype(img, tf.float32)
            seg = tf.image.convert_image_dtype(seg, tf.float32)
            if mask is not None:
                mask = 1.0 - tf.cast(tf.equal(mask, 0), tf.float32)
            return self.augment(img, seg, mask)
        return dataset.map(process_images, num_parallel_calls=4)

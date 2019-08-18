import tensorflow as tf
import imageio
import pathlib
import numpy as np


"""
We consider the following types of augmentations, that arise from transformation properties of the segmentation process.
For utility, for an input image `x` we call the corresponding ground truth `y=g(x)`.
1) Invariances. A transformation `f` such that `g(x) = g(f(x))`. Examples are noise modifications, brightness and 
contrast changes. 
"""


class TransformationProperty:
    """
    This class encodes certain transformation properties of the segmentation problem. Let `x` be the input image, and
    `y = g(x)` the ground truth segmentation. A transformation property is some form of invariance given by
    `g(f(x; t)) = h(g(x); t)` with a parameter `t`. By convention, a parameter of `t=0` indicates that `g, h` are
    identity mappings. We can classify the transformations based on the possible values of `t`. It is either a
    continuous symmetry, when t is from the interval `[0, 1]`, or a discrete symmetry, where t is from `{0, 1, ..., k}`.
    For implementation purposes, we allow "random" operations, which get an additional integer seed passed, which is the
    same for the image and the segmentation transformation.

    A further problem arises is the mapping `h` is not bijective. This happens e.g. for rotations that are not multiples
    of 90°. In that case a binary mask is needed to determine which pixels are valid and which are artifacts. For that
    reason, and additional `mask_transform` function needs to be defined. It is assumed that masking does not depend on
    the actual contents of `image` and `segmentation`, but only on the shape.
    """
    def __init__(self, num_discrete: int):
        """
        :param num_discrete: The number of discrete transformations represented with this property. Set to 0 for
        continuous transformations.
        """
        self._num_discrete = num_discrete

    def image_transform(self, image: tf.Tensor, argument: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()

    def segmentation_transform(self, segmentation: tf.Tensor, argument: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()

    def mask_transform(self, mask: tf.Tensor, argument: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()

    def inverse_segmentation_transform(self, segmentation: tf.Tensor, argument: tf.Tensor,
                                       seed: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()

    def inverse_mask_transform(self, mask: tf.Tensor, argument: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()

    def augment(self, source: tf.Tensor, segmentation: tf.Tensor, mask: tf.Tensor):
        # verify inputs
        source.shape.assert_has_rank(3)

        choice = self._get_argument()
        seed = tf.random.uniform((2,), 0, tf.int32.max, dtype=tf.int32)

        return (self.image_transform(source, argument=choice, seed=seed),
                self.segmentation_transform(segmentation, argument=choice, seed=seed),
                self.mask_transform(mask, argument=choice, seed=seed))

    def boost(self, source: tf.Tensor, mask: tf.Tensor):
        # verify inputs
        source.shape.assert_has_rank(3)

        choice = self._get_argument()
        seed = tf.random.uniform((), 0, tf.int32.max, dtype=tf.int32)

        def back_trafo(segmentation, mask):
            return (self.segmentation_transform(segmentation, argument=choice, seed=seed),
                    self.mask_transform(mask, argument=choice, seed=seed))

        return (self.image_transform(source, argument=choice, seed=seed),
                self.mask_transform(mask, argument=choice, seed=seed),
                back_trafo)

    def _get_argument(self):
        if self._num_discrete > 0:
            choice = tf.random.uniform((), 0, self._num_discrete + 1, dtype=tf.int32)
        else:
            choice = tf.random.uniform((), 0.0, 1.0)
        return choice

# TODO figure out how we can make zoom/crop work sensibly.


class AugmentationPipeline:
    def __init__(self, transformations):
        self._transformations = transformations or []

    def add_transformation(self, trafo):
        self._transformations.append(trafo)

    @tf.function
    def augment(self, source: tf.Tensor, segmentation: tf.Tensor, mask: tf.Tensor = None):
        if mask is None:
            mask = tf.ones_like(segmentation)

        for trafo in self._transformations:
            source, segmentation, mask = trafo.augment(source, segmentation, mask)
        return source, segmentation, mask

    def test_for_image(self, source, segmentation, target_folder, num_samples: int, mask=None):
        if not isinstance(target_folder, pathlib.Path):
            target_folder = pathlib.Path(target_folder)

        if not target_folder.exists():
            target_folder.mkdir()

        for i in range(num_samples):
            img, seg, msk = self.augment(source, segmentation, mask)
            imageio.imwrite(target_folder / ("%03d-image.png" % i), img.numpy())
            imageio.imwrite(target_folder / ("%03d-segmentation.png" % i), seg.numpy())
            imageio.imwrite(target_folder / ("%03d-mask.png" % i), msk.numpy())

    def augmented_dataset(self, source_images, segmentation_images):
        dataset = self.images_from_list(source_images, segmentation_images, shuffle=True).repeat()
        return self.augment_dataset(dataset)

    @staticmethod
    def images_from_list(source_images, segmentation_images, shuffle=True, channels_in=3, channels_out=3):
        """
        Given a list of source images and a list of corresponding ground truth segmentations,
        builds a `tf.data.Dataset` out of these image pairs.
        :param source_images: List of paths to the source images.
        :param segmentation_images: List of paths to the ground truth segmentations.
        :param shuffle: Whether to shuffle the dataset.
        :param channels_in: Number of channels in the input image.
        :param channels_out: Number of channels in the segmentation.
        :return: A dataset of pairs `(image, ground_truth)`.
        """
        source_images = list(map(str, source_images))
        segmentation_images = list(map(str, segmentation_images))

        assert len(source_images) == len(segmentation_images)

        dataset = tf.data.Dataset.from_tensor_slices((source_images, segmentation_images))

        def load_image(img_path, seg_path):
            image = tf.io.decode_image(tf.io.read_file(img_path), channels=channels_in)
            image.set_shape((None, None, channels_in))
            segmentation = tf.io.decode_image(tf.io.read_file(seg_path), channels=channels_out)
            segmentation.set_shape((None, None, channels_out))
            return image, segmentation

        dataset = dataset.map(load_image, num_parallel_calls=4).cache()
        if shuffle:
            return dataset.shuffle(len(source_images))
        else:
            return dataset

    @staticmethod
    def images_from_directories(source_dir, segmentation_dir, shuffle=True, pattern="*.png",
                                channels_in=3, channels_out=3):
        """
        Given a source an a segmentations folder, this function returns a `tf.data.Dataset` containing
        all pairs of images. This assumes that the filename in `source_dir` is the same as the
        corresponding filename in the `segmentation_dir`. Source files for which no segmentation file
        does not exist wil be ignored.
        :param source_dir: Directory with the input files.
        :param segmentation_dir: Directory with the ground truth segmentations.
        :param shuffle: Whether to shuffle the image. Otherwise they are sorted by file name.
        :param pattern: File name pattern to select image files. Defaults to `*.png` for png files.
        :param channels_in: Number of channels in the input image.
        :param channels_out: Number of channels in the segmentation.
        :return: A dataset of image pairs, as per `images_from_list`.
        """
        source_dir = pathlib.Path(source_dir)
        seg_dir = pathlib.Path(segmentation_dir)

        sources = []
        segmentations = []
        for source in sorted(source_dir.glob(pattern)):
            segmentation_image = (seg_dir / source)
            if not segmentation_image.exists():
                continue
            sources.append(source)
            segmentations.append(segmentation_image)

        return AugmentationPipeline.images_from_list(sources, segmentations, shuffle=shuffle,
                                                     channels_in=channels_in, channels_out=channels_out)

    def augment_dataset(self, dataset: tf.data.Dataset):
        """
        Takes as input an image dataset consists of tuples (image, segmentation)
        (the image tensors have rank 3) or tuples (image, segmentation, mask).
        :param dataset: The dataset containing augmented images.
        :return:
        """
        def process_images(img, seg):
            img = tf.image.convert_image_dtype(img, tf.float32)
            seg = tf.image.convert_image_dtype(seg, tf.float32)
            return self.augment(img, seg)
        return dataset.map(process_images, num_parallel_calls=4)

"""
class GeometricAugmentation:
    def __init__(self, rotate: bool = True, flip=True, free_rotation: bool = True, scale_factor: float = 0.5,
                 warping: WarpLayer = None):
        self.rotate = rotate
        self.free_rotation = free_rotation
        self.flip = flip
        self.scale_factor = scale_factor

        self._warp_layer = warping

    def augment(self, source_image: tf.Tensor, segmentation: tf.Tensor):
        source_image.shape.assert_has_rank(3)

        assert_compatible = tf.debugging.assert_equal(tf.shape(source_image), tf.shape(segmentation))
        with tf.control_dependencies([assert_compatible]):
            img, seg = tf.identity(source_image), tf.identity(segmentation)

        if self.rotate and self.free_rotation:
            rangle = tf.random.uniform((), minval=0.0, maxval=2 * np.pi)
            img = tfa.image.rotate(img, rangle)
            seg = tfa.image.rotate(seg, rangle)

        h, w = tf.shape(source_image)[0], tf.shape(source_image)[1]
        zoom_x = tf.random.uniform((), minval=tf.cast(h*self.scale_factor, tf.int32), maxval=h, dtype=tf.int32)
        zoom_y = tf.random.uniform((), minval=tf.cast(w*self.scale_factor, tf.int32), maxval=w, dtype=tf.int32)

        # cannot stack the batch dimension as segmentation and image might have different number of channels
        stacked = tf.concat([img, seg], axis=2)
        stacked = tf.image.random_crop(stacked, (zoom_x, zoom_y, tf.shape(stacked)[-1]))
        if self._warp_layer:
            stacked = self._warp_layer(stacked)

        if self.rotate and not self.free_rotation:
            k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
            stacked = tf.image.rot90(stacked, k)

        if self.flip is True or self.flip == "lr":
            stacked = tf.image.random_flip_left_right(stacked)
        if self.flip is True or self.flip == "ud":
            stacked = tf.image.random_flip_up_down(stacked)

        # move channels back to last index
        seg = stacked[..., 3:]
        img = stacked[..., 0:3]

        img, seg = self.prepare_images(img, seg)
        return img, seg
"""

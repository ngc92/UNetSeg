from typing import Optional, TYPE_CHECKING
import pathlib
import functools
from easydict import EasyDict

from tensorflow import keras
from unet.augmentation import AugmentationPipeline
from unet.augmentation.invariances import LocalBrightnessInvariance
from unet.ops import segmentation_error_visualization
import tensorflow as tf
if TYPE_CHECKING:
    from unet.model import UNetModel


class UNetTrainer(tf.Module):
    def __init__(self, model: "UNetModel", optimizer: keras.optimizers.Optimizer,
                 summary_dir: pathlib.Path = None, symmetries=None):
        super().__init__()
        self._model = model
        self._augmenter = AugmentationPipeline(symmetries)
        self._optimizer = optimizer

        self._loss_functions = {}

        self._seg_metrics = [keras.metrics.BinaryAccuracy(name="accuracy"),
                             keras.metrics.Precision(name="precision"),
                             keras.metrics.Recall(name="recall")
        ]
        self._precision_metric = keras.metrics.Precision()
        self._recall_metric = keras.metrics.Recall()
        self._metrics = {"loss/total": keras.metrics.Mean("total_loss")}
        for m in self._seg_metrics:
            self._metrics["segmentation/" + m.name] = m

        self._train_step = tf.Variable(0, trainable=False, name="global_step")
        self._num_epoch = tf.Variable(0, dtype=tf.int64, trainable=False, name="epoch")

        if summary_dir is None:
            summary_dir = pathlib.Path(".logs/unet")
        else:
            summary_dir = pathlib.Path(summary_dir)

        self._summary_writers = EasyDict()
        self._summary_writers.train = tf.summary.create_file_writer(str(summary_dir / 'train'))
        self._summary_writers.eval = tf.summary.create_file_writer(str(summary_dir / 'eval'))

    def add_loss(self, key: str, loss: callable):
        self._loss_functions["loss/" + key] = loss
        self._metrics["loss/" + key] = keras.metrics.Mean(name=key)

    def train_epoch(self, dataset, num_steps=None):
        """
        Trains of `dataset` for one epoch, which is either until the end of `dataset` or for `num_steps`.
        :param dataset: The dataset on which to train.
        :param num_steps: The number of training steps, if `dataset` is infinite.
        """
        if tf.data.experimental.cardinality(dataset) == tf.data.experimental.INFINITE_CARDINALITY and num_steps is None:
            raise ValueError("Need `num_steps` when given an infinite dataset.")

        with self._summary_writers.train.as_default():
            self._train_epoch(dataset, num_steps)

    def _train_epoch(self, dataset, num_steps=None):
        self._reset_metrics()

        augmented_data = self._augmenter.augment_dataset(dataset).batch(1).prefetch(1)
        for i, data in enumerate(augmented_data):
            loss, seg = self.train_step(*data)
            if i == 0:
                self._record_images(*data, seg)

            if num_steps is not None and i > num_steps:
                break

        # epoch summaries. increase epoch counter first so that train_epoch/evaluate have consistent epoch numbers
        self._num_epoch.assign_add(1)
        self.summarize(self._num_epoch)

    def evaluate(self, dataset):
        with self._summary_writers.eval.as_default():
            self._evaluate(dataset)

    def _evaluate(self, dataset):
        self._reset_metrics()
        for i, data in enumerate(dataset.batch(2)):
            image, ground_truth, mask = data
            logits, mask = self._prepare_logits_and_mask(image, mask)
            losses = self.loss(ground_truth, logits, mask)
            total_loss = tf.add_n([tf.reduce_mean(l) for l in losses.values()])

            # convert logits to actual segmentation for further processing
            segmentation = self._model.logits_to_prediction(logits)

            self._record_metric("loss/total", total_loss)
            self._record_metrics(losses)
            for m in self._seg_metrics:
                m.update_state(ground_truth, segmentation, sample_weight=mask)

            if i == 0:
                self._record_images(*data, segmentation)
        self.summarize(self._num_epoch)

    def _record_images(self, image, ground_truth, mask, segmentation):
        tf.summary.image("input", image, step=self._num_epoch)
        tf.summary.image("ground_truth", ground_truth, step=self._num_epoch)
        tf.summary.image("mask", mask, step=self._num_epoch)
        tf.summary.image("segmentation", segmentation, step=self._num_epoch)

        if mask is not None:
            mask = self._model.input_mask_to_output_mask(mask)
            segmentation = tf.image.resize_with_crop_or_pad(segmentation, tf.shape(mask)[1], tf.shape(mask)[2])
            ground_truth = tf.image.resize_with_crop_or_pad(ground_truth, tf.shape(mask)[1], tf.shape(mask)[2])
        error_img = segmentation_error_visualization(ground_truth, segmentation, mask, channel=0)
        tf.summary.image("error", error_img, step=self._num_epoch)

    def _prepare_logits_and_mask(self, image, mask):
        logits = self._model.logits(image, training=False)
        logits = tf.image.resize_with_crop_or_pad(logits, tf.shape(image)[1], tf.shape(image)[2])

        if mask is None:
            mask = tf.ones_like(logits)
        else:
            mask = self._model.input_mask_to_output_mask(mask)
        mask = tf.image.resize_with_crop_or_pad(mask, tf.shape(image)[1], tf.shape(image)[2])
        return logits, mask

    @tf.function
    def train_step(self, image, ground_truth, mask):
        assert ground_truth.shape[-1] == self._model.channels, \
            "Mismatch between number of channels in " \
            "model (%d) and ground truth (%d)" % (self._model.channels, ground_truth.shape[-1])

        with tf.GradientTape() as tape:
            logits, mask = self._prepare_logits_and_mask(image, mask)
            losses = self.loss(ground_truth, logits, mask)
            total_loss = tf.add_n([tf.reduce_mean(l) for l in losses.values()])

        # convert logits to actual segmentation for further processing
        segmentation = self._model.logits_to_prediction(logits)

        self._record_metric("loss/total", total_loss)
        self._record_metrics(losses)
        for m in self._seg_metrics:
            m.update_state(ground_truth, segmentation, sample_weight=mask)

        grads = tape.gradient(total_loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))
        self._train_step.assign_add(1)

        return total_loss, segmentation

    @tf.function
    def loss(self, ground_truth: tf.Tensor, segmentation: tf.Tensor, mask: Optional[tf.Tensor]):
        return {key: loss(ground_truth, segmentation, mask) for key, loss in self._loss_functions.items()}

    def summarize(self, step=None):
        step = step or self._num_epoch

        # scalar metrics
        for key, metric in self._metrics.items():  # type: str, keras.metrics.Metric
            tf.summary.scalar(key, metric.result(), step)

        # other relevant data
        tf.summary.scalar("lr", self._optimizer.lr, step)

    @property
    def summary_dict(self):
        return {key: metric.result().numpy() for key, metric in self._metrics.items()}

    def _record_metric(self, key: str, value: tf.Tensor):
        self._metrics[key].update_state(value)

    def _record_metrics(self, values: dict):
        for k, v in values.items():
            self._record_metric(k, v)

    def _reset_metrics(self):
        for metric in self._metrics.values():
            metric.reset_states()


def default_unet_trainer(model: keras.Model, log_path: pathlib.Path = None):
    from unet.augmentation import HorizontalFlips, VerticalFlips, Rotation90Degrees, FreeRotation, Warp
    from unet.augmentation.invariances import NoiseInvariance, BrightnessInvariance, ContrastInvariance, LocalContrastInvariance
    trainer = UNetTrainer(model, keras.optimizers.Adam(0.0001), log_path,
                          symmetries=[
                              HorizontalFlips(), VerticalFlips(), Rotation90Degrees(), FreeRotation(),
                              Warp(1.0, 10.0, blur_size=5),
                              ContrastInvariance(0.7, 1.1), NoiseInvariance(0.1), BrightnessInvariance(0.1),
                              LocalContrastInvariance(0.5), LocalBrightnessInvariance(0.2)
    ])

    def xent_with_mask(ground_truth, logits, mask):
        loss = tf.nn.softmax_cross_entropy_with_logits(ground_truth, logits)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(ground_truth, logits)
        if mask is not None:
            loss = loss * mask
        return loss

    trainer.add_loss("xent", xent_with_mask)
    return trainer

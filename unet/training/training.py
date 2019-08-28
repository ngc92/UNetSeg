from typing import Optional, TYPE_CHECKING
import pathlib

from tensorflow import keras
from unet.dataset import AugmentationPipeline
from unet.ops import segmentation_error_visualization
import tensorflow as tf

from unet.training.base import TrainerBase

if TYPE_CHECKING:
    from unet.model import UNetModel


class UNetTrainer(TrainerBase):
    def __init__(self, model: "UNetModel", optimizer: keras.optimizers.Optimizer,
                 summary_dir: pathlib.Path, checkpoint_dir: pathlib.Path, symmetries=None):
        super().__init__(model=model, optimizer=optimizer, checkpoint_dir=checkpoint_dir, summary_dir=summary_dir)
        self._augmenter = AugmentationPipeline(symmetries)

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

    def add_loss(self, key: str, loss: callable):
        self._loss_functions["loss/" + key] = loss
        self._metrics["loss/" + key] = keras.metrics.Mean(name=key)

    def train_epoch(self, dataset, unsupervised_data=None, num_steps=None):
        """
        Trains of `dataset` for one epoch, which is either until the end of `dataset` or for `num_steps`.
        :param dataset: The dataset on which to train.
        :param num_steps: The number of training steps, if `dataset` is infinite.
        """
        if tf.data.experimental.cardinality(dataset) == tf.data.experimental.INFINITE_CARDINALITY and num_steps is None:
            raise ValueError("Need `num_steps` when given an infinite dataset.")

        with self.get_summary_writer("train").as_default():
            self._train_epoch(dataset, unsupervised_data, num_steps)

    def _train_epoch(self, dataset, unsupervised_data=None, num_steps=None):
        self._reset_metrics()

        augmented_data = self._augmenter.augment_dataset(dataset).batch(1).prefetch(2)
        if unsupervised_data is None:
            unsupervised_data = []
        else:
            unsupervised_data = unsupervised_data.batch(1).prefetch(2)

        for i, data in enumerate(roundrobin(augmented_data, unsupervised_data)):
            if len(data) == 3:
                loss, seg = self.train_step(*data)
            else:
                logits, _ = self._prepare_logits_and_mask(data[0], data[1])
                virtual_ground_truth = self._model.logits_to_prediction(logits)
                loss, seg = self.train_step(*self._augmenter.augment_batch(data[0], virtual_ground_truth, data[1]))
            if i == 0:
                self._record_images(*data, seg)

            if num_steps is not None and i > num_steps:
                break

        # epoch summaries. increase epoch counter first so that train_epoch/evaluate have consistent epoch numbers
        self._num_epoch.assign_add(1)
        self.summarize(self._num_epoch)

    def evaluate(self, dataset, tag="eval"):
        with self.get_summary_writer(tag).as_default():
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

        self._gradient_descent_update(tape, total_loss)

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


def default_unet_trainer(model: keras.Model, name: str, log_path: pathlib.Path = None, ckp_path: pathlib.Path = None):
    from unet.dataset import HorizontalFlips, VerticalFlips, Rotation90Degrees, FreeRotation, Warp
    from unet.dataset import NoiseInvariance, BrightnessInvariance, ContrastInvariance, LocalContrastInvariance, \
        LocalBrightnessInvariance, OcclusionInvariance

    if log_path is None:
        log_path = pathlib.Path(".logs")

    if ckp_path is None:
        ckp_path = pathlib.Path(".ckp")

    trainer = UNetTrainer(model, keras.optimizers.Adam(0.0001), summary_dir=log_path / name,
                          checkpoint_dir=ckp_path / name,
                          symmetries=[
                              HorizontalFlips(), VerticalFlips(), Rotation90Degrees(), FreeRotation(),
                              Warp(1.0, 10.0, blur_size=5),
                              ContrastInvariance(0.7, 1.1), NoiseInvariance(0.1), BrightnessInvariance(0.1),
                              LocalContrastInvariance(0.5), LocalBrightnessInvariance(0.2),
                              OcclusionInvariance(16, 32, 4)
    ])

    def xent_with_mask(ground_truth, logits, mask):
        loss = tf.nn.softmax_cross_entropy_with_logits(ground_truth, logits)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(ground_truth, logits)
        if mask is not None:
            loss = loss * mask
        return loss

    trainer.add_loss("xent", xent_with_mask)
    return trainer


def roundrobin(*iterables):
    from itertools import cycle, islice

    pending = len(iterables)
    nexts = cycle(iter(iterable).__next__ for iterable in iterables)
    while pending:
        try:
            for next in nexts: yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))

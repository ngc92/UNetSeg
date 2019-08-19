from typing import Optional, TYPE_CHECKING

from tensorflow import keras
from unet.augmentation import AugmentationPipeline
import tensorflow as tf
if TYPE_CHECKING:
    from unet.model import UNetModel


class UNetTrainer(tf.Module):
    def __init__(self, model: "UNetModel", optimizer: keras.optimizers.Optimizer, symmetries=None):
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
        self._num_epoch = tf.Variable(0, trainable=False, name="epoch")

    def add_loss(self, key: str, loss: callable):
        self._loss_functions["loss/" + key] = loss
        self._metrics["loss/" + key] = keras.metrics.Mean(name=key)

    def train_epoch(self, dataset, num_steps=None):
        augmented_data = self._augmenter.augment_dataset(dataset).batch(1)
        for i, data in enumerate(augmented_data):
            loss, seg = self.train_step(*data)
            if i == 0:
                tf.summary.image("input", data[0], step=self._num_epoch)
                tf.summary.image("ground_truth", data[1], step=self._num_epoch)
                tf.summary.image("mask", data[2], step=self._num_epoch)
                tf.summary.image("segmentation", seg, step=self._num_epoch)

            if num_steps is not None and i > num_steps:
                break

        # epoch summaries
        self.summarize(self._num_epoch)
        self._num_epoch.assign_add(1)

    @tf.function
    def train_step(self, image, ground_truth, mask):
        assert ground_truth.shape[-1] == self._model.channels, \
            "Mismatch between number of channels in " \
            "model (%d) and ground truth (%d)" % (self._model.channels, ground_truth.shape[-1])

        with tf.GradientTape() as tape:
            segmentation = self._model.logits(image, training=True)
            segmentation = tf.image.resize_with_crop_or_pad(segmentation, tf.shape(image)[1], tf.shape(image)[2])
            if mask is None:
                mask = tf.ones_like(segmentation)
            mask = tf.image.resize_with_crop_or_pad(mask, tf.shape(image)[1], tf.shape(image)[2])
            losses = self.loss(ground_truth, segmentation, mask)
            total_loss = tf.reduce_sum(tf.add_n(list(losses.values())))

            # convert logits to actual segmentation for further processing
            segmentation = self._model.logits_to_prediction(segmentation)

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

    def _record_metric(self, key: str, value: tf.Tensor):
        self._metrics[key].update_state(value)

    def _record_metrics(self, values: dict):
        for k, v in values.items():
            self._record_metric(k, v)


def default_unet_trainer(model: keras.Model):
    from unet.augmentation import HorizontalFlips, VerticalFlips, Rotation90Degrees
    trainer = UNetTrainer(model, keras.optimizers.Adam(), [
        HorizontalFlips(), VerticalFlips(), Rotation90Degrees()
    ])

    def xent_with_mask(ground_truth, logits, mask):
        loss = tf.nn.softmax_cross_entropy_with_logits(ground_truth, logits)
        if mask is not None:
            loss = loss * mask
        return loss

    trainer.add_loss("xent", xent_with_mask)
    return trainer

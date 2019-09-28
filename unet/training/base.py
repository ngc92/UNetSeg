import pathlib
import tensorflow as tf
from tensorflow import keras


class TrainerBase(tf.Module):
    """
    The base class for a training module. It keeps track of training step and epoch, manages checkpoints
    and summary writing.
    """
    def __init__(self, model: keras.Model, optimizer: keras.optimizers.Optimizer,
                 checkpoint_dir: pathlib.Path, summary_dir: pathlib.Path):
        super().__init__()
        self._model = model
        self._optimizer = optimizer

        self._train_step = tf.Variable(0, trainable=False, name="global_step")
        self._num_epoch = tf.Variable(0, dtype=tf.int64, trainable=False, name="epoch")

        self._checkpoint = tf.train.Checkpoint(trainer=self)
        self._ckpt_manager = tf.train.CheckpointManager(checkpoint=self._checkpoint, directory=str(checkpoint_dir),
                                                        max_to_keep=5, checkpoint_name="trainer")

        self._summary_dir = summary_dir
        self._summary_writers = {}

        self._metrics = {}

    def restore(self):
        if self._ckpt_manager.latest_checkpoint is not None:
            self._checkpoint.restore(self._ckpt_manager.latest_checkpoint)

    def save(self):
        self._ckpt_manager.save(checkpoint_number=self._num_epoch)

    @property
    def epoch(self):
        return self._num_epoch.value().numpy()

    @property
    def model(self):
        return self._model

    def get_summary_writer(self, tag: str) -> tf.summary.SummaryWriter:
        """
        Gets a summary writer for a given tag. If that writer does not exist yet, it will be created.
        :param tag: The tag associated with the writer. Summaries will be written to `summary_dir/tag`.
        :return: A summary writer.
        """
        if tag not in self._summary_writers:
            self._summary_writers[tag] = tf.summary.create_file_writer(str(self._summary_dir / tag))
        return self._summary_writers[tag]

    def _gradient_descent_update(self, tape, total_loss):
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self._train_step.assign_add(1)

    # metrics handling: currently only scalar metrics
    def add_metric(self, tag: str, metric):
        self._metrics[tag] = metric

    def reset_metrics(self):
        """
        Resets all tracked metrics.
        """
        for metric in self._metrics.values():
            metric.reset_states()

    def record_metric(self, key: str, value: tf.Tensor):
        """
        Record a single, named metric.
        """
        self._metrics[key].update_state(value)

    def record_metrics(self, _arg_: dict = None, **kwargs):
        """
        Record multiple metrics at ones. This is just a convenience
        method that calls `record_metric` for every entry.
        :param _arg_: A dictionary of str/Tensor type, which for each metric with the given key contains the
        corresponding update.
        :param kwargs: Any additional keyword arguments will be interpreted as further keys.
        """
        if _arg_ is not None:
            for k, v in _arg_.items():
                self.record_metric(k, v)

        if kwargs:
            self.record_metrics(kwargs)

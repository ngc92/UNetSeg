import pathlib
import tensorflow as tf
from tensorflow import keras


class TrainerBase(tf.Module):
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

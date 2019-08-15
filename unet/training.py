import tensorflow as tf
from tensorflow import keras


class UNetTrainer:
    def __init__(self, unet: keras.Model):
        self._model = unet

    @tf.function
    def train_step(self, image, ground_truth, mask):
        with tf.GradientTape() as tape:
            prediction = self._model(image)
            loss = self.calculate_loss(prediction, ground_truth, mask)


    def calculate_loss(self, prediction, ground_truth, mask):
        pass

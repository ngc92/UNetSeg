import tensorflow as tf


def multiscale_loss(image, target, depth, base_loss):
    losses = []

    for i in range(depth):
        if i != 0:
           image = tf.layers.average_pooling2d(image, 2, 2)
           target = tf.layers.average_pooling2d(target, 2, 2)

        flat_image = tf.layers.flatten(image)
        flat_target = tf.layers.flatten(target)

        loss = base_loss(flat_target, flat_image)
        losses.append(loss)

    return tf.add_n(losses)

import tensorflow as tf


def _image_format(image, data_format):
    if data_format == "channels_first":
        channel_index = 1
        image = tf.transpose(image, [0, 3, 1, 2])
    else:
        channel_index = 3
    return image, channel_index


def make_discriminator(layers, data_format="channels_last"):
    def discriminator(image, is_training=True):
        image, channel_index = _image_format(image, data_format)

        hidden = image
        features = []
        for layer in range(layers):
            hidden = tf.layers.conv2d(hidden, 64 * 2**layer, kernel_size=4, strides=2, name="conv%i" % layer,
                                      data_format=data_format)
            if layer > 0:
                hidden = tf.layers.batch_normalization(hidden, training=is_training, name="batchnorm%i" % layer,
                                                       axis=channel_index)

            hidden = tf.nn.leaky_relu(hidden)
            features += [hidden]

        final = tf.reduce_mean(hidden, [1, 2, 3])
        return final, features[1:]

    return discriminator


def discrimination_loss(logits_fake, logits_real, noise=None, scope="discriminator_loss"):
    with tf.name_scope(scope):
        if noise is None:
            target_fake = tf.zeros_like(logits_fake)
            target_real = tf.ones_like(logits_real)
        else:
            target_fake = tf.random_uniform(tf.shape(logits_fake), 0.0, 1.0)
            target_fake = tf.cast(tf.less(target_fake, noise), tf.float32)

            target_real = tf.random_uniform(tf.shape(logits_fake), 0.0, 1.0)
            target_real = tf.cast(tf.greater(target_real, noise), tf.float32)

        fake_loss = tf.losses.sigmoid_cross_entropy(target_fake, logits=logits_fake, scope="fake_loss",
                                                    reduction=tf.losses.Reduction.MEAN)
        tf.summary.scalar("fake", fake_loss)
        tf.summary.scalar("fake_p", tf.reduce_mean(tf.nn.sigmoid(logits_fake)))

        real_loss = tf.losses.sigmoid_cross_entropy(target_real, logits_real, scope="real_loss",
                                                    reduction=tf.losses.Reduction.MEAN, label_smoothing=0.1)
        tf.summary.scalar("real", real_loss)
        tf.summary.scalar("real_p", tf.reduce_mean(tf.nn.sigmoid(logits_real)))

        total = real_loss + fake_loss
        tf.summary.scalar("total", total)
    return total


def _feature_matching(fake, real, scope="feature_matching"):
    with tf.name_scope(scope):
        losses = []
        for f, r in zip(fake, real):
            mf = tf.reduce_mean(f, axis=0)
            mr = tf.reduce_mean(r, axis=0)
            loss = tf.losses.mean_squared_error(mr, mf, reduction=tf.losses.Reduction.MEAN)
            losses += [loss]
        return losses


def generation_loss(fake_logits, fake_features, real_features, feature_matching_weight, scope):
    with tf.name_scope(scope):
        # discrimination loss
        discrimination = tf.losses.sigmoid_cross_entropy(tf.ones_like(fake_logits), fake_logits,
                                                         scope="discriminate", reduction=tf.losses.Reduction.MEAN)
        tf.summary.scalar("discrimination", discrimination)

        # feature matching loss
        feature_matching = tf.add_n(_feature_matching(fake_features, real_features, "matching"))
        tf.summary.scalar("feature_matching", feature_matching)

        total = (feature_matching_weight * feature_matching + (1.0 - feature_matching_weight)*discrimination)
        tf.summary.scalar("total", total)
    return total

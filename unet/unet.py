from collections import namedtuple

import tensorflow as tf


def _image_format(image, data_format):
    if data_format == "channels_first":
        channel_index = 1
        image = tf.transpose(image, [0, 3, 1, 2])
    else:
        channel_index = 3
    return image, channel_index


def make_unet_generator(layers, features, data_format="channels_last", use_upsampling=False):
    def unet(image, is_training=True):
        image, channel_index = _image_format(image, data_format)

        def block_down(input, features, data_format):
            conv1 = tf.layers.conv2d(input, features, 3, activation=tf.nn.relu, padding="same", data_format=data_format)
            conv2 = tf.layers.conv2d(conv1, features, 3, activation=tf.nn.relu, padding="same", data_format=data_format)
            pool = tf.layers.max_pooling2d(conv2, 2, 2, data_format=data_format)
            return conv2, pool

        if use_upsampling:
            def block_up(input, cat, features, data_format):
                # deconv upscaling
                if data_format == "channels_first":
                    input = tf.transpose(input, [0, 2, 3, 1])
                up = tf.image.resize_images(input, tf.shape(input)[1:3]*2)
                if data_format == "channels_first":
                    up = tf.transpose(up, [0, 3, 1, 2])

                # concat
                up = tf.concat([up, cat], axis=1 if data_format == "channels_first" else 3)

                conv1 = tf.layers.conv2d(up, features, 3, activation=tf.nn.relu, padding="same",
                                         data_format=data_format)
                conv2 = tf.layers.conv2d(conv1, features, 3, activation=tf.nn.relu, padding="same",
                                         data_format=data_format)
                return conv2
        else:
            def block_up(input, cat, features, data_format):
                # deconv upscaling
                up = tf.layers.conv2d_transpose(input, features, 2, strides=2, activation=tf.nn.relu, padding="same",
                                                data_format=data_format)
                # concat
                up = tf.concat([up, cat], axis=1 if data_format == "channels_first" else 3)

                conv1 = tf.layers.conv2d(up, features, 3, activation=tf.nn.relu, padding="same", data_format=data_format)
                conv2 = tf.layers.conv2d(conv1, features, 3, activation=tf.nn.relu, padding="same", data_format=data_format)
                return conv2

        intermediate = []
        hidden = image
        for i in range(layers):
            conv, hidden = block_down(hidden, 2**i * features, data_format)
            intermediate += [conv]

        hidden, _ = block_down(hidden, 2 ** layers * features, data_format)

        up_results = []
        low_res = []

        for i in range(layers-1, -1, -1):
            hidden = block_up(hidden, intermediate[i], 2**i * features, data_format)
            up_results += [hidden]

            # generate low-res images from intermediate results.
            lr = tf.layers.conv2d(hidden, image.shape.as_list()[channel_index], 1, padding="same",
                                  data_format=data_format)
            if data_format == "channels_first":
                lr = tf.transpose(lr, [0, 2, 3, 1])
            low_res += [lr]

        result = low_res[-1]

        return result, low_res[:-1]

    return unet

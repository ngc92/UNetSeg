import tensorflow as tf
from im2im_records import preprocess


def crop_and_resize_image(crop_size, image_size):
    def crop_and_resize_image_(image):
        channels = image.shape[2]
        # cropping and resizing
        if crop_size == "min":
            cropping = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
        elif crop_size == "max":
            cropping = tf.maximum(tf.shape(image)[0], tf.shape(image)[1])
        else:
            cropping = crop_size

        with tf.name_scope("crop_or_pad"):
            cropped = tf.image.resize_image_with_crop_or_pad(image, cropping, cropping)
        tf.summary.image("cropped", cropped[None, :, :, :])
        with tf.name_scope("resize"):
            resized = tf.image.resize_images(cropped, [image_size, image_size])
            resized.set_shape([image_size, image_size, channels])
        tf.summary.image("resized", resized[None, :, :, :])
        return tf.cast(resized, tf.uint8)
    return preprocess.Preprocessor(crop_and_resize_image_)


def thicken():
    def thicken_(image):
        return tf.layers.max_pooling2d(tf.expand_dims(image, axis=0), 2, 1, "same")[0, :, :, :]
    return preprocess.Preprocessor(thicken_)


def make_noise_pattern(resolution, shape, seed):
    noise_pattern_low = tf.random_uniform([resolution, resolution, 1], 0, 1, seed=seed)
    noise_pattern_up = tf.image.resize_images(noise_pattern_low, shape[0:2])
    blur_fn = blur(1.0).f
    noise_pattern = tf.minimum(blur_fn(noise_pattern_up) * 0.5, 1.0)
    return noise_pattern


def augment_local_contrast(contrast_factor, seed):
    def augment_local_contrast_(image):
        low_contrast_image = tf.image.adjust_contrast(image, contrast_factor)

        # individual noise pattern
        noise_pattern = make_noise_pattern(16, tf.shape(image), seed)
        noise_factor = tf.truncated_normal((), 0.0, 0.5)
        noise_pattern *= noise_factor
        return image * (1.0 - noise_pattern) + low_contrast_image * noise_pattern

    return preprocess.Preprocessor(augment_local_contrast_)


def augment_local_brightness(delta, seed):
    def augment_local_brightness_(image):
        # individual noise pattern
        noise_pattern = make_noise_pattern(16, tf.shape(image), seed)
        noise_factor = tf.truncated_normal((), 0.0, 0.5)
        return tf.clip_by_value(image + delta * noise_pattern * noise_factor, 0.0, 1.0)

    return preprocess.Preprocessor(augment_local_brightness_)


def batched(f):
    def g(arg):
        return f(arg[None, ...])[0, ...]
    return preprocess.Preprocessor(g)


def blur(length_scale, add_and_clip=False):
    import scipy.ndimage
    import numpy as np

    def blur_fn(image):
        return scipy.ndimage.gaussian_filter(image, length_scale).astype(np.float32)

    def blur_(image):
        blurred = tf.py_func(blur_fn, [image], tf.float32, False)
        blurred.set_shape(image.shape)
        if add_and_clip:
            blurred += image
            blurred = tf.clip_by_value(blurred, 0.0, 1.0)
        return blurred

    return preprocess.Preprocessor(blur_)


def downscale(mode="avg", factor=2):
    @batched
    def downscale_(image):
        if mode == "avg":
            return tf.layers.average_pooling2d(image, factor, factor, padding="same")
        elif mode == "max":
            return tf.layers.max_pooling2d(image, factor, factor, padding="same")
    return preprocess.Preprocessor(downscale_)


def stack_images(stack_name, keys):
    def stack(data):
        images = [data[key] for key in keys]
        data[stack_name] = tf.concat(images, 2)
        return data
    return preprocess.Preprocessor(stack)


def unstack_images(stack_name, keys):
    def stack(data):
        images = tf.unstack(data[stack_name], axis=2)
        for key, image in zip(keys, images):
            data[key] = image[:, :, None]
        return data

    return preprocess.Preprocessor(stack)

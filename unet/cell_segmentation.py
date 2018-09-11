from collections import namedtuple
from im2im_records import load_tf_records, preprocess

from unet.discriminator import make_discriminator, discrimination_loss, generation_loss
from unet.loss import multiscale_loss
from unet.seg_test import segmentation_iou_op, iou_from_intersection_op
from unet.unet import make_unet_generator

from tensorboard import summary as summary_lib
import tensorflow as tf

InputConfig = namedtuple("InputConfig",
                         ["batch_size",
                          "local_augmentation",
                          "scale_factor",  # default: 2
                          "crop_size",  # default: 256
                          "use_blur",  # default: True
                          "take_only"])  # default: -1 -- all

CellSegmentationConfig = namedtuple("CellSegmentationConfig",
                                    ["num_layers",  # UNET params
                                     "num_features",
                                     "resize_up",
                                     "xent_weight",  # Loss params 0.1
                                     "mse_weight",  # 1.0
                                     "multi_scale_depth",  # 3
                                     "discrimination_loss",
                                     "feature_matching",
                                     "disc_lr",
                                     "max_steps",
                                     "disc_opt",  # either SGD or ADAM
                                     "disc_noise",  # None, or a float in (0, 0.5)
                                     ]
                                    )


class CellSegmentationModel:
    def __init__(self, checkpoint_path, icfg: InputConfig, cfg: CellSegmentationConfig):
        self._ckp_path = checkpoint_path
        self.input_config = icfg
        self.config = cfg
        self.unet_gen = make_unet_generator(cfg.num_layers, cfg.num_features, "channels_first", cfg.resize_up)

    def make_input(self, source_file, batch_size=None, reps=1, is_training=False, threads=-1):
        cfg = self.input_config

        if batch_size is None:
            batch_size = cfg.batch_size

        if threads == -1:
            import multiprocessing
            threads = multiprocessing.cpu_count()

        from unet.input import blur, downscale, augment_local_brightness, augment_local_contrast, stack_images, \
            unstack_images
        if is_training:
            consistent_trafo = preprocess.random_crop(cfg.crop_size, 32) | \
                               preprocess.random_flips(True, True) | \
                               preprocess.random_rotations()

            # stack the images together before applying transformations, to ensure that they are rotated/flipped
            # consistently
            prepare_common = stack_images("image_stack", ["A/image", "B/image"]) | \
                             consistent_trafo.apply_to("image_stack") | \
                             unstack_images("image_stack", ["A/image", "B/image"])

            prepare_original = preprocess.random_contrast(0.25, 1.1) | \
                               downscale("avg", cfg.scale_factor)

            if cfg.local_augmentation:
                prepare_original |= augment_local_contrast(0.1, 42) | augment_local_brightness(0.1, 21)

            prepare_segmented = downscale("max", cfg.scale_factor)
            if cfg.use_blur:
                prepare_segmented |= blur(1.0, True)
        else:
            prepare_original = downscale("avg", cfg.scale_factor)
            prepare_segmented = downscale("max", cfg.scale_factor)
            prepare_common = preprocess.nothing

        copy = preprocess.copy_feature("A/image", "A/original") | preprocess.copy_feature("B/image", "B/original")
        mapping_original = prepare_original.apply_to("A/image")
        mapping_segmented = prepare_segmented.apply_to("B/image")

        return load_tf_records(source_file, copy | prepare_common | mapping_original | mapping_segmented,
                               repeat_count=reps,
                               num_threads=threads, batch_size=batch_size, greyscale=True, cache=True, take=cfg.take_only)

    def make_estimator(self):
        # only save one checkpoint
        run_config = tf.estimator.RunConfig()
        run_config = run_config.replace(keep_checkpoint_max=1)
        return tf.estimator.Estimator(CellSegmentationBuilder(self).model_fn, self._ckp_path, config=run_config)

    @property
    def global_step(self):
        try:
            return self.make_estimator().get_variable_value("global_step")
        except ValueError:
            return 0

    def train(self, training_data, reps, input_threads=-1):
        input_fn = lambda: self.make_input(training_data, reps=reps, is_training=True, threads=input_threads)
        self.make_estimator().train(input_fn,
                                    max_steps=self.config.max_steps)

    def eval(self, eval_data, name=None):
        return self.make_estimator().evaluate(lambda: self.make_input(eval_data), name=name)

    def predict(self, predict_data):
        estimator = self.make_estimator()
        estimator = tf.contrib.estimator.forward_features(estimator, ["key", "A/image", "B/image"])
        yield from estimator.predict(lambda: self.make_input(predict_data, batch_size=1))


class CellSegmentationBuilder:
    def __init__(self, model: CellSegmentationModel):
        self.model = model

        self._generated = None
        self._gen_image = None
        self._bin_image = None
        self._low_res_gen = None

        self._predictions = {}
        self._evals = {}

    def _generate(self, input_image):
        generated, low_res = self.model.unet_gen(input_image)
        generated_img = tf.nn.sigmoid(generated)
        binary_image = tf.round(generated_img)

        tf.summary.histogram("original_hist", input_image)
        tf.summary.histogram("generated_hist", generated)
        tf.summary.histogram("gen_img_hist", generated_img)

        tf.summary.image("preprocessed", input_image)
        tf.summary.image("generated", generated_img)
        tf.summary.image("segmented", binary_image)

        self._bin_image = binary_image
        self._gen_image = generated_img
        self._generated = generated
        self._low_res_gen = low_res

    def _classification_loss(self, target):
        tf.summary.image("target", target)
        xent = self._xent(target, self._generated)

        mse = self._mse(target, self._generated)
        # multiscale_loss(self._gen_image, target, self.model.config.multi_scale_depth,
        #                tf.losses.mean_squared_error)

        tf.summary.scalar("xent/", xent)
        tf.summary.scalar("mse/", mse)
        self._evals["xent"] = tf.metrics.mean(xent)
        self._evals["mse"] = tf.metrics.mean(mse)

        return self.model.config.mse_weight * mse + self.model.config.xent_weight * xent

    def _mse(self, labels, image_logits):
        white = tf.reduce_mean(labels)
        black = 1 - white
        ft = tf.layers.flatten(labels)
        return tf.losses.mean_squared_error(tf.layers.flatten(labels),
                                            tf.layers.flatten(tf.nn.sigmoid(image_logits)),
                                            reduction=tf.losses.Reduction.MEAN,
                                            weights=ft / white + (1 - ft) / black)

    def _xent(self, labels, image_logits):
        white = tf.reduce_mean(labels)
        black = 1 - white
        ft = tf.layers.flatten(labels)
        xent = tf.losses.sigmoid_cross_entropy(ft, tf.layers.flatten(image_logits),
                                               label_smoothing=0.01, reduction=tf.losses.Reduction.MEAN,
                                               weights=ft / white + (1 - ft) / black)
        return xent

    def model_fn(self, features, labels, mode):
        tf.summary.image("original", features["A/original"])
        # build generator
        with tf.variable_scope("UNet"):
            self._generate(features["A/image"])

        # prediction dict
        self._predictions = {"generated_soft": self._gen_image, "generated_segmentation": self._bin_image}

        # build loss
        if mode != tf.estimator.ModeKeys.PREDICT:
            # normal unet loss
            with tf.name_scope("unet_loss"):
                classify_loss = self._classification_loss(features["B/image"])

            if mode == tf.estimator.ModeKeys.EVAL:
                self._evals["pr_curve"] = summary_lib.pr_curve_streaming_op(name='pr_curve',
                                                                            predictions=self._gen_image,
                                                                            labels=features["B/image"],
                                                                            num_thresholds=11)
            total_los = classify_loss

            tf.summary.image("unet_loss/result",
                             tf.concat([self._gen_image, self._gen_image, features["B/image"]], axis=3))
        else:
            total_los = None

        # connected components
        cct = tf.contrib.image.connected_components(tf.less(features["B/image"][:, :, :, 0], 0.5))
        ccr = tf.contrib.image.connected_components(1.0 - self._bin_image[:, :, :, 0])
        tf.summary.image("connected_components_target", tf.cast(cct, tf.float32)[:, :, :, None])
        tf.summary.image("connected_components_result", tf.cast(ccr, tf.float32)[:, :, :, None])

        self._predictions["connected_components"] = ccr

        # GAN loss
        if self.model.config.discrimination_loss > 0.0:
            disc_loss, gan_loss = self._gan_loss(features, mode != tf.estimator.ModeKeys.PREDICT)

            if mode != tf.estimator.ModeKeys.PREDICT:
                total_los = classify_loss + gan_loss

        iou = iou_from_intersection_op(cct, ccr)
        tf.summary.scalar("iou", tf.reduce_mean(iou))

        if mode == tf.estimator.ModeKeys.EVAL:
            self._evals["iou"] = tf.metrics.mean(tf.reduce_mean(iou))

        if mode == tf.estimator.ModeKeys.TRAIN:
            # boundaries = [2000, 5000]
            # values = [1e-3, 5e-4, 1e-4]
            learning_rate = 1e-3  # tf.train.piecewise_constant(tf.train.get_or_create_global_step(), boundaries, values)
            g_optimizer = tf.train.AdamOptimizer(learning_rate)
            train_step = g_optimizer.minimize(total_los,
                                              global_step=tf.train.get_or_create_global_step(),
                                              var_list=tf.trainable_variables("UNet"))
            # discriminator update
            if self.model.config.discrimination_loss > 0.0:
                # only start training the discriminator after we have done some UNet training
                lr = tf.train.piecewise_constant(tf.train.get_or_create_global_step(),
                                                 [250], [0.0, self.model.config.disc_lr])
                if self.model.config.disc_opt.upper() == "ADAM":
                    d_optimizer = tf.train.AdamOptimizer(lr)
                elif self.model.config.disc_opt.upper() == "SGD":
                    d_optimizer = tf.train.MomentumOptimizer(lr, momentum=0.9)
                else:
                    raise NotImplementedError()
                g_step = train_step
                d_step = d_optimizer.minimize(disc_loss,
                                              global_step=None,
                                              var_list=tf.trainable_variables("Discriminator"))
                disc_up_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, "Discriminator")
                train_step = tf.group((g_step, d_step) + tuple(disc_up_ops))
        else:
            train_step = None
        return tf.estimator.EstimatorSpec(mode=mode, predictions=self._predictions,
                                          loss=total_los, eval_metric_ops=self._evals,
                                          train_op=train_step)

    def _gan_loss(self, features, train_or_eval):
        disc = make_discriminator(3, "channels_first")
        with tf.variable_scope("Discriminator"):
            gfinal, gfeatures = disc(self._gen_image, train_or_eval)
        pgen = tf.nn.sigmoid(gfinal)
        self._predictions["p_gen"] = pgen

        # we also want to add the image gradient of the discriminator to the prediction results.
        with tf.variable_scope("Discriminator", reuse=True):
            rfinal, rfeatures = disc(features["B/image"], train_or_eval)

        gen_loss = generation_loss(fake_logits=gfinal, fake_features=gfeatures, real_features=rfeatures,
                                   feature_matching_weight=self.model.config.feature_matching,
                                   scope="GeneratorLoss")

        # image change
        image_gradient = tf.gradients(gen_loss, self._gen_image)[0]
        self._predictions["GAN_gradient"] = image_gradient
        tf.summary.image("DiscLoss/ImageGradient", image_gradient)
        vis = tf.concat([image_gradient / tf.reduce_max(tf.abs(image_gradient), [1, 2, 3], keepdims=True),
                         self._gen_image, features["B/image"]], axis=3)
        tf.summary.image("DiscLoss/UpdatedImage", vis)
        self._predictions["GAN_vis"] = vis

        if train_or_eval:
            disc_loss = discrimination_loss(logits_fake=gfinal,
                                            logits_real=rfinal,
                                            noise=self.model.config.disc_noise,
                                            scope="DiscriminatorLoss")

            # summaries / metrics
            self._evals["p_generated"] = tf.metrics.mean(pgen)
            self._evals["p_real"] = tf.metrics.mean(tf.nn.sigmoid(rfinal))

            # do not do any generator training for the first 500 steps
            gen_factor = tf.cond(tf.less(tf.train.get_or_create_global_step(), 500),
                                 lambda: tf.constant(0.0),
                                 lambda: tf.constant(1.0))
            gan_loss = gen_factor * gen_loss * self.model.config.discrimination_loss

            tf.summary.scalar("GeneratorLoss/effective", gan_loss)

            self._evals["generator_loss/raw"] = tf.metrics.mean(gen_loss)
            self._evals["generator_loss/effective"] = tf.metrics.mean(gan_loss)
            self._evals["discriminator_loss"] = tf.metrics.mean(disc_loss)
        else:
            disc_loss = None
            gan_loss = None
        return disc_loss, gan_loss

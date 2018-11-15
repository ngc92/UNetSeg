import os, json
import numpy as np
from unet.cell_segmentation import CellSegmentationConfig, CellSegmentationModel, InputConfig
from collections import namedtuple
from easydict import EasyDict

ExpConfig = namedtuple("ExpConfig", ["name", "input", "model", "steps_per_eval"])

# GLOBAL PARAMS
BATCH_SIZE = 16
SCALE_FACTOR = 2
CROP_SIZE = 256
NUM_STEPS = 25500
NUM_RUNS = 10
STEPS_PER_EVAL = 500


# define a set of configurations that will be run
def input_cfg(local_aug, blurring, data_amount=-1):
    return InputConfig(BATCH_SIZE, local_aug, SCALE_FACTOR, CROP_SIZE, blurring, data_amount)


def seg_cfg(resize=False, num_layers=4, xent_w=1.0, mse_w=0.0, multiscale=1, disc_w=0.0,
            fm=0.9, disc_lr=5e-4, disc_opt="ADAM", disc_noise=None, discriminator="make_discriminator"):
    generator_config = EasyDict(layers=num_layers, features=32, data_format="channels_first", use_upsampling=resize)
    disc_config = EasyDict(Discriminator=discriminator, layers=3, data_format="channels_first")
    return CellSegmentationConfig(generator_config, disc_config, xent_w, mse_w, multiscale, disc_w, fm, disc_lr,
                                  NUM_STEPS, disc_opt, disc_noise)


good_input = input_cfg(True, True)


def FExpConfig(name, model):
    return ExpConfig(name, good_input, model, steps_per_eval=STEPS_PER_EVAL)


configurations = [
    # use a standard UNet with fixed parameters and vary only the input preprocessing
    ExpConfig("simple_no_extra_aug", input_cfg(False, False), seg_cfg(), STEPS_PER_EVAL),
    FExpConfig("simple", seg_cfg()),

    # test different network sizes and loss functions
    FExpConfig("upscaling", seg_cfg(resize=True)),
    FExpConfig("depth3", seg_cfg(num_layers=3)),
    FExpConfig("depth3-ups", seg_cfg(num_layers=3, resize=True)),
    FExpConfig("mse", seg_cfg(xent_w=0.0, mse_w=1.0)),

    # gan configs: vary depths and prefactor
    FExpConfig("gan_3_w_0.25", seg_cfg(num_layers=3, resize=True, disc_w=0.25, disc_lr=5e-5)),
    FExpConfig("gan_4_w_0.25", seg_cfg(num_layers=4, resize=True, disc_w=0.25, disc_lr=5e-5)),

    FExpConfig("gan_3_w_0.15", seg_cfg(num_layers=3, resize=True, disc_w=0.15, disc_lr=5e-5)),
    FExpConfig("gan_4_w_0.15", seg_cfg(num_layers=4, resize=True, disc_w=0.15, disc_lr=5e-5)),

    FExpConfig("gan_3_w_0.33", seg_cfg(num_layers=3, resize=True, disc_w=0.33, disc_lr=5e-5)),
    FExpConfig("gan_4_w_0.33", seg_cfg(num_layers=4, resize=True, disc_w=0.33, disc_lr=5e-5)),

    # vary discriminator learning rate
    FExpConfig("gan_3_slow", seg_cfg(num_layers=3, resize=True, disc_w=0.25, disc_lr=1e-5)),
    FExpConfig("gan_3_sgd", seg_cfg(num_layers=3, resize=True, disc_w=0.25, disc_lr=1e-3, disc_opt="SGD")),
    FExpConfig("gan_3_fast", seg_cfg(num_layers=3, resize=True, disc_w=0.25, disc_lr=1e-4)),

    # show utility of feature matching
    FExpConfig("gan_3_no_fm", seg_cfg(num_layers=3, resize=True, disc_w=0.25, disc_lr=5e-5, disc_opt="ADAM", fm=0.0)),

    # conditioned discriminator
    FExpConfig("gan_4_cd", seg_cfg(num_layers=4, resize=True, disc_w=0.25, disc_lr=5e-5,
                                   discriminator="make_conditioned_discriminator")),

    ExpConfig("less_simple", input_cfg(True, True, 20), seg_cfg(), 10 * STEPS_PER_EVAL),
    ExpConfig("less_gan", input_cfg(True, True, 20), seg_cfg(resize=True, disc_w=0.25, disc_lr=5e-5, disc_opt="ADAM"),
              10 * STEPS_PER_EVAL),
]


def train_model(config: ExpConfig, model: CellSegmentationModel):
    # if not yet trained: train model
    while model.global_step < NUM_STEPS:
        print(config.name + " steps: ", model.global_step)
        next_stepcount = np.ceil((1.0 + model.global_step) / config.steps_per_eval) * config.steps_per_eval
        model.train("train.tfrecords", num_steps=int(next_stepcount) - model.global_step)
        model.eval("eval.tfrecords")


def predict_with_model(config, model: CellSegmentationModel):
    # if not yet existing: make predictions
    result_dir = os.path.join("result", config.name)
    if not os.path.exists(result_dir):
        pred = model.predict("eval.tfrecords")
        os.makedirs(result_dir)
        for p in pred:
            import scipy.misc

            name = p["key"].decode("utf-8")
            scipy.misc.imsave(os.path.join(result_dir, name + ".png"),
                              p["generated_soft"][:, :, 0])
            scipy.misc.imsave(os.path.join(result_dir, name + "_seg.png"),
                              p["connected_components"])

            if "GAN_gradient" in p:
                scipy.misc.imsave(os.path.join(result_dir, name + "_gg.png"),
                                  p["GAN_gradient"][:, :, 0])
                scipy.misc.imsave(os.path.join(result_dir, name + "_gv.png"),
                                  p["GAN_vis"])


for i in range(NUM_RUNS):
    for config in configurations:
        model = CellSegmentationModel(os.path.join("ckp", config.name, str(i)), config.input, config.model)
        train_model(config, model)
        if i == 0:
            predict_with_model(config, model)

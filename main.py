import os, json
import numpy as np
from unet.cell_segmentation import CellSegmentationConfig, CellSegmentationModel, InputConfig
from collections import namedtuple

from unet.summary import get_scalars_from_event

ExpConfig = namedtuple("ExpConfig", ["name", "input", "model", "epochs_per_eval"])

# GLOBAL PARAMS
BATCH_SIZE = 16
SCALE_FACTOR = 2
CROP_SIZE = 256
NUM_STEPS = 20500
NUM_RUNS = 5


# define a set of configurations that will be run
def input_cfg(local_aug, blurring, data_amount=-1):
    return InputConfig(BATCH_SIZE, local_aug, SCALE_FACTOR, CROP_SIZE, blurring, data_amount)


def seg_cfg(resize=False, num_layers=4, xent_w=1.0, mse_w=0.0, multiscale=1, disc_w=0.0,
            fm=0.9, disc_lr=5e-4, disc_opt="ADAM", disc_noise=None):
    return CellSegmentationConfig(num_layers, 32, resize, xent_w, mse_w, multiscale, disc_w, fm, disc_lr,
                                  NUM_STEPS, disc_opt, disc_noise)


good_input = input_cfg(True, True)


configurations = [
    # use a standard UNet with fixed parameters and vary only the input preprocessing
    ExpConfig("simple_no_extra_aug", input_cfg(False, False), seg_cfg(), 50),
    ExpConfig("simple", input_cfg(True, True), seg_cfg(), 50),

    # test different network sizes and loss functions
    ExpConfig("upscaling", input_cfg(True, True), seg_cfg(resize=True), 50),
    ExpConfig("depth3", good_input, seg_cfg(num_layers=3), 50),
    ExpConfig("depth3-ups", good_input, seg_cfg(num_layers=3, resize=True), 50),
    ExpConfig("mse", good_input, seg_cfg(xent_w=0.0, mse_w=1.0), 50),

    # gan configs: vary depths and prefactor
    ExpConfig("gan_3_w_0.25", good_input, seg_cfg(num_layers=3, resize=True, disc_w=0.25, disc_lr=5e-5), 50),
    ExpConfig("gan_4_w_0.25", good_input, seg_cfg(num_layers=4, resize=True, disc_w=0.25, disc_lr=5e-5), 50),

    ExpConfig("gan_3_w_0.15", good_input, seg_cfg(num_layers=3, resize=True, disc_w=0.15, disc_lr=5e-5), 50),
    ExpConfig("gan_4_w_0.15", good_input, seg_cfg(num_layers=4, resize=True, disc_w=0.15, disc_lr=5e-5), 50),

    ExpConfig("gan_3_w_0.33", good_input, seg_cfg(num_layers=3, resize=True, disc_w=0.33, disc_lr=5e-5), 50),
    ExpConfig("gan_4_w_0.33", good_input, seg_cfg(num_layers=4, resize=True, disc_w=0.33, disc_lr=5e-5), 50),

    # vary discriminator learning rate
    ExpConfig("gan_3_slow", good_input, seg_cfg(num_layers=3, resize=True, disc_w=0.25, disc_lr=1e-5), 50),
    ExpConfig("gan_3_sgd", good_input, seg_cfg(num_layers=3, resize=True, disc_w=0.25, disc_lr=1e-3, disc_opt="SGD"), 50),
    ExpConfig("gan_3_fast", good_input, seg_cfg(num_layers=3, resize=True, disc_w=0.25, disc_lr=1e-4), 50),

    # show utility of feature matching
    ExpConfig("gan_3_no_fm", good_input, seg_cfg(num_layers=3, resize=True, disc_w=0.25, disc_lr=5e-5, disc_opt="ADAM",
                                                 fm=0.0), 50),


    ExpConfig("less_simple", input_cfg(True, True, 20), seg_cfg(), 250),
    ExpConfig("less_gan", input_cfg(True, True, 20), seg_cfg(resize=True, disc_w=0.25, disc_lr=5e-5, disc_opt="ADAM"), 250),
    #ExpConfig("less_disc_0.1_up", input_cfg(True, True, 20), seg_cfg(disc_w=0.1, resize=True), 250),
]


def train_model(config, model: CellSegmentationModel):
    # if not yet trained: train model
    while model.global_step < NUM_STEPS:
        print(config.name + " steps: ", model.global_step)
        model.train("train.tfrecords", reps=int(config.epochs_per_eval * (0.2 * np.random.rand() + 0.8)) + 1)
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

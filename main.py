import os, json
import numpy as np
from unet.cell_segmentation import CellSegmentationConfig, CellSegmentationModel, InputConfig
from collections import namedtuple
from easydict import EasyDict

ExpConfig = namedtuple("ExpConfig", ["name", "input", "model", "steps_per_eval"])

TRAIN_FILES = ["abdomen1.tfrecords", "abdomen2.tfrecords", "abdomen3.tfrecords"]
EVAL_FILES = ["wingdisk.tfrecords"]

# GLOBAL PARAMS
DEFAULT_BATCH_SIZE = 16
SCALE_FACTOR = 2
CROP_SIZE = 256
NUM_STEPS = 25500
NUM_RUNS = 10
STEPS_PER_EVAL = 500


# define a set of configurations that will be run
def input_cfg(local_aug=True, blurring=True, num_samples=-1, batch_size=DEFAULT_BATCH_SIZE):
    return InputConfig(batch_size, local_aug, SCALE_FACTOR, CROP_SIZE, blurring, num_samples)


def seg_cfg(resize=False, num_layers=4, xent_w=1.0, mse_w=0.0, multiscale=1, disc_w=0.0,
            fm=0.9, disc_lr=5e-4, disc_opt="ADAM", disc_noise=None, discriminator="make_discriminator"):
    generator_config = EasyDict(layers=num_layers, features=32, data_format="channels_first", use_upsampling=resize)
    disc_config = EasyDict(Discriminator=discriminator, layers=3, data_format="channels_first")
    return CellSegmentationConfig(generator_config, disc_config, xent_w, mse_w, multiscale, disc_w, fm, disc_lr,
                                  NUM_STEPS, disc_opt, disc_noise)


good_input = input_cfg(True, True)


def FExpConfig(name, model):
    return ExpConfig(name, good_input, model, steps_per_eval=STEPS_PER_EVAL)


def load_config(filename):
    with open(filename, "r") as file:
        configs = json.load(file)

    setting_list = []
    for key in configs:
        data = configs[key]
        model_data = data.get("model", {})
        input_data = data.get("data", {})
        setting_list.append(ExpConfig(key, input_cfg(**input_data), seg_cfg(**model_data), steps_per_eval=STEPS_PER_EVAL))

    return setting_list


configurations = load_config("configs.json")


def train_model(config: ExpConfig, model: CellSegmentationModel):
    # if not yet trained: train model
    while model.global_step < NUM_STEPS:
        print(config.name + " steps: ", model.global_step)
        next_stepcount = np.ceil((1.0 + model.global_step) / config.steps_per_eval) * config.steps_per_eval
        model.train(TRAIN_FILES, num_steps=int(next_stepcount) - model.global_step)
        model.eval(EVAL_FILES)


def predict_with_model(config, model: CellSegmentationModel):
    # if not yet existing: make predictions
    result_dir = os.path.join("result", config.name)
    if not os.path.exists(result_dir):
        pred = model.predict(EVAL_FILES)
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
